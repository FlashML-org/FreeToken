from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.attention.base import AttentionSpec, BaseAttnBackend

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.kvcache.linear_state_pool import LinearStatePool, LinearVerifyScratch


@dataclass
class FLAMetadata:
    """Per-forward GatedDeltaNet (flash-linear-attention) metadata, built once per
    forward and shared by every GDN layer -- mirrors ``BaseAttnMetadata``. Replaces the
    per-layer rebuilds the GDN op used to do (``cu_seqlens`` arange, per-request
    ``cache_indices``/``has_initial_state``), which were pageable, synchronous H2D copies
    issued in each of the 30 GDN layers.

    Fields:
      cu_seqlens          query indptr; decode = arange(bs+1) (1 token/req), prefill =
                          cumsum of extend_len. int32 on device.
      cache_indices       per-request recurrent/conv state slot (Req.linear_slot_idx, else
                          Req.table_idx). int32.
      has_initial_state   prefill only: whether each request continues a cached prefix
                          (cached_len > 0). None for decode (state always present).
      fresh_state_indices prefill only: the state-pool slots whose sequence is fresh
                          (cached_len == 0) and must be zeroed before the chunk kernel
                          reads them in place. None if there are none / for decode.
    """

    cu_seqlens: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor | None = None
    fresh_state_indices: torch.Tensor | None = None
    # verify only: the state pool's scratch the verify writes (row = batch row) and the commit replays
    verify_scratch: "LinearVerifyScratch | None" = None

    # --- hybrid-radix track-checkpoint (extra_buffer) fields; all None when not caching ---
    # For each request crossing a chunk-aligned (×CHUNK) boundary this forward, snapshot its
    # recurrent + conv state into a donatable pool slot, written on the forward stream by the
    # GDN op (see GatedDeltaNet._write_track_snapshot). Built by the scheduler in P2;
    # left None by build_fla_metadata so the existing path is unchanged.
    track_dst: torch.Tensor | None = None        # [nt] int64 dst pool slot per tracked req
    track_h_row: torch.Tensor | None = None      # [nt] int64 row into h (boh_i + aligned//CHUNK)
    track_conv_src: torch.Tensor | None = None   # [nt, kernel-1] int64 conv-input token positions
    track_boundary_row: torch.Tensor | None = None  # [nt] int64 forward-local row of the track boundary; states with their own left context (qwen4_exp PLE) derive their windows from it


def gdn_slot(req) -> int:
    """The request's GDN state slot: its hybrid-radix live slot (decoupled from table_idx) when allocated, else table_idx (naive GDN keeps the old keying)."""
    return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx


def build_fla_metadata(batch: "Batch", device: torch.device, pool: LinearStatePool) -> FLAMetadata:
    """Build the per-forward GDN metadata. Uses pinned host staging + non_blocking H2D
    (the input_ids/attn-metadata pattern), so the copies overlap the forward instead of
    stalling it.

    Decode and verify run ``rows_per_req`` rows per request, so ``cu_seqlens`` is
    ``arange(bs+1) * rows_per_req``. Under CUDA graph ``LinearAttnBackend`` copies this
    metadata into its persistent buffers (stable addresses) before each replay.
    """
    reqs = batch.padded_reqs
    pin = {"device": "cpu", "pin_memory": torch.cuda.is_available()}

    if not batch.is_prefill:
        # every request continues its live state
        bs = len(reqs)
        idx_host = torch.tensor([gdn_slot(r) for r in reqs], dtype=torch.int32, **pin)
        metadata = FLAMetadata(
            cu_seqlens=torch.arange(0, (bs + 1) * batch.rows_per_req, batch.rows_per_req, dtype=torch.int32, device=device),
            cache_indices=idx_host.to(device, non_blocking=True),
        )
        if batch.phase == "verify":
            metadata.verify_scratch = pool.verify_scratch
        return metadata

    # prefill: cumsum of query (extend) lengths, per-request slot + continuation flags.
    lens = [r.extend_len for r in reqs]
    cu_host = torch.tensor([0, *lens], dtype=torch.int64, **pin).cumsum_(0)
    idx_host = torch.tensor([gdn_slot(r) for r in reqs], dtype=torch.int32, **pin)
    has_init_host = torch.tensor([r.cached_len > 0 for r in reqs], dtype=torch.bool, **pin)
    fresh = [gdn_slot(r) for r in reqs if r.cached_len == 0]
    fresh_host = torch.tensor(fresh, dtype=torch.int64, **pin) if fresh else None

    track = _build_track_metadata(reqs, cu_host, device, pin, pool.conv_states.shape[-1])

    return FLAMetadata(
        cu_seqlens=cu_host.to(device, non_blocking=True),
        cache_indices=idx_host.to(device, non_blocking=True),
        has_initial_state=has_init_host.to(device, non_blocking=True),
        fresh_state_indices=(
            fresh_host.to(device, non_blocking=True) if fresh_host is not None else None
        ),
        **track,
    )


def _build_track_metadata(reqs, cu_host, device, pin, km1: int):
    """Hybrid-radix (extra_buffer): for each request that crosses a ×CHUNK boundary this
    prefill forward, snapshot its GDN state at the deepest mid-chunk boundary into its current
    ping-pong slot. Returns the ``FLAMetadata`` track kwargs, all None when no request
    tracks (non-hybrid, or all extends < CHUNK+1)."""
    empty = dict(track_dst=None, track_h_row=None, track_conv_src=None, track_boundary_row=None)
    if not any(r.mamba_ping_pong is not None for r in reqs):
        return empty
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kernel.fla.index import prepare_chunk_offsets

    assert km1 <= CHUNK_SIZE, (
        f"conv history {km1} exceeds CHUNK_SIZE {CHUNK_SIZE}: the snapshot window "
        "would reach before this forward's first token"
    )
    boh = prepare_chunk_offsets(cu_host, CHUNK_SIZE).tolist()
    dst, h_row, conv_src, boundary_rows = [], [], [], []
    for i, r in enumerate(reqs):
        if r.mamba_ping_pong is None:
            continue
        # deepest mid-chunk boundary strictly inside the extend (h has the per-chunk state;
        # the exact extend-end / aligned-final state lives in the live slot -> finish-donate).
        c = (r.extend_len - 1) // CHUNK_SIZE
        if c < 1:
            continue
        off = int(cu_host[i])
        boundary = r.cached_len + c * CHUNK_SIZE
        dst.append(r.mamba_ping_pong[r.mamba_next_track_idx])
        h_row.append(boh[i] + c)
        conv_src.append([off + c * CHUNK_SIZE - km1 + j for j in range(km1)])
        boundary_rows.append(off + c * CHUNK_SIZE)
        r.mamba_last_track_seqlen = boundary
        r.mamba_next_track_idx = 1 - r.mamba_next_track_idx
    if not dst:
        return empty
    to = lambda xs, **kw: torch.tensor(xs, **pin, **kw).to(device, non_blocking=True)
    return dict(
        track_dst=to(dst, dtype=torch.int64),
        track_h_row=to(h_row, dtype=torch.int64),
        track_conv_src=to(conv_src, dtype=torch.int64),
        track_boundary_row=to(boundary_rows, dtype=torch.int64),
    )


@dataclass
class FLACaptureData:
    """The static buffers the linear-attention layers' CUDA graphs read, sized for the largest captured batch."""

    cu_seqlens: torch.Tensor
    cache_indices: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, device: torch.device, rows_per_req: int = 1) -> FLACaptureData:
        return cls(
            # constant under a graph: every request has exactly `rows_per_req` query rows
            cu_seqlens=torch.arange(max_bs + 1, dtype=torch.int32, device=device) * rows_per_req,
            cache_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )


class LinearAttnBackend(BaseAttnBackend):
    """Metadata of the linear-attention layers (GatedDeltaNet, KDA), next to the attention backend: ``FLAMetadata`` per forward, ``FLACaptureData`` for CUDA graphs.

    The layers run their own kernels and read ``batch.fla_metadata``, so ``forward`` is never called.
    """

    def __init__(self, device: torch.device, pool: LinearStatePool) -> None:
        self.device = device
        self.pool = pool
        self.capture: FLACaptureData | None = None
        self.capture_bs: List[int] = []

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: "Batch",
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        # the GDN and KDA layers call their fla kernels themselves on batch.fla_metadata
        raise NotImplementedError("linear-attention layers run their own kernels")

    def prepare_metadata(self, batch: "Batch") -> None:
        batch.fla_metadata = build_fla_metadata(batch, self.device, self.pool)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], rows_per_req: int = 1) -> None:
        assert self.capture is None, "Capture already initialized."
        self.capture = FLACaptureData.create(max(bs_list), self.device, rows_per_req=rows_per_req)
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: "Batch") -> None:
        assert (bs := batch.padded_size) in self.capture_bs and self.capture
        capture = self.capture
        # capture on the padding request's state slot, so the captured gather/scatter touches scratch; replay writes the real slots
        capture.cache_indices[:bs].fill_(gdn_slot(batch.padded_reqs[0]))
        batch.fla_metadata = FLAMetadata(cu_seqlens=capture.cu_seqlens[: bs + 1], cache_indices=capture.cache_indices[:bs])
        if batch.phase == "verify":
            batch.fla_metadata.verify_scratch = self.pool.verify_scratch

    def prepare_for_replay(self, batch: "Batch") -> None:
        metadata, bs = batch.fla_metadata, batch.padded_size
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens is the constant arange * rows_per_req under a graph (i.e. no-op)
        self.capture.cache_indices[:bs].copy_(metadata.cache_indices)


__all__ = ["FLACaptureData", "FLAMetadata", "LinearAttnBackend", "build_fla_metadata"]
