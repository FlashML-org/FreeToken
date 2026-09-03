"""Per-Layer Embedding (PLE) for Qwen3.8-Flash-Next: hashed n-gram features injected at layer 1.

HF reference: ``Qwen4ExpTextNGramEmbedding`` (modeling_qwen4_exp.py:1018) and
``Qwen4ExpTextPLELayer`` (:1117). Per token::

    E = table[hash(ngram)]                       # 16 heads (8 x 2-gram, 8 x 3-gram) x 160 -> 2560
    K = norm_key(key_proj(E)).view(hc, hidden)   # V = value_proj(E) [hidden]
    Q = norm_query(R).view(hc, hidden)
    u = <K_i, Q_i> / sqrt(hidden)                # per stream
    U = sigmoid(sign(u) * sqrt(max(|u|, 1e-6))) * V
    D = U + silu(conv1d(norm_conv(U)))           # depthwise, kernel 4, dilation ngram_size
    R += D                                       # before the attention hyper-connection mix

The table is the 47.7 GiB FP8 n-gram store: ``PinnedUVATable`` keeps it in pinned host memory and
gathers rows over UVA, optionally started early on a side stream (``PLELayer.start_prefetch``).
``GpuResidentTable`` is the small-table oracle the pinned backend is diffed against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Sequence, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated

from .config import PLE_CONV_STATE, PLE_NGRAM_STATE
from .hc import GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

    from .config import Qwen4ExpArgs


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


class PLETableBackend(Protocol):
    """Row store for one PLE layer's n-gram embedding table (Qwen3.8: 40M rows x 160, FP8 + one scalar scale).

    Frozen contract. ``GpuResidentTable`` (oracle, small tables) and ``PinnedUVATable`` (the real 47.7 GiB pinned-host table) implement it. Rows are addressed by the
    GLOBAL hashed id, i.e. the per-head vocab offset is already added by ``NGramEmbedding``.

    ``lookup`` gets ``row_ids [T, num_ngram_heads]`` (int64, device) and returns
    ``[T, num_ngram_heads * head_dim]`` in ``dtype``, already dequantized (fp8 -> dtype, times the
    scalar weight_scale). ``out``, when given, is the destination and is returned as-is (CUDA-graph
    decode reuses a fixed buffer).

    ``prefetch`` may start the gather early on a side stream (the model issues it before layer 0 and
    joins it in ``lookup``); a backend with no async path makes it a no-op.
    """

    num_rows: int
    head_dim: int
    dtype: torch.dtype

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor: ...

    def prefetch(self, row_ids: torch.Tensor) -> None: ...


class GpuResidentTable:
    """PLE table held whole in GPU memory; ``index_select`` oracle for the pinned-host backend."""

    def __init__(
        self, weight: torch.Tensor, scale: float = 1.0, dtype: torch.dtype | None = None
    ) -> None:
        self.weight = weight
        self.scale = float(scale)
        self.num_rows, self.head_dim = weight.shape
        self.dtype = dtype if dtype is not None else (
            torch.bfloat16 if weight.dtype.itemsize < 2 else weight.dtype
        )

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = self.weight.index_select(0, row_ids.reshape(-1)).to(self.dtype)
        if self.scale != 1.0:
            rows = rows * self.scale
        rows = rows.view(*row_ids.shape[:-1], -1)
        if out is None:
            return rows
        out.copy_(rows)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None


class ZeroTable:
    """Dummy-weight stand-in: every lookup reads zeros (dummy checkpoints ship no table)."""

    def __init__(self, num_rows: int, head_dim: int, dtype: torch.dtype = torch.bfloat16) -> None:
        self.num_rows = int(num_rows)
        self.head_dim = head_dim
        self.dtype = dtype

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is not None:
            return out.zero_()
        return torch.zeros(
            (*row_ids.shape[:-1], row_ids.shape[-1] * self.head_dim),
            dtype=self.dtype,
            device=row_ids.device,
        )

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None


class PinnedUVATable:
    """PLE table left in pinned host memory; rows are gathered over UVA by a Triton kernel.

    ``weight`` must be the filled and ``pin()``ed ``HostBank.tensor`` from
    ``weight.load_ple_table`` (``[num_rows, head_dim]``, fp8-e4m3 or bf16); an unregistered host
    buffer is not device-addressable and the kernel faults on it. ``scale`` is the checkpoint's
    scalar ``weight_scale``. Gathers emit bf16 into a staging buffer, one per captured decode size
    and one growable buffer for everything else.

    ``prefetch`` runs the gather on a private stream and the next ``lookup`` joins it. ``lookup``
    returns a view of that staging buffer, so the rows must be consumed before the next lookup.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scale: float = 1.0,
        *,
        device: torch.device | None = None,
        prefetch: bool = True,
    ) -> None:
        assert weight.device.type == "cpu" and weight.is_contiguous()
        assert weight.dtype in (torch.float8_e4m3fn, torch.bfloat16), weight.dtype
        from freetoken.kernel.pinned import device_ptr

        self.weight = weight
        self.scale = float(scale)
        self.num_rows, self.head_dim = weight.shape
        self.dtype = torch.bfloat16
        self._is_fp8 = weight.dtype == torch.float8_e4m3fn
        self._device = device or torch.device("cuda", torch.cuda.current_device())
        # WDDM maps registered host memory at a different device address; on Linux/UVA this is data_ptr
        self._table_ptr = device_ptr(weight)
        self._stream = torch.cuda.Stream(device=self._device) if prefetch else None
        self._staging: torch.Tensor | None = None
        self._graph_staging: dict[int, torch.Tensor] = {}
        self._pending: Tuple[torch.Tensor, torch.Tensor] | None = None

    def _stage(self, rows: int) -> torch.Tensor:
        # Captured graphs keep one buffer per size for good: growing the eager one would free the
        # block a replay still writes to.
        if torch.cuda.is_current_stream_capturing():
            buf = self._graph_staging.get(rows)
            if buf is None:
                buf = torch.empty((rows, self.head_dim), dtype=self.dtype, device=self._device)
                self._graph_staging[rows] = buf
            return buf
        buf = self._staging
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((rows, self.head_dim), dtype=self.dtype, device=self._device)
            self._staging = buf
        return buf[:rows]

    def _gather(self, row_ids: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.ple import ple_gather_rows

        return ple_gather_rows(
            self._table_ptr,
            self.num_rows,
            self.head_dim,
            row_ids.reshape(-1),
            dst,
            self.scale,
            self._is_fp8,
        )

    def prefetch(self, row_ids: torch.Tensor) -> None:
        if self._stream is None or row_ids.numel() == 0:
            return
        dst = self._stage(row_ids.numel())
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        if not torch.cuda.is_current_stream_capturing():
            row_ids.record_stream(self._stream)
        with torch.cuda.stream(self._stream):
            self._gather(row_ids, dst)
        self._pending = (row_ids, dst)

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        pending, self._pending = self._pending, None
        if pending is not None:
            # join even on a miss: the stale prefetch owns the staging buffer about to be reused
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
        if pending is not None and pending[0] is row_ids:
            rows = pending[1]
        else:
            rows = self._gather(row_ids, self._stage(row_ids.numel()))
        rows = rows.view(*row_ids.shape[:-1], -1)
        if out is None:
            return rows
        out.copy_(rows)
        return out


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def derive_ngram_hash_constants(
    *,
    vocab_size: int,
    ngram_size: int,
    num_ngram_heads: int,
    ngram_vocab_size_base: int,
    ple_layer_index: int,
    seed: int = 1234,
) -> Tuple[List[int], List[int], List[int]]:
    """Recompute (multipliers, per-head vocab sizes, per-head offsets) the way HF derives them at init.

    The checkpoint ships these as int64 tensors, so serving loads them; this is the dummy-weight
    path and the oracle a loader test can check the checkpoint values against.
    """
    half_bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(ngram_size)
    ]
    sizes: List[int] = []
    offsets: List[int] = []
    total = 0
    for head in range(num_ngram_heads):
        global_head = ple_layer_index * num_ngram_heads + head
        size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return multipliers, sizes, offsets


@dataclass
class PLEMetadata:
    """Per-forward PLE inputs, built once and shared by every PLE layer (sibling of ``FLAMetadata``).

    Frozen contract:
      input_ids      [T] int device -- this forward's tokens, ragged, concatenated in request order
      cu_seqlens     [B+1] int device -- query indptr; decode is ``arange(B+1)``
      seq_lens       host per-request token counts; avoids a device sync in the ragged conv loop
      ngram_context  [B, ngram_size-1] int64 device -- the tokens immediately BEFORE each request's
                     first token of this forward, read from the ``ple_ngram_ctx`` slot state and
                     forced to the boundary (eos) id for fresh rows. The hash never crosses eos,
                     so a fresh sequence passes all-eos.
      state_slots    [B] int64 device -- linear-state slot per request (``Req.linear_slot_idx`` or
                     ``Req.table_idx``); keys every PLE slot state
      fresh_slots    [B] bool device or None -- request starts a new sequence, so read a zero state
      is_decode      one token per request (the batched 4-tap path)
    """

    input_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens: Sequence[int]
    ngram_context: torch.Tensor
    state_slots: torch.Tensor
    fresh_slots: torch.Tensor | None
    is_decode: bool


def _state_slot(req) -> int:
    slot = getattr(req, "linear_slot_idx", None)
    return req.table_idx if slot is None else slot


def _ngram_context_pool() -> torch.Tensor:
    pool = get_global_ctx().linear_state_pool
    assert pool is not None and pool.has_slot_state(PLE_NGRAM_STATE), (
        "PLE needs the ple_ngram_ctx slot state (or an explicit context_pool=)"
    )
    return pool.slot_state(PLE_NGRAM_STATE)


def build_ple_metadata(
    batch: Batch,
    args: Qwen4ExpArgs,
    device: torch.device,
    context_pool: torch.Tensor | None = None,
) -> PLEMetadata:
    """Build ``PLEMetadata`` from a scheduler batch.

    The n-gram context is per-request device state (``ple_ngram_ctx`` [num_slots, ngram_size-1],
    rolled forward once per forward by ``commit_ngram_context``), so it never lags the sampled
    token under overlap scheduling and follows the slot on COW/snapshot. A decode batch reads it
    straight off the persistent ``linear_table_idx`` buffer, so the build is capture-safe and
    sync-free. Reuses ``batch.fla_metadata`` (slots / indptr / fresh mask) when the scheduler
    built it.
    """
    reqs = batch.padded_reqs
    ctx_len = args.ngram_size - 1
    eos = args.ngram_boundary_token_id
    if context_pool is None:
        context_pool = _ngram_context_pool()
    assert context_pool.shape[-1] == ctx_len, (
        f"ple_ngram_ctx holds {context_pool.shape[-1]} ids, config wants {ctx_len}"
    )
    fla = getattr(batch, "fla_metadata", None)
    slots_dev = getattr(batch, "linear_table_idx", None)

    if batch.is_decode and slots_dev is not None:
        slots = slots_dev.long()
        bs = slots.numel()
        return PLEMetadata(
            input_ids=batch.input_ids,
            cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
            seq_lens=(1,) * bs,
            ngram_context=context_pool.index_select(0, slots).long(),
            state_slots=slots,
            fresh_slots=None,
            is_decode=True,
        )

    lens = [r.extend_len for r in reqs]
    if fla is not None and fla.has_initial_state is not None:
        cu = fla.cu_seqlens
        slots = fla.cache_indices.long()
        fresh = ~fla.has_initial_state
    else:  # direct-op callers (tests) with no scheduler metadata
        pin = {"device": "cpu", "pin_memory": torch.cuda.is_available()}
        cu = torch.tensor([0, *lens], dtype=torch.int64, **pin).cumsum_(0).to(device, non_blocking=True)
        slots = torch.tensor([_state_slot(r) for r in reqs], dtype=torch.int64, **pin).to(device, non_blocking=True)
        fresh = torch.tensor([r.cached_len == 0 for r in reqs], dtype=torch.bool, **pin).to(device, non_blocking=True)
    context = context_pool.index_select(0, slots).long()
    context = torch.where(fresh.unsqueeze(1), context.new_full((), eos), context)
    return PLEMetadata(
        input_ids=batch.input_ids,
        cu_seqlens=cu,
        seq_lens=tuple(lens),
        ngram_context=context,
        state_slots=slots,
        fresh_slots=fresh,
        is_decode=batch.is_decode,
    )


def commit_ngram_context(meta: PLEMetadata, fla, context_pool: torch.Tensor | None = None) -> None:
    """Roll each request's ``ple_ngram_ctx`` forward past this forward's tokens.

    Called ONCE per forward after every PLE layer ran (the layers only read the context);
    also writes the boundary-aligned window to the track slot so a donated snapshot restores
    the context together with the conv state. Pure device arithmetic, capture-safe.
    """
    if context_pool is None:
        context_pool = _ngram_context_pool()
    ids = meta.input_ids.long()
    ctx_len = meta.ngram_context.shape[1]
    steps = torch.arange(ctx_len, device=ids.device)
    if meta.is_decode:
        nxt = torch.cat([meta.ngram_context[:, 1:], ids.view(-1, 1)], dim=1)
    else:
        cu = meta.cu_seqlens.long()
        cand = cu[1:].unsqueeze(1) - ctx_len + steps
        # short extends fall back to the old context: token j of the new window sits at
        # old-context column extend_len + j when it predates this forward
        old = meta.ngram_context.gather(
            1, ((cu[1:] - cu[:-1]).unsqueeze(1) + steps).clamp_(max=ctx_len - 1)
        )
        nxt = torch.where(cand >= cu[:-1].unsqueeze(1), ids[cand.clamp_min(0)], old)
    context_pool.index_copy_(0, meta.state_slots, nxt.to(context_pool.dtype))
    if fla is not None and fla.track_boundary_row is not None:
        win = ids[fla.track_boundary_row.unsqueeze(1) - ctx_len + steps]
        context_pool.index_copy_(0, fla.track_dst, win.to(context_pool.dtype))


class NGramEmbedding(BaseOP):
    """Hashed n-gram lookup: splitmix64 mix of the last n token ids -> per-head prime vocab -> table rows.

    Weight keys (checkpoint names): ``layer_multipliers`` [ngram_size], ``ngram_heads_vocab_sizes``
    and ``ngram_heads_offsets`` [num_ngram_heads], all int64. The table itself is NOT a state-dict
    entry (128 checkpoint shards land in a ``PLETableBackend``); attach it with ``attach_table``.
    """

    def __init__(self, args: Qwen4ExpArgs, table: PLETableBackend | None = None) -> None:
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.num_heads = args.num_ngram_heads
        self.eos_token_id = args.ngram_boundary_token_id
        self.layer_multipliers = torch.empty(args.ngram_size, dtype=torch.int64)
        self.ngram_heads_vocab_sizes = torch.empty(self.num_heads, dtype=torch.int64)
        self.ngram_heads_offsets = torch.empty(self.num_heads, dtype=torch.int64)
        self._table = table

    def attach_table(self, table: PLETableBackend) -> None:
        self._table = table

    @property
    def table(self) -> PLETableBackend:
        assert self._table is not None, "PLE table backend was never attached"
        return self._table

    def _window(self, meta: PLEMetadata):
        """The hash window as ``(packed [B, W], select)``, where ``select`` picks this forward's tokens."""
        ids = meta.input_ids.long()
        ctx_len = self.ngram_size - 1
        if meta.is_decode:
            # a window of exactly ngram_size columns holds every shift the hash can reach
            return torch.cat([meta.ngram_context, ids.view(-1, 1)], dim=1), lambda t: t[:, -1]

        num_reqs = len(meta.seq_lens)
        width = ctx_len + max(meta.seq_lens)
        cu = meta.cu_seqlens.long()
        # Pack the ragged tokens into [B, ctx+max_len] so the shift/boundary logic is one gather.
        flat_pos = torch.arange(ids.numel(), device=ids.device)
        req = (torch.searchsorted(cu, flat_pos, right=True) - 1).clamp_(max=num_reqs - 1)
        col = flat_pos - cu[req] + ctx_len
        packed = ids.new_full((num_reqs, width), self.eos_token_id)
        packed[:, :ctx_len] = meta.ngram_context
        packed[req, col] = ids
        return packed, lambda t: t[req, col]

    def _shift_ignore_eos(self, packed: torch.Tensor) -> List[torch.Tensor]:
        """``out[s][b, p]`` = the token ``s`` places left of ``p``, or eos when the window crosses a boundary."""
        num_reqs, width = packed.shape
        pos = torch.arange(width, device=packed.device)
        eos_pos = torch.where(packed == self.eos_token_id, pos, -1)
        prev_eos = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat([eos_pos.new_full((num_reqs, 1), -1), prev_eos[:, :-1]], dim=1)
        in_segment = pos.unsqueeze(0) - prev_eos - 1

        shifted = [packed]
        for shift in range(1, self.ngram_size):
            src = pos - shift
            gathered = packed.gather(1, src.clamp_min(0).unsqueeze(0).expand(num_reqs, -1))
            valid = (src.unsqueeze(0) >= 0) & (in_segment >= shift)
            shifted.append(torch.where(valid, gathered, packed.new_full((), self.eos_token_id)))
        return shifted

    def row_ids(self, meta: PLEMetadata) -> torch.Tensor:
        """Global table row per (token, hash head): ``[T, num_ngram_heads]`` int64."""
        packed, select = self._window(meta)
        tokens = [select(s) for s in self._shift_ignore_eos(packed)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, tokens[position] * self.layer_multipliers[position])
            head_ids = torch.remainder(mixed.unsqueeze(-1), self.ngram_heads_vocab_sizes[start:end])
            blocks.append(head_ids + self.ngram_heads_offsets[start:end])
        return torch.cat(blocks, dim=-1)

    def forward(self, meta: PLEMetadata, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.table.lookup(self.row_ids(meta), out)


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[width, 1, kernel]`` (key ``conv1d.weight``)."""

    def __init__(self, width: int, kernel: int) -> None:
        self.weight = torch.empty(width, 1, kernel)


def short_conv_reference(
    x: torch.Tensor,
    meta: PLEMetadata,
    states: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Per-request ``F.conv1d`` over ``[state | chunk]``, advancing ``states`` in place.

    Transcription of the HF conv; the shipping paths (one packed conv for prefill, a tap read for
    decode) are diffed against it.
    """
    groups = weight.shape[0]
    state_len = states.shape[-1]
    slots = meta.state_slots
    state = states.index_select(0, slots).to(x.dtype)
    if meta.fresh_slots is not None:
        state = torch.where(meta.fresh_slots.view(-1, 1, 1), torch.zeros_like(state), state)

    outs = []
    new_state = torch.empty_like(state)
    offset = 0
    for i, n in enumerate(meta.seq_lens):
        chunk = x[offset : offset + n].transpose(0, 1).unsqueeze(0)
        history = torch.cat([state[i : i + 1], chunk], dim=-1)
        out = F.conv1d(history, weight, groups=groups, dilation=dilation)
        outs.append(out.squeeze(0).transpose(0, 1))
        new_state[i] = history[0, :, -state_len:]
        offset += n
    states.index_copy_(0, slots, new_state.to(states.dtype))
    return F.silu(torch.cat(outs, dim=0))


class PLELayer(BaseOP):
    """PLE block: hashed n-gram value gated by the residual streams, then a dilated depthwise conv.

    ``forward(R, batch) -> D [T, hc_count*hidden]``; the caller adds ``D`` to ``R`` before the
    attention hyper-connection mix. ``meta`` defaults to ``build_ple_metadata(batch, ...)``;
    ``conv_states`` defaults to ``ctx.linear_state_pool.slot_state("ple_conv", layer_id)`` and is
    ``[num_slots, hc_count*hidden, (ple_conv_kernel_size-1)*ngram_size]`` in the model dtype -- the
    last conv-input columns per request, oldest first. Both are arguments so the reference is
    testable before the pool and the scheduler carry them.

    ``start_prefetch(batch)`` builds the metadata and starts the table gather on the backend's side
    stream; call it at the top of the model forward so the rows land while layer 0 runs, and
    ``forward`` joins it.

    Weight keys (checkpoint names, prefix stripped): ``key_proj.weight`` [hc*hidden, ple_embed_dim],
    ``value_proj.weight`` [hidden, ple_embed_dim], ``norm_key/norm_query/norm_conv.weight``
    [hc*hidden] (zero-centered, loaded RAW), ``conv1d.weight`` [hc*hidden, 1, kernel], plus the
    three ``ple_embedding`` int64 hash buffers.
    """

    def __init__(
        self, config: ModelConfig, layer_id: int, table: PLETableBackend | None = None
    ) -> None:
        args = config.qwen4_args
        self.args = args
        self.layer_id = layer_id
        self.ple_index = args.ple_layer_ids.index(layer_id)
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.dilation = args.ple_conv_dilation
        self.state_len = args.ple_conv_state_len
        width = args.ple_state_width
        self.ple_embedding = NGramEmbedding(args, table)
        self.key_proj = LinearReplicated(args.ple_embed_dim, width, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, args.hidden_size, has_bias=False)
        self.norm_key = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.norm_query = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.norm_conv = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.conv1d = _DepthwiseConv1d(width, args.ple_conv_kernel_size)
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        # the track snapshot gathers the last state_len conv inputs before a xCHUNK boundary; a longer history would reach before the forward's first token
        assert self.state_len <= CHUNK_SIZE, (
            f"PLE conv history {self.state_len} exceeds CHUNK_SIZE {CHUNK_SIZE}"
        )
        self._pending: Tuple[PLEMetadata, torch.Tensor] | None = None

    def start_prefetch(self, batch: Batch, meta: PLEMetadata | None = None) -> None:
        """Hash this forward's n-grams and start the table gather on the side stream."""
        if meta is None:
            meta = build_ple_metadata(batch, self.args, batch.input_ids.device)
        row_ids = self.ple_embedding.row_ids(meta)
        self._pending = (meta, row_ids)
        self.ple_embedding.table.prefetch(row_ids)

    def forward(
        self,
        R: torch.Tensor,
        batch: Batch,
        meta: PLEMetadata | None = None,
        conv_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pending, self._pending = self._pending, None
        row_ids = None
        if meta is None:
            if pending is not None:
                meta, row_ids = pending
            else:
                meta = build_ple_metadata(batch, self.args, R.device)
        elif pending is not None and pending[0] is meta:
            row_ids = pending[1]
        if row_ids is None:
            row_ids = self.ple_embedding.row_ids(meta)

        embeddings = self.ple_embedding.table.lookup(row_ids).to(R.dtype)
        key = self.norm_key.forward(self.key_proj.forward(embeddings))
        value = self.value_proj.forward(embeddings)
        query = self.norm_query.forward(R)
        shape = (-1, self.hc_count, self.hidden_size)
        gate = (key.view(shape) * query.view(shape)).sum(-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = (gate * value.unsqueeze(-2)).flatten(-2)
        states = conv_states if conv_states is not None else self._conv_state_slab(R)
        x = self.norm_conv.forward(gated)
        fla = getattr(batch, "fla_metadata", None)
        if fla is not None and fla.track_boundary_row is not None:
            self._write_track_snapshot(states, x, fla)
        return gated + self._short_conv(x, meta, states)

    def _write_track_snapshot(self, states: torch.Tensor, x: torch.Tensor, fla) -> None:
        """Copy the conv history at the GDN track boundary into the same donatable slot, so a radix
        prefix hit restores PLE and GDN state together. Track slots never alias the live slots this
        forward advances, so the two writes are order-independent."""
        src = fla.track_boundary_row.unsqueeze(1) + torch.arange(
            -self.state_len, 0, device=x.device
        )
        window = x[src].transpose(-1, -2).contiguous()
        states.index_copy_(0, fla.track_dst, window.to(states.dtype))

    def _conv_state_slab(self, R: torch.Tensor) -> torch.Tensor:
        pool = get_global_ctx().linear_state_pool
        assert pool is not None, "PLE needs ctx.linear_state_pool or an explicit conv_states"
        assert pool.has_slot_state(PLE_CONV_STATE), (
            "ModelConfig.slot_states does not declare the PLE conv history"
        )
        return pool.slot_state(PLE_CONV_STATE, self.layer_id)

    def _read_state(
        self, meta: PLEMetadata, states: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        state = states.index_select(0, meta.state_slots).to(dtype)
        if meta.fresh_slots is not None:
            state = torch.where(meta.fresh_slots.view(-1, 1, 1), torch.zeros_like(state), state)
        return state

    def _short_conv(
        self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor
    ) -> torch.Tensor:
        """silu of the dilated depthwise conv over [state | x], and roll the per-request state."""
        if meta.is_decode:
            return self._decode_conv(x, meta, states)
        return self._prefill_conv(x, meta, states)

    def _decode_conv(
        self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor
    ) -> torch.Tensor:
        """Batched tap read: taps t-9, t-6, t-3 come off the state slab, tap t from this token."""
        state = self._read_state(meta, states, x.dtype)
        column = x.unsqueeze(-1)
        # fp32 products, like the conv1d the prefill path runs
        window = torch.cat([state[..., :: self.dilation], column], dim=-1).float()
        out = (window * self.conv1d.weight.squeeze(1).float()).sum(-1)
        states.index_copy_(
            0, meta.state_slots, torch.cat([state[..., 1:], column], dim=-1).to(states.dtype)
        )
        return F.silu(out.to(x.dtype))

    def _prefill_conv(
        self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor
    ) -> torch.Tensor:
        """One conv over every request packed as ``[state_0 | chunk_0 | state_1 | chunk_1 | ...]``.

        The blocks abut exactly, so each output window stays inside its own request: request i's
        first token reads history columns base_i .. base_i+state_len, which is its own state.
        """
        lens = list(meta.seq_lens)
        num_reqs, width = len(lens), x.shape[1]
        out_index, state_index, next_state_index = self._prefill_indices(lens, x.device)

        state = self._read_state(meta, states, x.dtype)
        history = x.new_empty(width, x.shape[0] + num_reqs * self.state_len)
        history.index_copy_(1, state_index, state.permute(1, 0, 2).reshape(width, -1))
        history.index_copy_(1, out_index + self.state_len, x.transpose(0, 1).contiguous())

        out = F.conv1d(
            history.unsqueeze(0), self.conv1d.weight, groups=width, dilation=self.dilation
        ).squeeze(0)
        new_state = history.index_select(1, next_state_index).view(width, num_reqs, self.state_len)
        states.index_copy_(
            0, meta.state_slots, new_state.permute(1, 0, 2).to(states.dtype).contiguous()
        )
        return F.silu(out.index_select(1, out_index).transpose(0, 1))

    def _prefill_indices(self, lens: List[int], device: torch.device):
        """Columns of the packed history: this forward's outputs, the state block, the next state block."""
        state_len = self.state_len
        counts = torch.tensor(lens, dtype=torch.int64)
        cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        pad = torch.arange(len(lens), dtype=torch.int64) * state_len
        base = cu[:-1] + pad
        out_index = torch.arange(int(cu[-1])) + torch.repeat_interleave(pad, counts)
        span = torch.arange(state_len, dtype=torch.int64)
        packed = torch.cat(
            [
                out_index,
                (base.unsqueeze(1) + span).reshape(-1),
                ((base + counts).unsqueeze(1) + span).reshape(-1),
            ]
        )
        if torch.cuda.is_available():
            packed = packed.pin_memory()
        packed = packed.to(device, non_blocking=True)
        n_out, n_state = out_index.numel(), len(lens) * state_len
        return packed[:n_out], packed[n_out : n_out + n_state], packed[n_out + n_state :]


__all__ = [
    "GpuResidentTable",
    "NGramEmbedding",
    "ZeroTable",
    "PLELayer",
    "PLEMetadata",
    "PLETableBackend",
    "PinnedUVATable",
    "build_ple_metadata",
    "derive_ngram_hash_constants",
    "short_conv_reference",
]
