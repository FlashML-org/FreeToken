"""DeepSeek-V4-Flash sparse-attention backend.

DSV4's attention is bespoke (MLA sliding window + learned KV compression + Lightning
Indexer), so it does not share the generic paged kernels. It does follow the same backend
contract as every other model: the backend owns the per-forward KV ADDRESSING, allocates the
CUDA-graph input buffers, and stages them before a replay -- the model computes projections
and the compressor/indexer picks, and asks the backend to resolve them into pool slots.

The addressing DSV4 needs is a whole-history full-loc row per request:

* the window tier is a 128-entry ring, so a decode query's window candidates are
  ``translate(full_loc[pos - (pos - j) % win])`` for j in [0, win);
* the compressed tiers are addressed by BLOCK index, so block ``b`` lives at compressed row
  ``full_loc(b * ratio) // ratio``.

Both read positions anywhere in the request's history, so the backend keeps a per-batch
SNAPSHOT of those rows (``DSV4AttnMetadata.full_snap``) instead of letting the captured kernels
walk the live page table: the next batch's ``allocate_paged`` mutates that table while the
current graph is still replaying. This is the same shape as the generic backend's
``_point_to_capture`` (which copies its ragged ``indices`` into the captured buffer) and as
sglang's ``DSV4AttnMetadata.copy_()``, which copies ``page_table`` into the captured metadata.

The snapshot's width comes from ``init_capture_graph``'s ``max_seq_len``: the engine's live
ceiling min(model max, KV token budget), rounded up to the page table's 32-column alignment.
The decode staging grids are sized off that same width, so no static grid can address a column
the snapshot lacks, and the only columns either has beyond the scheduler's admission bound are
the <= 31 alignment ones -- permanently -1, hence masked in the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .dsv4_compress import CompressorBackendMixin
from .dsv4_indexer import IndexerBackendMixin

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class DSV4AttnMetadata(BaseAttnMetadata):
    """Per-forward DSV4 addressing.

    ``last_indices`` is the generic contract (the LM head picks each request's final row).
    A prefill batch carries ``segments`` -- one ``(offset, extend_len, table_idx, start_pos)``
    per request, tiling the flat token stream -- so the model reads addressing off the
    metadata, never off ``Req``. ``full_snap`` is decode-only and None on a prefill batch,
    where the model addresses the live slot maps directly -- no captured graph is in flight,
    so there is nothing to race.
    """

    last_indices: torch.Tensor
    # Prefill only: per-request (offset, extend_len, table_idx, start_pos) tiling [0, T).
    segments: List[Tuple[int, int, int, int]] | None = None
    full_snap: torch.Tensor | None = None  # [B, stage_width] int64 whole-history full locs
    # Eager decode defers the snapshot: a graph-eligible batch has its metadata replaced by
    # prepare_for_replay before anything reads it, so taking it here would be pure waste on the
    # hot path. ``full_snapshot`` materializes it from these rows on first use.
    table_rows: torch.Tensor | None = None
    # The backend's cached arange(window_size) (address-stable across the capture's life).
    window_ar: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]

    @property
    def stage_width(self) -> int:
        """Static width of the compressed-candidate staging grids. A capture bakes it, so it
        must cover every position a replay can reach -- which is exactly the snapshot's width,
        since that is the buffer those grids gather from."""
        assert self.full_snap is not None, "stage_width is capture/replay-only"
        return self.full_snap.shape[1]

    def full_snapshot(self) -> torch.Tensor:
        """This decode batch's whole-history full locs (the buffer every decode gather reads).

        Under a replay this is the captured buffer, already staged. Eager, it is materialized
        here on first use -- a copy, so the next batch's allocate_paged cannot move the rows
        this forward reads.
        """
        if self.full_snap is None:
            assert self.table_rows is not None, "snapshot is decode-only"
            pool = get_global_ctx().kv_cache
            self.full_snap = pool.full_loc_map.index_select(0, self.table_rows).to(torch.int64)
        return self.full_snap

    def window_ctx(
        self, pos: torch.Tensor, rows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer-INVARIANT 128-ring context for a decode step: ``(window_slots,
        prev_window_slots, window_slots_topk)``. The model resolves it ONCE per forward and
        threads it into every layer.

        Computed FRESH on every call -- NEVER cached on the metadata. Under CUDA-graph capture
        the GraphRunner runs a warm forward and then the captured forward on the SAME batch and
        metadata; a cached tensor would leave these gathers out of the captured graph, freezing
        every replay at the capture-time ring slots (-1 fills included -> kernel/index_copy_
        OOB on the first real step). Reads the snapshot rather than the live map, so a
        concurrent allocate_paged cannot redirect an in-flight replay; runs inside the captured
        graph (its inputs -- the snapshot and ``positions`` -- are graph buffers), which is why
        it takes ``pos`` instead of reading it off the batch.
        """
        snap = self.full_snapshot()
        translate = get_global_ctx().kv_cache.translate_full_to_window
        bs = pos.shape[0]
        j = self.window_ar
        assert j is not None, "window_ctx is decode-only"
        win = j.shape[0]
        window_slots = translate(snap[rows, pos])
        prev_window_slots = translate(snap[rows, (pos - 1).clamp_min(0)])
        # Ring slot j holds the latest position p <= pos with p % win == j, i.e.
        # p = pos - ((pos - j) % win); p < 0 (early decode) is masked to -1.
        p = pos[:, None] - ((pos[:, None] - j[None, :]) % win)
        ws = translate(snap[rows[:, None], p.clamp(min=0)])
        ring = torch.where(p >= 0, ws, torch.full_like(ws, -1))
        window_slots_topk = torch.where(
            j[None, :] <= pos[:, None], ring, torch.full_like(ring, -1)
        ).view(bs, 1, win)
        return window_slots, prev_window_slots, window_slots_topk


@dataclass
class DSV4CaptureData:
    """CUDA-graph input buffers. Addresses must stay stable for the life of the capture, so
    these are allocated once per capture generation and only ever overwritten in place."""

    full_snap: torch.Tensor
    last_indices: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, width: int, device: torch.device) -> DSV4CaptureData:
        return cls(
            # -1 is the "no such position" sentinel: the window translate maps it to -1 (which
            # the sparse kernel masks) and the compressed row arithmetic floors below zero.
            full_snap=torch.full((max_bs, width), -1, dtype=torch.int64, device=device),
            last_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
        )


class DSV4SparseAttnBackend(BaseAttnBackend, CompressorBackendMixin, IndexerBackendMixin):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = get_global_ctx().kv_cache.device
        self.window_size = config.dsv4_args.window_size
        self.capture: DSV4CaptureData | None = None
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self._window_ar = torch.arange(self.window_size, device=self.device)

    # Read the pool live: a runtime rebuild swaps it with no backend rebind (same reason the
    # model reads its buffers off ctx.kv_cache per access).
    @property
    def pool(self):
        return get_global_ctx().kv_cache

    # ----- generic contract -------------------------------------------------------------
    def forward(self, q, k, v, layer_id, batch, attn_spec: AttentionSpec | None = None):
        raise NotImplementedError(
            "DeepSeek-V4 attention is driven per-tier from the model module; use "
            "DSV4SparseAttnBackend.attend()."
        )

    def prepare_metadata(self, batch: Batch) -> None:
        last = torch.tensor(
            [r.extend_len for r in batch.padded_reqs], dtype=torch.int32, device=self.device
        ).cumsum_(0) - 1
        if not batch.is_decode:
            # Segments tile the flat token stream per request: (offset, extend_len, table_idx,
            # start_pos). Built host-side here (the scheduler stream under overlap) from the
            # same Req counters every backend's prepare_metadata reads, so the model's forward
            # takes its addressing off the metadata, never off Req.
            segments = []
            off = 0
            for r in batch.reqs:
                segments.append((off, r.extend_len, r.table_idx, r.cached_len))
                off += r.extend_len
            batch.attn_metadata = DSV4AttnMetadata(last_indices=last, segments=segments)
            return
        # Decode: carry the rows, not the snapshot. A graph-eligible batch gets its metadata
        # replaced by prepare_for_replay (into the captured buffer) before any read, so taking
        # the snapshot here would be a full-width gather thrown away on every replay. The eager
        # path materializes it lazily in ``full_snapshot()``, still inside this batch's forward.
        batch.attn_metadata = DSV4AttnMetadata(
            last_indices=last, table_rows=self._table_rows(batch), window_ar=self._window_ar
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        self.max_graph_bs = max(bs_list)
        self.capture = DSV4CaptureData.create(self.max_graph_bs, max_seq_len, self.device)
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        # The capture batch is all dummy rows; stage them so the captured gather reads the
        # reserved page-0 slots instead of the -1 fill.
        bs = batch.size
        rows = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._point_to_capture(batch, bs, rows)

    def prepare_for_replay(self, batch: Batch) -> None:
        self._point_to_capture(batch, batch.padded_size, self._table_rows(batch))

    def prepare_for_spec_capture(self, batch: Batch) -> None:
        """Point a fixed-width DSpark target capture at the reserved dummy row."""
        bs = batch.padded_size
        rows = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._point_to_capture(batch, bs, rows)

    def prepare_for_spec_replay(self, batch: Batch) -> None:
        """Stage one snapshot row per request for a DSpark target replay."""
        self._point_to_capture(batch, batch.padded_size, self._table_rows(batch))

    # ----- DSV4 addressing vocabulary ---------------------------------------------------
    def snapshot(self) -> torch.Tensor:
        """The current decode batch's snapshot (see ``DSV4AttnMetadata.full_snapshot``);
        internal vocabulary for the compressor/indexer mixins."""
        md = get_global_ctx().batch.attn_metadata
        assert isinstance(md, DSV4AttnMetadata)
        return md.full_snapshot()

    def window_slots_of(self, ti: int, lo: int, hi: int) -> torch.Tensor:
        """Window slots for a prefill/extend range, off the request's LIVE full locs (positions
        that have slid out of the window translate to -1)."""
        return self.pool.translate_full_to_window(self.pool.full_loc_map[ti, lo:hi])

    def window_slots_at(self, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Window slots for SCATTERED ``(row, position)`` pairs, one per token.

        ``window_slots_of`` covers one request's contiguous range. A flat batch spans
        several requests, and its tokens arrive as parallel row/position vectors, so the
        same lookup is a gather rather than a slice. Positions that have slid out of the
        window translate to -1, exactly as they do there.
        """
        return self.pool.translate_full_to_window(
            self.pool.full_loc_map[rows.long(), positions.long()]
        )

    def win_cols_to_global(self, win_cols: torch.Tensor, slot_lut: torch.Tensor) -> torch.Tensor:
        """Per-query window columns (or -1) -> GLOBAL window-pool slots via ``slot_lut``."""
        g = slot_lut[win_cols.clamp_min(0)]
        return torch.where(win_cols < 0, torch.full_like(g, -1), g)

    def blocks_to_global(
        self, blocks: torch.Tensor, ratio: int, ti: int | None = None,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compressed BLOCK indices (or -1) -> GLOBAL compressed rows: block b lives at
        ``full_loc(b * ratio) // ratio``. Prefill passes ``ti`` (the live map); decode passes
        ``rows`` (the snapshot)."""
        safe = blocks.clamp_min(0)
        if rows is None:
            assert ti is not None
            full_at = self.pool.full_loc_map[ti, safe * ratio]
        else:
            full_at = self.snapshot()[rows[:, None, None], safe * ratio]
        g = self.pool.cmp_rows(full_at, ratio)
        return torch.where(blocks < 0, torch.full_like(g, -1), g)

    def store_window(self, kv: torch.Tensor, layer_id: int, window_slots: torch.Tensor) -> None:
        self.pool.store_window(kv, layer_id, window_slots)

    def attend(
        self, q: torch.Tensor, layer_id: int, topk_idxs: torch.Tensor, n_window: int,
        attn_sink: torch.Tensor, softmax_scale: float,
        cmp_counts: torch.Tensor | None = None, has_compression: bool = True,
    ) -> torch.Tensor:
        """Paged sparse MLA attention over ``[window | compressed]`` global slots.

        ``cmp_counts`` bounds the compressed half per query from DEVICE memory, so a captured
        graph's work tracks the live position instead of the staged width.
        """
        from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

        pool = self.pool
        # ratio-0 layers have no compressed pool; the kernel never reads it there (n_window ==
        # topk), so alias the window pool to keep the two-pool stride assert happy.
        cmp = pool.cmp_pool[layer_id] if has_compression else pool.window_pool[layer_id]
        return sparse_attn_paged(
            q, pool.window_pool[layer_id], cmp, attn_sink,
            topk_idxs.int(), n_window, softmax_scale, cmp_counts=cmp_counts,
        )

    # ----- internals --------------------------------------------------------------------
    def _table_rows(self, batch: Batch) -> torch.Tensor:
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        return batch.active_table_idx.to(torch.int64)

    def _point_to_capture(self, batch: Batch, bs: int, rows_ti: torch.Tensor) -> None:
        """Stage this batch's rows into the captured snapshot, then hand the model a view of it.

        On replay this runs on the engine stream immediately before ``graph.replay()``, so the
        copy is ordered against both the replay that reads it and the next batch's allocation.
        """
        assert self.capture is not None and bs <= self.max_graph_bs
        cap = self.capture
        prior = getattr(batch, "attn_metadata", None)
        segments = getattr(prior, "segments", None)
        src = self.pool.full_loc_map.index_select(0, rows_ti)
        w = min(src.shape[1], cap.full_snap.shape[1])
        cap.full_snap[:bs, :w].copy_(src[:bs, :w])
        batch.attn_metadata = DSV4AttnMetadata(
            last_indices=cap.last_indices[:bs], segments=segments,
            full_snap=cap.full_snap[:bs],
            window_ar=self._window_ar,
        )


__all__ = ["DSV4SparseAttnBackend", "DSV4AttnMetadata"]
