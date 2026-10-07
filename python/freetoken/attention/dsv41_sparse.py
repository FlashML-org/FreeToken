"""DeepSeek-V4.1 sparse-attention backend.

Same contract and division of labour as ``dsv4_sparse``: the backend owns the per-forward KV
ADDRESSING over the shared page table (window ring slots, compressed rows, compress-state carry,
the decode snapshot and the CUDA-graph staging buffers) plus the selection helpers that are pure
index arithmetic (causal top-k, the hierarchical candidate pool); the model computes projections
and hands the picks back to be resolved into pool rows.

What DSV41 adds on top of DSV4's vocabulary:

* **cross-layer sharing** -- the compressed tiers belong to the kv-source layer; a consumer layer
  addresses ``source_of(layer)``'s pools. The Top-K rows and the candidate pool the Full / Reindex
  layers produce ride on the per-forward ``SharedSelection`` for the Reuse layers to read.
* **packed rows** -- writes quantize into the pool (``pool.store_*``), reads dequantize in-kernel.
* **prefill passes with a window floor** -- a ``PrefillSegment`` carries ``window_floor``: the
  absolute position a query's sliding window may not reach below (Decoder SWA Bounded Replay
  truncates the decoder's window at ``L - W``, the start of the prompt's last window). The metadata
  carries the encoder pass (every new token) and the decoder pass (the new tokens inside the
  prompt's last window, or every new token in exact mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class PrefillSegment:
    """One request's slice of a flat prefill token stream.

    ``offset``/``n`` tile the stream; ``start_pos`` is the absolute position of the first token;
    ``window_floor`` bounds every query's sliding window from below (0 = the exact window; bounded
    replay sets it to the start of the prompt's last window).
    """

    offset: int
    n: int
    table_idx: int
    start_pos: int
    window_floor: int = 0

    @property
    def end(self) -> int:
        return self.start_pos + self.n


@dataclass
class SharedSelection:
    """What the index-producing layers hand down the stack within ONE forward.

    ``topk_rows`` are GLOBAL main-pool rows (``-1`` masked) for the current kv source, laid out
    exactly as the attention kernel consumes them; ``cmp_counts`` is the decode-only per-row live
    count that bounds the kernel's loop; ``candidates`` is the Hierarchical Sparse Indexer's
    compressed-position pool (``-1`` = empty slot). Layers run in order and every producer writes
    before its consumers read, so one slot each suffices.
    """

    source: int | None = None
    topk_rows: torch.Tensor | None = None
    cmp_counts: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


@dataclass
class DSV41AttnMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor
    # Prefill: the encoder pass tiles the batch's new tokens; the decoder pass tiles the tokens the
    # decoder layers process (``decoder_rows`` gathers them out of the encoder stream; None = all of
    # it). ``decoder_pad``: some request has no decoder row this chunk, and its ``last_indices`` entry
    # points at one placeholder row the model appends after the decoder stream (its logits are unused).
    segments: List[PrefillSegment] | None = None
    decoder_segments: List[PrefillSegment] | None = None
    decoder_rows: torch.Tensor | None = None
    decoder_pad: bool = False
    # Decode: the whole-history full-loc snapshot (captured buffer under a replay, lazy eager copy).
    full_snap: torch.Tensor | None = None
    table_rows: torch.Tensor | None = None
    window_ar: torch.Tensor | None = None
    selection: SharedSelection = field(default_factory=SharedSelection)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]

    @property
    def stage_width(self) -> int:
        assert self.full_snap is not None, "stage_width is capture/replay-only"
        return self.full_snap.shape[1]

    def full_snapshot(self) -> torch.Tensor:
        if self.full_snap is None:
            assert self.table_rows is not None, "snapshot is decode-only"
            pool = get_global_ctx().kv_cache
            self.full_snap = pool.full_loc_map.index_select(0, self.table_rows).to(torch.int64)
        return self.full_snap

    def window_ctx(self, pos: torch.Tensor, rows: torch.Tensor):
        """Layer-invariant decode ring context ``(window_slots, prev_window_slots,
        window_slots_topk [B, 1, win])``; computed fresh on every call (never cached -- a capture
        must record these gathers), off the snapshot so a concurrent allocation cannot redirect
        an in-flight replay."""
        snap = self.full_snapshot()
        translate = get_global_ctx().kv_cache.translate_full_to_window
        bs = pos.shape[0]
        j = self.window_ar
        assert j is not None, "window_ctx is decode-only"
        win = j.shape[0]
        window_slots = translate(snap[rows, pos])
        prev_window_slots = translate(snap[rows, (pos - 1).clamp_min(0)])
        p = pos[:, None] - ((pos[:, None] - j[None, :]) % win)
        ws = translate(snap[rows[:, None], p.clamp(min=0)])
        window_slots_topk = torch.where((p >= 0) & (j[None, :] <= pos[:, None]), ws, -1).view(bs, 1, win)
        return window_slots, prev_window_slots, window_slots_topk

    def private_window_ctx(self, pos: torch.Tensor, rows: torch.Tensor):
        """The private-ring counterpart of ``window_ctx`` for request-private window layers:
        ``(window_slots [B], window_slots_topk [B, 1, win])`` in the layer's ring, where a request's
        row holds positions by ``pos % win`` (candidate ``j`` is the position congruent to ``j``)."""
        assert self.table_rows is not None and self.window_ar is not None, "private_window_ctx is decode-only"
        pool = get_global_ctx().kv_cache
        bs = pos.shape[0]
        j = self.window_ar
        win = j.shape[0]
        table = self.table_rows[rows]
        window_slots = pool.ring_slots(table, pos)
        p = pos[:, None] - ((pos[:, None] - j[None, :]) % win)  # the position at ring index j (< 0: none yet)
        window_slots_topk = pool.ring_slots(table[:, None], p).view(bs, 1, win)
        return window_slots, window_slots_topk


@dataclass
class DSV41CaptureData:
    full_snap: torch.Tensor
    last_indices: torch.Tensor
    table_rows: torch.Tensor  # each batch row's page-table row (the private window rings key on it)

    @classmethod
    def create(cls, max_bs: int, width: int, device: torch.device) -> "DSV41CaptureData":
        return cls(
            full_snap=torch.full((max_bs, width), -1, dtype=torch.int64, device=device),
            last_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            table_rows=torch.zeros(max_bs, dtype=torch.int64, device=device),
        )


class DSV41SparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = get_global_ctx().kv_cache.device
        args = config.dsv41_args
        self.window_size = args.window_size
        self.swa_decoder_replay: str = args.swa_decoder_replay
        self.capture: DSV41CaptureData | None = None
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self._window_ar = torch.arange(self.window_size, device=self.device)

    @property
    def pool(self):
        return get_global_ctx().kv_cache

    # ----- generic contract -------------------------------------------------------------
    def forward(self, q, k, v, layer_id, batch, attn_spec: AttentionSpec | None = None):
        raise NotImplementedError("DSV41 attention is driven per-tier from the model module; use DSV41SparseAttnBackend.attend().")

    def prepare_metadata(self, batch: Batch) -> None:
        if not batch.is_decode:
            batch.attn_metadata = self._prefill_metadata(batch)
            return
        last = torch.tensor([r.extend_len for r in batch.padded_reqs], dtype=torch.int32, device=self.device).cumsum_(0) - 1
        batch.attn_metadata = DSV41AttnMetadata(last_indices=last, table_rows=self._table_rows(batch), window_ar=self._window_ar)

    def _prefill_metadata(self, batch: Batch) -> DSV41AttnMetadata:
        """The encoder pass runs every request's new tokens ``[cached_len, device_len)``. Exact mode runs
        the decoder on the same stream. Decoder SWA Bounded Replay runs it on the new tokens inside the
        prompt's last window ``[L - W, L)`` (``L = prompt_len``) with every query's window floored at
        ``L - W``: the chunks of a chunked prefill each take their part of that window (earlier rows
        come from the request's private ring), and a chunk that ends before it runs no decoder row."""
        segments: List[PrefillSegment] = []
        off = 0
        for r in batch.reqs:
            segments.append(PrefillSegment(off, r.extend_len, r.table_idx, r.cached_len))
            off += r.extend_len
        if self.swa_decoder_replay == "exact":
            last = torch.tensor([s.n for s in segments], dtype=torch.int32, device=self.device).cumsum_(0) - 1
            return DSV41AttnMetadata(last_indices=last, segments=segments, decoder_segments=segments)
        win = self.window_size
        dec: List[PrefillSegment] = []
        rows: list[torch.Tensor] = []
        last: list[int] = []
        doff = 0
        for r, s in zip(batch.reqs, segments):
            floor = max(0, r.prompt_len - win)
            lo = max(s.start_pos, floor)
            n = max(0, s.end - lo)
            if n:
                # the decoder's window KV goes to the request's private rings, never to radix-shared pages
                dec.append(PrefillSegment(doff, n, s.table_idx, lo, window_floor=floor))
                rows.append(torch.arange(s.offset + lo - s.start_pos, s.offset + s.n))
                doff += n
            last.append(doff - 1 if n else -1)
        pad = -1 in last
        last_indices = torch.tensor([doff if i < 0 else i for i in last], dtype=torch.int32, device=self.device)
        decoder_rows = torch.cat(rows).to(self.device, non_blocking=True) if rows else torch.empty(0, dtype=torch.int64, device=self.device)
        return DSV41AttnMetadata(last_indices=last_indices, segments=segments, decoder_segments=dec, decoder_rows=decoder_rows, decoder_pad=pad)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        self.max_graph_bs = max(bs_list)
        self.capture = DSV41CaptureData.create(self.max_graph_bs, max_seq_len, self.device)
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        rows = torch.full((bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device)
        self._point_to_capture(batch, bs, rows)

    def prepare_for_replay(self, batch: Batch) -> None:
        self._point_to_capture(batch, batch.padded_size, self._table_rows(batch))

    def _table_rows(self, batch: Batch) -> torch.Tensor:
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        return batch.active_table_idx.to(torch.int64)

    def _point_to_capture(self, batch: Batch, bs: int, rows_ti: torch.Tensor) -> None:
        assert self.capture is not None and bs <= self.max_graph_bs
        cap = self.capture
        src = self.pool.full_loc_map.index_select(0, rows_ti)
        w = min(src.shape[1], cap.full_snap.shape[1])
        cap.full_snap[:bs, :w].copy_(src[:bs, :w])
        cap.table_rows[:bs].copy_(rows_ti[:bs])
        batch.attn_metadata = DSV41AttnMetadata(
            last_indices=cap.last_indices[:bs], full_snap=cap.full_snap[:bs], table_rows=cap.table_rows[:bs], window_ar=self._window_ar,
        )

    # ----- metadata / selection access -------------------------------------------------
    @property
    def metadata(self) -> DSV41AttnMetadata:
        md = get_global_ctx().batch.attn_metadata
        assert isinstance(md, DSV41AttnMetadata)
        return md

    @property
    def selection(self) -> SharedSelection:
        return self.metadata.selection

    def snapshot(self) -> torch.Tensor:
        return self.metadata.full_snapshot()

    # ----- window tier -------------------------------------------------------------------
    def window_slots_of(self, ti: int, lo: int, hi: int) -> torch.Tensor:
        """SHARED window slots of positions ``[lo, hi)`` off the request's LIVE full locs (slid-out -> -1)."""
        return self.pool.translate_full_to_window(self.pool.full_loc_map[ti, lo:hi])

    def ring_slots_of(self, ti: int, lo: int, hi: int) -> torch.Tensor:
        """PRIVATE-ring slots of positions ``[lo, hi)`` of page-table row ``ti`` (``hi - lo <= win``)."""
        assert hi - lo <= self.window_size, f"a private ring holds one window, not {hi - lo} positions"
        return self.pool.ring_slots(torch.tensor(ti, device=self.device), torch.arange(lo, hi, device=self.device))

    def layer_window_slots_of(self, layer_id: int, ti: int, lo: int, hi: int) -> torch.Tensor:
        """Where ``layer_id`` keeps the window KV of positions ``[lo, hi)``: its private ring or the shared pool."""
        if self.pool.is_private_window(layer_id):
            return self.ring_slots_of(ti, lo, hi)
        return self.window_slots_of(ti, lo, hi)

    def store_window(self, kv: torch.Tensor, layer_id: int, window_slots: torch.Tensor) -> None:
        self.pool.store_window(kv, layer_id, window_slots)

    def window_topk_prefill(self, seg: PrefillSegment, layer_id: int | None = None) -> torch.Tensor:
        """Per-query window candidates for a prefill segment as GLOBAL window slots ``[1, n, win]``
        (``-1`` where empty) in the tier ``layer_id`` reads (shared pool by default). Query at
        absolute ``p`` sees ``[max(floor, first_retained, p - win + 1), p]``."""
        win, device = self.window_size, self.device
        lo = max(0, seg.start_pos - win + 1, seg.window_floor)
        private = layer_id is not None and self.pool.is_private_window(layer_id)
        ws_pool = (self.ring_slots_of if private else self.window_slots_of)(seg.table_idx, lo, seg.end)  # [end - lo]
        abs_p = seg.start_pos + torch.arange(seg.n, device=device).unsqueeze(1)
        cand = (abs_p - win + 1).clamp(min=lo) + torch.arange(win, device=device)
        cols = torch.where(cand > abs_p, -1, cand - lo)
        g = ws_pool[cols.clamp_min(0)]
        return torch.where(cols < 0, -1, g).unsqueeze(0)

    # ----- compressed tiers (per kv source) ----------------------------------------------
    def source_of(self, layer_id: int) -> int:
        return self.pool.source_of(layer_id)

    def compressed_rows_of(self, ti: int, group_starts: torch.Tensor, ratio: int) -> torch.Tensor:
        """Main / index rows of the compressed groups whose ABSOLUTE first positions are
        ``group_starts``, off the request's LIVE full locs (a page is ratio-divisible, so every
        position of a group shares one row)."""
        return self.pool.cmp_rows(self.pool.full_loc_map[ti, group_starts], ratio)

    def locs_prefill(self, ti: int, end: int) -> torch.Tensor:
        """``[1, end]`` int32: the request's live full locs for positions ``[0, end)`` -- what the indexer
        derives compressed rows from (``loc // ratio``) for the positions its queries may see."""
        return self.pool.full_loc_map[ti : ti + 1, :end]

    def store_main(self, latent: torch.Tensor, source: int, rows: torch.Tensor) -> None:
        self.pool.store_main(latent, source, rows)

    def store_index(self, k: torch.Tensor, source: int, rows: torch.Tensor) -> None:
        self.pool.store_index(k, source, rows)

    def decode_store_rows(self, rows: torch.Tensor, pos: torch.Tensor, ratio: int, source: int, completed: torch.Tensor) -> torch.Tensor:
        """Per-row decode destination: the completed group's arithmetic row, or the row's OWN scratch
        row when this step did not complete a group (graph-safe, collision-free masked store)."""
        row_of_group = self.pool.cmp_rows(self.snapshot()[rows, pos], ratio)
        scratch = rows + self.pool.scratch_base[source]
        return torch.where(completed, row_of_group, scratch)

    # ----- compress-state ring (ratio > 1 sources) ---------------------------------------
    def ring_page_base(self, window_slots: torch.Tensor, ring_size: int) -> torch.Tensor:
        return torch.div(window_slots, self.window_size, rounding_mode="floor") * ring_size

    def carry_state_loc(self, window_slot: int, ring_size: int) -> torch.Tensor:
        base = (window_slot // self.window_size) * ring_size
        return torch.arange(base, base + ring_size, device=self.device, dtype=torch.int64)

    def read_carry(self, source: int, window_slot: int) -> torch.Tensor:
        """The ``[ring_size, 2 * head_dim]`` carry block at ``window_slot``'s page (kv | score)."""
        ring = self.pool.state_ring[source]
        return ring.get(self.carry_state_loc(window_slot, ring.ring_size))

    def write_carry(self, source: int, window_slot: int, kv_score: torch.Tensor) -> None:
        ring = self.pool.state_ring[source]
        ring.set(self.carry_state_loc(window_slot, ring.ring_size), kv_score)

    def read_carry_blocks(self, source: int, window_slots: torch.Tensor) -> torch.Tensor:
        ring = self.pool.state_ring[source]
        return ring.get_blocks(self.ring_page_base(window_slots, ring.ring_size))

    def write_carry_blocks(self, source: int, window_slots: torch.Tensor, blocks: torch.Tensor) -> None:
        ring = self.pool.state_ring[source]
        ring.set_blocks(self.ring_page_base(window_slots, ring.ring_size), blocks)

    def write_boundary_carries(self, source: int, *, lo: int, hi: int, window_slots: torch.Tensor) -> None:
        """Persist the compressor carry at every window-page boundary ``B`` in ``(lo, hi]`` so a
        page-aligned radix match can resume by value. A page holds whole groups (``P % ratio == 0``),
        so the carry at a boundary is the empty group -- the reset block is written so a resume never
        reads a stale one. ``window_slots`` is indexed by ``pos - lo``. One batched ring write, no host syncs."""
        ring = self.pool.state_ring.get(source)
        if ring is None:
            return
        P = self.window_size
        first = (lo // P + 1) * P
        if first > hi:
            return
        bounds = torch.arange(first, hi + 1, P, device=self.device)
        page_slots = window_slots[bounds - 1 - lo].to(torch.int64)  # the last slot of each page
        locs = (torch.div(page_slots, P, rounding_mode="floor") * ring.ring_size)[:, None] + torch.arange(ring.ring_size, device=self.device)
        empty = torch.cat([
            torch.zeros(ring.ring_size, ring.item_size, dtype=torch.float32, device=self.device),
            torch.full((ring.ring_size, ring.item_size), float("-inf"), dtype=torch.float32, device=self.device),
        ], dim=-1)
        ring.set(locs.flatten(), empty.repeat(bounds.numel(), 1))

    # ----- indexer ----------------------------------------------------------------------
    def indexer_logits(
        self, q: torch.Tensor, weights: torch.Tensor, source: int, locs: torch.Tensor, ratio: int, live: torch.Tensor,
        T: int | None = None, candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, S, T]`` full-range scores (live columns written) or ``[B, S, NC]`` candidate scores; the
        row of compressed position ``t`` is ``locs[b, t * ratio] // ratio``."""
        from freetoken.kernel.triton.dsv41.indexer import indexer_logits_packed

        return indexer_logits_packed(q, weights, self.pool.idx_pool[source], self.pool.idx_fmt, locs, ratio, live, T=T, candidates=candidates)

    @staticmethod
    def select_topk(scores: torch.Tensor, live: torch.Tensor, topk: int) -> torch.Tensor:
        """Exact causal top-k over ``[B, S, T]`` full-range scores (columns at or past ``live`` unread):
        compressed positions ``[B, S, topk]`` int32 in ascending order, ``-1`` in the tail for picks
        that do not exist (fewer live or finite positions than ``topk``). Ties: lowest position wins."""
        from freetoken.kernel.triton.dsv41.topk import dsv41_topk

        b, s, t = scores.shape
        return dsv41_topk(scores.reshape(b * s, t), live.reshape(b * s), topk).view(b, s, topk)

    @staticmethod
    def select_topk_in_candidates(scores: torch.Tensor, candidates: torch.Tensor, topk: int) -> torch.Tensor:
        """Top-k over ``[B, S, NC]`` candidate-aligned scores -> the picked compressed positions
        ``[B, S, topk]`` int32. The candidate list is a sorted valid prefix (``-1`` tail) and empty /
        unreachable slots score ``-inf``, so the ascending winning slots map to ascending positions."""
        from freetoken.kernel.triton.dsv41.topk import dsv41_topk

        b, s, nc = scores.shape
        live = torch.full((b * s,), nc, dtype=torch.int32, device=scores.device)
        slots = dsv41_topk(scores.reshape(b * s, nc), live, topk).view(b, s, topk)
        pos = candidates.gather(-1, slots.clamp_min(0).to(torch.int64)).to(torch.int32)
        return torch.where(slots < 0, -1, pos)

    @staticmethod
    def select_candidate_blocks(scores: torch.Tensor, live: torch.Tensor, topk_blocks: int, block_size: int) -> torch.Tensor:
        """Level one of the Hierarchical Sparse Indexer: the ``topk_blocks`` best live ``block_size``
        blocks by their best position, the block holding the newest live position always kept,
        expanded to the compact candidate list ``[B, S, topk_blocks * block_size]`` int32 of
        compressed positions (ascending, ``-1`` tail). Mirrors the reference ``select_candidate_blocks``
        (which returns the equivalent boolean mask)."""
        from freetoken.kernel.triton.dsv41.topk import dsv41_candidate_blocks

        b, s, t = scores.shape
        return dsv41_candidate_blocks(scores.reshape(b * s, t), live.reshape(b * s), topk_blocks, block_size).view(b, s, -1)

    @staticmethod
    def positions_to_rows(positions: torch.Tensor, locs: torch.Tensor, ratio: int) -> torch.Tensor:
        """Compressed positions ``[B, S, K]`` (``-1`` masked) -> GLOBAL main rows ``locs[b, p * ratio] // ratio``."""
        b = positions.shape[0]
        idx = (positions.to(torch.int64) * ratio).clamp_min(0).flatten(1)
        full = locs.to(torch.int64).gather(1, idx.clamp(max=locs.shape[1] - 1)).view_as(positions)
        rows = torch.div(full, ratio, rounding_mode="floor")
        return torch.where((positions < 0) | (full < 0), -1, rows)

    # ----- attention -----------------------------------------------------------------------
    def attend(
        self, q: torch.Tensor, layer_id: int, topk_idxs: torch.Tensor, n_window: int, attn_sink: torch.Tensor,
        softmax_scale: float, cmp_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Paged sparse attention over ``[window | compressed]`` global rows; the compressed half reads
        the layer's kv source (window-only layers pass ``n_window == topk``)."""
        from freetoken.kernel.triton.dsv41.sparse_attn import sparse_attn_packed

        pool, args = self.pool, self.config.dsv41_args
        src = args.roles[layer_id].kv_source
        cmp = pool.main_pool[src] if src is not None else pool.main_pool[args.backbone_kv_sources[0]]
        return sparse_attn_packed(
            q, pool.window_pool[layer_id], pool.win_fmt, cmp, pool.main_fmt, attn_sink,
            topk_idxs, n_window, softmax_scale, cmp_counts=cmp_counts,
        )


__all__ = ["DSV41SparseAttnBackend", "DSV41AttnMetadata", "PrefillSegment", "SharedSelection"]
