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
  truncates the decoder's window at the replay start). The metadata carries the encoder pass
  (every new token) and the decoder pass (the replay tokens, or the same tokens in exact mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from freetoken.core import Batch, Req, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


def prompt_len(req: Req) -> int:
    """The request's prompt length (its tokens before generation)."""
    return req.max_device_len - req.output_len


def replay_start(req: Req, window: int, bounded: bool) -> int:
    """Where this request's encoder pass starts under Decoder SWA Bounded Replay.

    The decoder replays the prompt's last ``window`` tokens; when a prefix hit or a chunk boundary
    leaves fewer than ``window`` new tokens, the pass re-runs the encoder over the cached tokens
    from the previous window-page boundary to produce their hidden states. That recompute is
    READ-ONLY on the cache (``PrefillSegment.write_from``) and reads SWA keys back to ``start -
    window + 1`` -- live by the cache contract (``KVCacheGroupSpec.swa_resume_history`` is two
    windows in bounded mode: matched, locked and retained by the cache manager; admission turns a
    hit that would read further back into a miss, see ``replay_history``). Exact mode (or a chunk
    that already carries a window) starts at ``cached_len``.
    """
    if not bounded or req.extend_len >= window or req.cached_len == 0:
        return req.cached_len
    return max(0, (req.device_len - window) // window * window)


def replay_history(cached_len: int, window: int) -> int:
    """The most window history behind ``cached_len`` a bounded-replay resume there reads: with one
    new token the recompute starts furthest back, and its first query reads one window before that.
    ``2 * window - 1`` for a hit on a window page."""
    if cached_len == 0:
        return 0
    start = max(0, (cached_len + 1 - window) // window * window)
    return cached_len - max(0, start - window + 1)


@dataclass(frozen=True)
class PrefillSegment:
    """One request's slice of a flat prefill token stream.

    ``offset``/``n`` tile the stream; ``start_pos`` is the absolute position of the first token;
    ``window_floor`` bounds every query's sliding window from below (0 = the exact window; a bounded
    replay sets it to the replay start); ``write_from`` is the first position whose KV / compressor
    carry this pass WRITES -- positions before it are cached history recomputed for their hidden
    states only (a bounded-replay extension), so their pool rows are never touched. None = ``start_pos``.
    """

    offset: int
    n: int
    table_idx: int
    start_pos: int
    window_floor: int = 0
    write_from: int | None = None

    def __post_init__(self) -> None:
        if self.write_from is None:
            object.__setattr__(self, "write_from", self.start_pos)
        assert self.start_pos <= self.write_from <= self.end, (self.start_pos, self.write_from, self.end)

    @property
    def end(self) -> int:
        return self.start_pos + self.n

    @property
    def write_offset(self) -> int:
        """Stream offset of the first written token (``offset`` when nothing is recomputed)."""
        return self.offset + (self.write_from - self.start_pos)


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
    # Prefill: the encoder pass tiles the encoder stream -- the new tokens, extended backwards
    # over re-prefilled cached tokens under bounded replay (``extended``; the model builds that
    # stream with ``encoder_stream``); the decoder pass tiles the tokens the decoder layers
    # process (``decoder_rows`` gathers them out of the encoder stream; None = all of it).
    segments: List[PrefillSegment] | None = None
    decoder_segments: List[PrefillSegment] | None = None
    decoder_rows: torch.Tensor | None = None
    extended: bool = False
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
        from freetoken.kvcache.dsv41_geometry import DSV41Geometry

        self.config = config
        self.device = get_global_ctx().kv_cache.device
        geom = next(g.geometry for g in config.attention_groups if isinstance(getattr(g, "geometry", None), DSV41Geometry))
        self.geom: DSV41Geometry = geom
        self.window_size = geom.window
        args = config.dsv41_args
        self.swa_decoder_replay: str = getattr(args, "swa_decoder_replay", "exact")
        self.decoder_start: int = getattr(args, "decoder_start_layer", geom.n_layers)
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
        """Exact mode: one stream of every new token, the decoder pass is that same stream.
        Bounded replay: each request's encoder segment starts at ``replay_start`` (extended
        backwards over cached tokens when the chunk is shorter than a window) and its decoder
        segment is its last ``min(n_win, n)`` tokens with the window floored at the replay start."""
        bounded = self.swa_decoder_replay != "exact"
        win = self.window_size
        segments: List[PrefillSegment] = []
        off = 0
        extended = False
        for r in batch.reqs:
            start = replay_start(r, win, bounded)
            extended |= start != r.cached_len
            segments.append(PrefillSegment(off, r.device_len - start, r.table_idx, start, write_from=r.cached_len))
            off += r.device_len - start
        if not bounded:
            last = torch.tensor([s.n for s in segments], dtype=torch.int32, device=self.device).cumsum_(0) - 1
            return DSV41AttnMetadata(last_indices=last, segments=segments, decoder_segments=segments, decoder_rows=None, extended=extended)
        dec: List[PrefillSegment] = []
        rows: list[torch.Tensor] = []
        doff = 0
        for s in segments:
            # the decoder replays the prompt's last window (the reference deployment): its KV goes to
            # the request's private rings, so nothing here touches radix-shared pages
            n_replay = min(win, s.n)
            floor = s.end - n_replay
            dec.append(PrefillSegment(doff, n_replay, s.table_idx, floor, window_floor=floor))
            rows.append(torch.arange(s.offset + s.n - n_replay, s.offset + s.n))
            doff += n_replay
        last = torch.tensor([d.n for d in dec], dtype=torch.int32, device=self.device).cumsum_(0) - 1
        decoder_rows = torch.cat(rows).to(self.device, non_blocking=True) if rows else torch.empty(0, dtype=torch.int64, device=self.device)
        return DSV41AttnMetadata(last_indices=last, segments=segments, decoder_segments=dec, decoder_rows=decoder_rows, extended=extended)

    def encoder_stream(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """The prefill encoder stream ``(input_ids [T], positions [T] int64)``: the batch's own tensors,
        or -- when a bounded replay extended a segment backwards -- rebuilt from the requests' host
        token ids over ``[start_pos, device_len)`` (the batch's tensors only carry the new tokens)."""
        md = self.metadata
        if not md.extended:
            return batch.input_ids, batch.positions.long()
        ids = torch.cat([r.input_ids[s.start_pos : r.device_len] for r, s in zip(batch.reqs, md.segments)])
        pos = torch.cat([torch.arange(s.start_pos, s.end) for s in md.segments])
        return ids.to(self.device, dtype=batch.input_ids.dtype, non_blocking=True), pos.to(self.device, non_blocking=True)

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
        if self.geom.is_private_window(layer_id):
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
        private = layer_id is not None and self.geom.is_private_window(layer_id)
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

    def write_boundary_carries(self, source: int, *, lo: int, hi: int, window_slots: torch.Tensor, write_from: int | None = None) -> None:
        """Persist the compressor carry at every window-page boundary ``B`` in ``(max(lo, write_from), hi]``
        so a page-aligned radix match can resume by value. A page holds whole groups (``P % ratio == 0``),
        so the carry at a boundary is the empty group -- the reset block is written so a resume never
        reads a stale one. ``window_slots`` is indexed by ``pos - lo``; boundaries at or before
        ``write_from`` belong to cached history and are left alone. One batched ring write, no host syncs."""
        ring = self.pool.state_ring.get(source)
        if ring is None:
            return
        P = self.window_size
        first = (max(lo, lo if write_from is None else write_from) // P + 1) * P
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

        return indexer_logits_packed(q, weights, self.pool.idx_pool[source], self.geom.idx_fmt, locs, ratio, live, T=T, candidates=candidates)

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

        pool, geom = self.pool, self.geom
        src = geom.kv_source_of(layer_id)
        cmp = pool.main_pool[src] if src is not None else pool.main_pool[geom.kv_source_layer_ids[0]]
        return sparse_attn_packed(
            q, pool.window_pool[layer_id], geom.win_fmt, cmp, geom.main_fmt, attn_sink,
            topk_idxs, n_window, softmax_scale, cmp_counts=cmp_counts,
        )


__all__ = ["DSV41SparseAttnBackend", "DSV41AttnMetadata", "PrefillSegment", "SharedSelection", "prompt_len", "replay_start"]
