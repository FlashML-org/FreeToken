"""Paged addressing for the DeepSeek-V4 compressors, mixed into the sparse-attention backend.

DSV4 runs two compressors per compressed layer -- the attention one (writes ``cmp_pool``) and
the Lightning Indexer's own (writes ``idx_pool``) -- over three tiers of paged state:

* the compressed-KV pool row for a block, which is arithmetic off the block's full loc
  (``full_loc(b * ratio) // ratio``), so it needs no slot map;
* the per-window-page compress-state RING, which carries the rolling reduction across page and
  request boundaries so a radix hit can resume a prefix by value;
* a per-row scratch row, the destination for a decode step whose block did not complete (a
  discarded write that keeps the masked store graph-safe and collision-free).

All of that is addressing, so it lives here rather than in the model: the module keeps its
weights and the reduction math and passes them in, mirroring sglang's ``CompressorBackendMixin``
(``forward_compress(*, ape, norm, freqs_cis_cache, ...)``).
"""

from __future__ import annotations

import os

import torch


class CompressorBackendMixin:
    # DSV4_RING_CHECK=1 -> assert the ring read-back equals what was written. Off by default.
    _RING_CHECK = os.environ.get("DSV4_RING_CHECK", "0") == "1"

    # tier -> (KV-pool, state-ring, scratch-base) attribute names on the pool.
    _TIER_ATTRS = {
        "attn": ("cmp_pool", "state_ring", "cmp_scratch_base"),
        "idx": ("idx_pool", "indexer_state_ring", "idx_scratch_base"),
    }

    # ----- pool views ------------------------------------------------------------------
    def compress_pool(self, layer_id: int, tier: str) -> torch.Tensor:
        return getattr(self.pool, self._TIER_ATTRS[tier][0])[layer_id]

    def compress_state_ring(self, layer_id: int, tier: str):
        return getattr(self.pool, self._TIER_ATTRS[tier][1])[layer_id]

    def compress_scratch_base(self, layer_id: int, tier: str) -> int:
        return getattr(self.pool, self._TIER_ATTRS[tier][2])[layer_id]

    # ----- compressed-row addressing ---------------------------------------------------
    def compress_rows_of(self, ti: int, block_starts: torch.Tensor, ratio: int) -> torch.Tensor:
        """Rows for blocks whose ABSOLUTE start positions are ``block_starts``, off the request's
        LIVE full locs (prefill/extend). A full page is ratio-divisible, so every position in a
        block shares one row."""
        return self.pool.cmp_rows(self.pool.full_loc_map[ti, block_starts], ratio)

    def decode_compress_rows(
        self, rows: torch.Tensor, pos: torch.Tensor, ratio: int, layer_id: int, tier: str,
        completed: torch.Tensor, ti: int | None = None,
    ) -> torch.Tensor:
        """Per-row decode store destination: the completed block's arithmetic row, or the row's
        OWN scratch row when this step did not finish a block.

        Reads the decode SNAPSHOT (not the live map) so a concurrent allocate_paged cannot
        redirect the write; scratch keeps the masked store free of negative indices (which
        ``index_copy_`` would treat as out of bounds) and collision-free across rows.

        ``ti`` addresses the LIVE map by table index instead, for a caller that is not a
        decode batch. A speculative block walks the compressor per token from inside a
        prefill, where no decode snapshot exists -- and it is safe there for the same
        reason the rest of prefill is: the batch owns its rows for the whole forward.
        """
        source = (
            self.pool.full_loc_map[ti, pos] if ti is not None
            else self.snapshot()[rows, pos]
        )
        row_of_block = self.pool.cmp_rows(source, ratio)
        scratch = rows + self.compress_scratch_base(layer_id, tier)
        return torch.where(completed, row_of_block, scratch)

    def scatter_compressed(
        self, layer_id: int, tier: str, rows: torch.Tensor, kv: torch.Tensor
    ) -> None:
        pool = self.compress_pool(layer_id, tier)
        pool.index_copy_(0, rows, kv.to(pool.dtype))

    # ----- compress-state ring ---------------------------------------------------------
    def carry_state_loc(self, window_slot: int, ring_size: int) -> torch.Tensor:
        """The window page's ring block: ``(window_slot // P) * ring_size + arange(ring_size)``."""
        base = (window_slot // self.window_size) * ring_size
        return torch.arange(base, base + ring_size, device=self.device, dtype=torch.int64)

    def ring_page_base(self, window_slots: torch.Tensor, ring_size: int) -> torch.Tensor:
        """Per-row ring block base. Distinct window pages map to disjoint blocks, which is what
        keeps co-tenant decode rows from contaminating each other's carry."""
        return torch.div(window_slots, self.window_size, rounding_mode="floor") * ring_size

    def read_carry(
        self, layer_id: int, tier: str, window_slot: int, ring_size: int
    ) -> torch.Tensor:
        """The ``[ring_size, 2 * item]`` carry block at ``window_slot``'s page -- kv half then
        score half. Used to resume a prefix by value on a radix hit."""
        ring = self.compress_state_ring(layer_id, tier)
        assert ring is not None
        return ring.get(self.carry_state_loc(window_slot, ring_size))

    def write_carry(
        self, layer_id: int, tier: str, window_slot: int, ring_size: int, kv_score: torch.Tensor
    ) -> None:
        ring = self.compress_state_ring(layer_id, tier)
        if ring is None:
            return
        state_loc = self.carry_state_loc(window_slot, ring_size)
        ring.set(state_loc, kv_score)
        if self._RING_CHECK:
            assert torch.equal(ring.get(state_loc), kv_score), (
                f"ring read-back != written block (layer={layer_id} tier={tier} "
                f"window_slot={window_slot})"
            )

    def read_carry_blocks(
        self, layer_id: int, tier: str, window_slots: torch.Tensor, ring_size: int
    ) -> torch.Tensor:
        """Per-row carry blocks ``[B, ring_size, 2 * item]`` for a decode step."""
        ring = self.compress_state_ring(layer_id, tier)
        return ring.get_blocks(self.ring_page_base(window_slots, ring_size))

    def write_carry_blocks(
        self, layer_id: int, tier: str, window_slots: torch.Tensor, ring_size: int,
        blocks: torch.Tensor,
    ) -> None:
        ring = self.compress_state_ring(layer_id, tier)
        ring.set_blocks(self.ring_page_base(window_slots, ring_size), blocks)

    def write_boundary_carries(
        self, *, layer_id: int, tier: str, ratio: int, overlap: bool, ring_size: int,
        ape: torch.Tensor, kv: torch.Tensor, score: torch.Tensor, lo: int, hi: int,
        window_slots: torch.Tensor,
    ) -> None:
        """Persist the carry at EVERY window-page boundary in ``(lo, hi]``, so any future radix
        match (always page-aligned) can read its carry by value.

        The carry at a page-aligned boundary B is position-local: with overlap (ratio 4) it is
        ``kv[B-ratio:B]`` in the overlap-seed slots, without it (ratio 128) it is the reset
        state. ``kv``/``score`` cover ``[lo, hi)``; ``window_slots`` is indexed by ``pos - lo``.
        The first boundary written is ``lo + P`` -- the one AT ``lo`` is the seed source and is
        already in the ring.
        """
        ring = self.compress_state_ring(layer_id, tier)
        if ring is None:
            return
        P, item = self.window_size, ring.item_size
        rows = ring_size
        empty_kv = kv.new_zeros(rows, item, dtype=torch.float32)
        empty_ss = kv.new_full((rows, item), float("-inf"), dtype=torch.float32)
        for B in range((lo // P + 1) * P, hi + 1, P):
            ws = int(window_slots[B - 1 - lo].item())
            if overlap:
                blk_kv, blk_ss = empty_kv.clone(), empty_ss.clone()
                blk_kv[:ratio] = kv[0, B - ratio - lo: B - lo]
                blk_ss[:ratio] = score[0, B - ratio - lo: B - lo] + ape
                kv_score = torch.cat([blk_kv, blk_ss], dim=-1)
            else:
                kv_score = torch.cat([empty_kv, empty_ss], dim=-1)
            self.write_carry(layer_id, tier, ws, ring_size, kv_score)


__all__ = ["CompressorBackendMixin"]
