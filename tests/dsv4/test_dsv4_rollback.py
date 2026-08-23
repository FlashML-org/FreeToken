"""Undoing a rejected speculative block's effect on the DSV4 compressor carry.

The carry is a reduction over the tokens seen so far, held per window page. Nothing
else in the engine can rebuild it from the KV, so if a speculative step advances it in
place and the block is then rejected, the state the next real step must read is simply
gone -- and the request continues from a carry that saw tokens it never emitted. That
is silent corruption, not a crash, so the contract is pinned here.

CPU-only: a fake backend stands in for the paged ring.
"""

from __future__ import annotations

import torch

from freetoken.models.deepseek_v4.rollback import CarrySnapshot, needs_rollback

P = 128  # DSV4 window page


class FakeRing:
    """The carry ring, keyed the way the real backend keys it: (layer, tier, slot)."""

    def __init__(self):
        self.store: dict[tuple[int, str, int], torch.Tensor] = {}
        self.writes = 0

    def read_carry_blocks(self, layer_id, tier, window_slots, ring_size):
        return torch.stack([
            self.store.setdefault(
                (layer_id, tier, int(s)), torch.zeros(ring_size, 4)
            )
            for s in window_slots
        ])

    def write_carry_blocks(self, layer_id, tier, window_slots, ring_size, blocks):
        self.writes += 1
        for i, s in enumerate(window_slots):
            self.store[(layer_id, tier, int(s))] = blocks[i].clone()


class FakeCompressor:
    ring_size = 8


class FakeAttn:
    """A layer as the snapshot sees it: an id, and whichever tiers it owns."""

    def __init__(self, layer_id, *, compressor=True, indexer=True):
        self.layer_id = layer_id
        self.compressor = FakeCompressor() if compressor else None
        self.indexer = (
            type("Idx", (), {"compressor": FakeCompressor()})() if indexer else None
        )


def _advance(ring, layers, slots, value):
    """Stand-in for a decode step advancing the carry in place."""
    for attn in layers:
        for tier in ("attn", "idx"):
            for s in slots:
                ring.store[(attn.layer_id, tier, int(s))] = torch.full((8, 4), value)


class TestCarrySnapshot:
    def test_restore_undoes_an_in_place_advance(self):
        ring, layers, slots = FakeRing(), [FakeAttn(0), FakeAttn(1)], torch.tensor([3])
        _advance(ring, layers, slots, 1.0)          # state the accepted token left
        snap = CarrySnapshot(ring, layers, slots)
        _advance(ring, layers, slots, 9.0)          # speculative step overwrites it
        assert ring.store[(0, "attn", 3)][0, 0] == 9.0

        snap.restore()
        for attn in layers:
            for tier in ("attn", "idx"):
                assert torch.equal(
                    ring.store[(attn.layer_id, tier, 3)], torch.full((8, 4), 1.0)
                ), "the accepted token's carry must come back exactly"

    def test_the_snapshot_is_not_a_view_of_the_live_ring(self):
        # If read_carry_blocks handed back a view, the speculative write would corrupt
        # the snapshot itself and restore would be a no-op.
        ring, layers, slots = FakeRing(), [FakeAttn(0)], torch.tensor([2])
        _advance(ring, layers, slots, 1.0)
        snap = CarrySnapshot(ring, layers, slots)
        _advance(ring, layers, slots, 7.0)
        snap.restore()
        assert ring.store[(0, "attn", 2)][0, 0] == 1.0

    def test_restore_is_idempotent(self):
        ring, layers, slots = FakeRing(), [FakeAttn(0)], torch.tensor([1])
        _advance(ring, layers, slots, 4.0)
        snap = CarrySnapshot(ring, layers, slots)
        _advance(ring, layers, slots, 5.0)
        snap.restore()
        snap.restore()
        assert ring.store[(0, "attn", 1)][0, 0] == 4.0

    def test_layers_without_a_compressor_are_skipped(self):
        ring = FakeRing()
        layers = [FakeAttn(0, compressor=False, indexer=False), FakeAttn(1)]
        snap = CarrySnapshot(ring, layers, torch.tensor([0]))
        assert len(snap) == 2, "only layer 1's two tiers"

    def test_a_ratio_128_layer_saves_its_compressor_but_no_indexer(self):
        snap = CarrySnapshot(FakeRing(), [FakeAttn(0, indexer=False)], torch.tensor([0]))
        assert len(snap) == 1

    def test_every_row_of_a_batch_is_restored(self):
        ring, layers = FakeRing(), [FakeAttn(0)]
        slots = torch.tensor([4, 9])
        _advance(ring, layers, slots, 2.0)
        snap = CarrySnapshot(ring, layers, slots)
        _advance(ring, layers, slots, 8.0)
        snap.restore()
        assert ring.store[(0, "attn", 4)][0, 0] == 2.0
        assert ring.store[(0, "attn", 9)][0, 0] == 2.0


class TestNeedsRollback:
    def test_needed_when_the_rejection_shares_a_page_with_the_last_accepted_token(self):
        assert needs_rollback(2, torch.tensor([10, 11, 12, 13]), P) is True

    def test_not_needed_when_the_block_crossed_into_a_new_page(self):
        # The last accepted token is on page 0, the first rejected on page 1: the
        # abandoned page carries its own ring block away, and page 0 was never touched.
        assert needs_rollback(2, torch.tensor([126, 127, 128, 129]), P) is False

    def test_not_needed_when_nothing_was_rejected(self):
        assert needs_rollback(4, torch.tensor([1, 2, 3, 4]), P) is False

    def test_not_needed_when_the_whole_block_was_rejected(self):
        # No accepted token means the carry never advanced past the previous step.
        assert needs_rollback(0, torch.tensor([5, 6, 7]), P) is False
