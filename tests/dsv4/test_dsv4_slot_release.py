"""A rejected block must give its pages and SWA slots back.

allocate_paged sizes itself from ``req.device_len``, so a speculative step allocates for
the whole block before it knows how much of it survives. Acceptance then lowers
device_len to the prefix it kept. Nothing else reclaims the difference -- the per-step
window free walks the live range, and the end-of-request free walks the page table from
the request's CURRENT length, so the abandoned tail is outside both.

The leak never fails a decode. It surfaces at the next idle integrity check, with a
message that names neither speculation nor the request that leaked:

    AssertionError: SWA-slot leak/double-free: free(11520) + tree(256) != capacity(12160)

These tests fail at the cause instead: conservation across an accept, a partial reject,
and a total reject.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")


class _Req:
    def __init__(self, table_idx=0, device_len=0):
        self.table_idx = table_idx
        self.device_len = device_len


class _Pool:
    """Counts slots in and out. Conservation is the whole property under test."""

    def __init__(self):
        self.live: set[int] = set()

    def alloc_swa(self, idx):
        for i in idx.tolist():
            assert i not in self.live, f"double-alloc of swa slot {i}"
            self.live.add(i)

    def free_swa(self, idx):
        for i in idx.tolist():
            assert i in self.live, f"double-free of swa slot {i}"
            self.live.discard(i)


def _manager(page_size=1, capacity=64):
    """A CacheManager with only the fields release_speculative_tail touches."""
    from freetoken.scheduler.cache import CacheManager

    m = CacheManager.__new__(CacheManager)
    m.page_size = page_size
    m.swa_paged = True
    m.is_swa = True
    m.swa_pool = _Pool()
    m.device = torch.device("cpu")
    m.free_slots = torch.empty(0, dtype=torch.int32)
    m.page_table = torch.zeros(1, capacity, dtype=torch.int32)
    return m


def _occupy(m, req, lo, hi):
    """Pretend allocate_paged handed positions [lo, hi) the slots [lo, hi)."""
    slots = torch.arange(lo, hi, dtype=torch.int32)
    m.page_table[req.table_idx, lo:hi] = slots
    m.swa_pool.alloc_swa(slots)
    req.device_len = hi


class TestConservation:
    def test_a_fully_rejected_block_returns_every_slot(self):
        m, req = _manager(), _Req()
        _occupy(m, req, 0, 1)          # the committed token
        _occupy(m, req, 1, 6)          # a 5-wide block
        m.release_speculative_tail(req, 1)
        assert m.swa_pool.live == {0}, "only the committed token should still hold a slot"
        assert m.free_slots.tolist() == [1, 2, 3, 4, 5]

    def test_a_partly_accepted_block_returns_only_the_tail(self):
        m, req = _manager(), _Req()
        _occupy(m, req, 0, 1)
        _occupy(m, req, 1, 6)
        m.release_speculative_tail(req, 3)   # kept 2 positions plus the bonus
        assert sorted(m.swa_pool.live) == [0, 1, 2]
        assert m.free_slots.tolist() == [3, 4, 5]

    def test_a_fully_accepted_block_returns_nothing(self):
        m, req = _manager(), _Req()
        _occupy(m, req, 0, 6)
        m.release_speculative_tail(req, 6)
        assert len(m.swa_pool.live) == 6
        assert m.free_slots.numel() == 0

    def test_release_is_a_no_op_above_the_current_length(self):
        # Defensive: a caller that passes a length beyond what was allocated must not
        # free slots the request never held.
        m, req = _manager(), _Req()
        _occupy(m, req, 0, 4)
        m.release_speculative_tail(req, 10)
        assert len(m.swa_pool.live) == 4
        assert m.free_slots.numel() == 0

    def test_an_unallocated_request_is_skipped(self):
        m, req = _manager(), _Req(table_idx=-1, device_len=5)
        m.release_speculative_tail(req, 1)   # must not index page_table[-1]
        assert m.free_slots.numel() == 0

    @pytest.mark.parametrize("keep", [1, 2, 3, 4, 5, 6])
    def test_every_prefix_conserves_the_total(self, keep):
        m, req = _manager(), _Req()
        _occupy(m, req, 0, 6)
        m.release_speculative_tail(req, keep)
        assert len(m.swa_pool.live) + m.free_slots.numel() == 6, (
            "every slot must be either still held or back on the free list"
        )


class TestPagedGeometry:
    def test_free_slots_receives_one_entry_per_page(self):
        # free_slots holds page-START slots. Pushing every token slot would inflate the
        # free list and hand out overlapping pages on the next allocation.
        m, req = _manager(page_size=4), _Req()
        _occupy(m, req, 0, 16)
        m.release_speculative_tail(req, 4)
        assert m.free_slots.tolist() == [4, 8, 12]
        assert len(m.swa_pool.live) == 4, "all 12 freed token slots leave the swa pool"


class TestItIsActuallyCalled:
    """The release exists only if the acceptance path invokes it."""

    def test_the_engine_releases_before_lowering_device_len(self):
        src = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python" / "freetoken" / "engine" / "engine.py"
        ).read_text()
        body = src[src.index("def _finish_speculative") :]
        body = body[: body.index("\n    def ", 10)]
        rel = body.index("release_tail(req")
        lower = body.index("req.cached_len, req.device_len = keep")
        assert rel < lower, (
            "release_speculative_tail reads req.device_len to find the range to return, "
            "so it must run BEFORE acceptance lowers it"
        )

    def test_the_scheduler_supplies_it(self):
        src = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python" / "freetoken" / "scheduler" / "scheduler.py"
        ).read_text()
        assert "batch.release_tail = self.cache_manager.release_speculative_tail" in src
