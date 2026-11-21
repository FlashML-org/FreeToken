"""SWASnapshotRadixCache: what the snapshot currency adds to the SWA radix.

A boundary is resumable only with a snapshot there *and* a live window in front of it; window KV
no snapshot reads is trimmed; a window eviction frees the snapshots that read it; a snapshot
eviction removes the leaf it alone kept reachable. The SWA machinery underneath is covered by
``test_swa_radix.py``. Expectations are closed-form positions on a page_size-1 tree with a
4-token window; every scenario ends with the class's own integrity check.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.swa_snapshot_radix_cache import SWASnapshotRadixCache

W = 4


@pytest.fixture
def cache():
    c = SWASnapshotRadixCache(torch.device("cpu"), page_size=1, sliding_window_size=W)
    yield c
    c.check_integrity()


def _ids(n: int) -> torch.Tensor:
    return torch.arange(1, n + 1, dtype=torch.int32)


def _kv(n: int) -> torch.Tensor:
    return torch.arange(100, 100 + n, dtype=torch.int32)


def _live(c: SWASnapshotRadixCache, n: int) -> list[bool]:
    """Per position of the path ``_ids(n)``: is its window KV live?"""
    out, node, pos = [], c.root, 0
    ids = _ids(n)
    while pos < n:
        node = node.children[c.key_fn(ids[pos:])]
        out += [not node.swa_tombstone] * node.length
        pos += node.length
    return out


def test_a_prefix_resumes_only_at_a_snapshot_boundary(cache):
    cache.insert(_ids(12), _kv(12))
    cold = cache.match_prefix(_ids(12))
    assert (cold.cached_len, cold.mamba_value, cold.raw_len) == (0, None, 12)
    assert cache.attach_snapshot(_ids(8), 7) is False
    hit = cache.match_prefix(_ids(12))
    assert (hit.cached_len, hit.mamba_value, hit.raw_len) == (8, 7, 12)
    assert hit.kv_indices.tolist() == list(range(100, 108))
    assert cache.match_prefix(_ids(6)).cached_len == 0


def test_a_snapshot_needs_its_window_live(cache):
    # positions [0, 6) slid out of the request's window before the commit
    cache.insert(_ids(12), _kv(12), swa_evicted_seqlen=6)
    assert cache.attach_snapshot(_ids(9), 1) is True  # window [5, 9) reaches a tombstone
    assert cache.attach_snapshot(_ids(10), 2) is False
    assert cache.match_prefix(_ids(12)).cached_len == 10


def test_a_second_snapshot_at_a_boundary_or_one_off_the_tree_is_refused(cache):
    cache.insert(_ids(8), _kv(8))
    assert cache.attach_snapshot(_ids(8), 1) is False
    assert cache.attach_snapshot(_ids(8), 2) is True
    assert cache.attach_snapshot(torch.tensor([9, 9], dtype=torch.int32), 3) is True
    assert cache.match_prefix(_ids(8)).mamba_value == 1


def test_trim_keeps_only_the_windows_in_front_of_snapshots(cache):
    cache.insert(_ids(16), _kv(16))
    cache.attach_snapshot(_ids(8), 1)
    cache.attach_snapshot(_ids(16), 2)
    freed = cache.trim_swa(_ids(16))
    assert sorted(freed.tolist()) == [100, 101, 102, 103, 108, 109, 110, 111]
    assert _live(cache, 16) == [False] * 4 + [True] * 4 + [False] * 4 + [True] * 4
    assert cache.match_prefix(_ids(16)).cached_len == 16
    # a hit between the two snapshot points falls back to the earlier one
    assert cache.match_prefix(_ids(12)).cached_len == 8


def test_trim_leaves_a_locked_window_alone(cache):
    cache.insert(_ids(12), _kv(12))
    cache.walk_path(_ids(8))  # the commit splits a window back so the lock covers that window alone
    node, _ = cache.walk_path(_ids(12))
    uuid = cache.inc_lock(node)  # a running request's trailing window, no snapshot yet
    assert cache.trim_swa(_ids(12)).tolist() == list(range(100, 108))
    assert _live(cache, 12) == [False] * 8 + [True] * 4
    cache.dec_lock(node, uuid)


def test_a_window_eviction_frees_the_snapshots_that_read_it(cache):
    cache.insert(_ids(12), _kv(12))
    cache.attach_snapshot(_ids(8), 1)
    cache.attach_snapshot(_ids(12), 2)
    node, _ = cache.walk_path(_ids(12))
    uuid = cache.inc_lock(node)  # pins the window [8, 12) of the deeper snapshot
    ev = cache.evict_swa(100)
    assert ev.mamba_slots == [1] and ev.kv_indices.numel() == 0
    assert sorted(ev.swa_indices.tolist()) == list(range(100, 108))
    assert cache.match_prefix(_ids(12)).cached_len == 12
    cache.dec_lock(node, uuid)


def test_a_snapshot_eviction_removes_the_leaf_it_kept_reachable(cache):
    cache.insert(_ids(8), _kv(8))
    cache.attach_snapshot(_ids(4), 1)
    cache.attach_snapshot(_ids(8), 2)
    first = cache.evict_mamba(1)  # the older snapshot sits on an internal node: only its slot goes
    assert (first.mamba_slots, first.kv_indices.numel()) == ([1], 0)
    second = cache.evict_mamba(1)  # the leaf and the snapshot-less node above it go whole
    assert second.mamba_slots == [2]
    assert sorted(second.kv_indices.tolist()) == sorted(second.swa_indices.tolist()) == list(range(100, 108))
    assert cache.root.is_leaf()


def test_a_locked_snapshot_survives_eviction(cache):
    cache.insert(_ids(8), _kv(8))
    cache.attach_snapshot(_ids(8), 1)
    node, _ = cache.walk_path(_ids(8))
    assert cache.inc_mamba(node) is True
    assert cache.evict_mamba(10).mamba_slots == []
    uuid = cache.inc_lock(node)
    assert cache.evict_full(100).mamba_slots == []
    cache.dec_lock(node, uuid)
    cache.dec_mamba(node)
    ev = cache.evict_full(100)
    assert ev.mamba_slots == [1] and ev.kv_indices.tolist() == list(range(100, 108))
