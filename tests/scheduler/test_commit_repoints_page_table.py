"""An unfinished commit that dedups against a prefix another request already published frees
the request's OWN pages for the shared span. Its page-table row must then name the tree's pages
instead: the attention backends read that row every decode step, and the freed pages go to the
next allocation. CPU, real CacheManager + real trees, no engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager

PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def _admit(cm, page_table, table_idx, ids, handle):
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
              cached_len=0, output_len=0, uid=table_idx, sampling_params=SamplingParams(),
              cache_handle=handle)
    req.device_len = len(ids)
    cm.lock(handle)
    cm.allocate_paged([req])
    req.cached_len = len(ids)
    return req


def _live_row(page_table, req):
    return set(page_table[req.table_idx, : req.cached_len].tolist())


def test_radix_unfinished_commit_repoints_the_row_off_the_freed_pages():
    page_table = torch.zeros(4, 32, dtype=torch.int32)
    cm = CacheManager(32, 1, page_table, "radix")

    a = _admit(cm, page_table, 0, PROMPT, cm.match_req(_pend(PROMPT)).cuda_handle)
    b = _admit(cm, page_table, 1, PROMPT, cm.match_req(_pend(PROMPT)).cuda_handle)
    assert _live_row(page_table, a).isdisjoint(_live_row(page_table, b))

    with cm.lazy_free_region():          # the scheduler drains commits inside this region
        cm.cache_req(a, finished=False)
        cm.cache_req(b, finished=False)

    free = set(cm.free_slots.tolist())
    # b's own duplicate pages went back to the pool ...
    assert free, "the later committer's duplicate pages should have been freed"
    # ... and no page b still reads is on the free list.
    assert _live_row(page_table, b).isdisjoint(free)
    assert _live_row(page_table, b) == set(b.cache_handle.get_matched_indices().tolist())


def test_hybrid_unfinished_commit_repoints_the_row_off_the_freed_pages():
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    pool = LinearStatePool(group=g, num_slots=16, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    page_table = torch.zeros(4, 32, dtype=torch.int32)
    cm = CacheManager(32, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    reqs = []
    for idx in (0, 1):
        r = _admit(cm, page_table, idx, PROMPT, cm.match_req(_pend(PROMPT)).cuda_handle)
        r.linear_slot_idx = pool.alloc(1)[0]
        r.mamba_ping_pong = tuple(pool.alloc(2))
        r.mamba_next_track_idx = 1
        r.mamba_last_track_seqlen = len(PROMPT)
        reqs.append(r)

    with cm.lazy_free_region():
        for r in reqs:
            cm.cache_req(r, finished=False)

    free = set(cm.free_slots.tolist())
    assert free, "the later committer's duplicate pages should have been freed"
    assert _live_row(page_table, reqs[1]).isdisjoint(free)


def test_lazy_free_snapshots_the_rows_it_was_handed():
    """The deferred free list must not be rewritten by a later re-point of the same row."""
    page_table = torch.zeros(2, 8, dtype=torch.int32)
    cm = CacheManager(8, 1, page_table, "radix")
    page_table[0, :4] = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
    before = cm.free_slots.clone()

    with cm.lazy_free_region():
        cm._free(page_table[0, :4])
        page_table[0, :4] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)  # a re-point

    appended = cm.free_slots[len(before):].tolist()
    assert appended == [4, 5, 6, 7]


def test_radix_subspan_commit_repoints_only_the_deduped_slice():
    """Same defect with old_handle.cached_len > 0: the committer admitted on top of a
    published prefix, so the dedup free and the re-point cover only the sub-span
    [old_cached, new_cached) -- the slice arithmetic the zero-prefix tests never touch."""
    LONG = list(range(1, 17))
    SHORT = LONG[:8]
    page_table = torch.zeros(4, 32, dtype=torch.int32)
    cm = CacheManager(32, 1, page_table, "radix")

    seed = _admit(cm, page_table, 0, SHORT, cm.match_req(_pend(SHORT)).cuda_handle)
    with cm.lazy_free_region():
        cm.cache_req(seed, finished=False)

    def _admit_on_prefix(table_idx):
        m = cm.match_req(_pend(LONG))
        matched = m.cuda_handle.cached_len
        assert matched > 0, "the seeded prefix should match"
        req = Req(input_ids=torch.tensor(LONG, dtype=torch.int32), table_idx=table_idx,
                  cached_len=matched, output_len=0, uid=table_idx,
                  sampling_params=SamplingParams(), cache_handle=m.cuda_handle)
        req.device_len = len(LONG)
        cm.lock(m.cuda_handle)
        page_table[table_idx, :matched] = m.cuda_handle.get_matched_indices()[:matched]
        cm.allocate_paged([req])          # only [matched, 16) -- the row prefix is canonical
        req.cached_len = len(LONG)
        return req, matched

    b, matched = _admit_on_prefix(1)
    d, _ = _admit_on_prefix(2)
    own_suffix_d = set(page_table[2, matched:].tolist())

    with cm.lazy_free_region():
        cm.cache_req(b, finished=False)   # b publishes [matched, 15)
        cm.cache_req(d, finished=False)   # d dedups against b: frees its own sub-span

    free = set(cm.free_slots.tolist())
    assert free & own_suffix_d, "d's duplicate sub-span pages should have been freed"
    assert _live_row(page_table, d).isdisjoint(free)
    canonical = d.cache_handle.get_matched_indices()
    assert page_table[2, : d.cache_handle.cached_len].tolist() == canonical[: d.cache_handle.cached_len].tolist()
    cm.check_integrity()
