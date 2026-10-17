"""An unfinished commit that dedups against a prefix another request already published frees
the request's OWN pages for the shared span. Its page-table row must then name the tree's pages
instead: the attention backends read that row every decode step, and the freed pages go to the
next allocation. CPU, real CacheManager + real trees, no engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import KVCacheGroupSpec, LinearGatedDeltaGroupConfig
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


def _swa_cache_manager(window, num_pages=256):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))
    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                         sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                         sliding_window=window),
    )
    pool = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=num_pages, page_size=1,
                            dtype=torch.bfloat16, device=torch.device("cpu"),
                            num_swa_tokens=num_pages)
    page_table = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(num_pages, 1, page_table, "swa_radix", swa_pool=pool,
                      sliding_window_size=window)
    return cm, page_table, pool


def _admit_swa(cm, page_table, table_idx, ids):
    handle = cm.match_req(_pend(ids)).cuda_handle
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
              cached_len=handle.cached_len, output_len=0, uid=table_idx,
              sampling_params=SamplingParams(), cache_handle=handle)
    cm.lock(handle)
    page_table[table_idx, : handle.cached_len] = handle.get_matched_indices()
    cm.allocate_paged([req])
    req.cached_len = len(ids)
    return req


def test_swa_unfinished_commit_keeps_its_pages_under_a_locked_tombstone():
    """SWA variant where there is nothing safe to re-point to. c full-locks a shared prefix whose
    swa a finishing request trims; b then prefills that prefix itself plus a tail shorter than
    the window. insert keeps the locked tombstone and frees b's copy, but the windowed re-match
    stops before the tombstone, so b's row named freed pages and b's finish freed them again
    (issue #204: duplicate pages in free_slots, then the SWA-slot integrity assert)."""
    window = 16
    cm, page_table, pool = _swa_cache_manager(window)
    shared = list(range(100, 164))
    c = _admit_swa(cm, page_table, 0, shared + list(range(300, 340)))
    with cm.lazy_free_region():
        cm.cache_req(c, finished=False)
    a = _admit_swa(cm, page_table, 1, shared + list(range(400, 440)))
    assert a.cache_handle.cached_len == len(shared)
    with cm.lazy_free_region():
        cm.cache_req(a, finished=True)    # trims the shared head's swa; c still full-locks it
    assert any(n.swa_tombstone and n.ref_count > 0 for n in cm.prefix_cache._all_nodes())

    b = _admit_swa(cm, page_table, 2, shared + list(range(500, 508)))
    assert b.cache_handle.cached_len == 0
    with cm.lazy_free_region():
        cm.cache_req(b, finished=False)

    assert _live_row(page_table, b).isdisjoint(set(cm.free_slots.tolist())), "b reads freed pages"
    in_window = page_table[b.table_idx, b.cached_len - window : b.cached_len].long()
    assert bool((pool.full_to_swa_index_mapping[in_window] != 0).all()), "b's window lost its swa"

    for r in (b, c):
        with cm.lazy_free_region():
            cm.cache_req(r, finished=True)
    assert cm.free_slots.numel() == torch.unique(cm.free_slots).numel(), "pages freed twice"
    cm.check_integrity()
