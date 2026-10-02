"""DSV4 through the GENERIC CacheManager (CPU, no model) -- the unified serving path.

The shared page_table is the virtual full-token coordinate (ShadowRadix); DSV4PagedKVCache plugs
in as the swa_pool: window pages bind page-atomically behind token-face alloc_swa/free_swa, the
radix tree is the same SWARadixCache every SWA model uses, and conservation is the generic
check_integrity (free + tree == capacity, exact at idle). Covers: cold + radix-hit + decode
lifecycles with the out-of-window driver, the crash-A residue class, naive mode, chunked prefill,
and the decode-snapshot staging contract.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.scheduler.cache import CacheManager

DEVICE = torch.device("cpu")
P = 128
RATIOS = (0, 0, 4, 128, 4, 128, 4, 0)
MRR = 4


def _args():
    return DeepseekV4Args(
        n_layers=8, compress_ratios=RATIOS, max_seq_len=8192,
        head_dim=512, index_head_dim=128, window_size=P,
    )


def _stack(num_pages=32, swa_ratio=1.0, cache_type="swa_radix"):
    """(cm, pool, page_table): the engine wiring in miniature -- physical pool sized
    num_pages + 1 (tail dummy), page_table dummy row filled with the advertised token count."""
    args = _args()
    sizes = dsv4_pool_sizes(num_pages=num_pages + 1, args=args, swa_ratio=swa_ratio, P=P)
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE, P=P, n_scratch=MRR + 1)
    pool._init_paged_state(MRR, cache_type != "naive")
    pt = torch.zeros(MRR + 1, args.max_seq_len, dtype=torch.int32)
    pt[MRR].fill_(num_pages * P)                       # engine.py:218 dummy convention
    pool.full_loc_map = pt                             # attach_page_table
    cm = CacheManager(num_pages=num_pages, page_size=P, page_table=pt, type=cache_type,
                      swa_pool=pool, sliding_window_size=P)
    assert cm.swa_paged
    return cm, pool, pt


def _req(ti, ids, n_decode=4):
    r = Req(input_ids=ids.to(torch.int32), table_idx=ti, cached_len=0, output_len=n_decode,
            uid=ti, sampling_params=SamplingParams(), cache_handle=None)
    r.input_len = len(ids)
    return r


def _lifecycle(cm, req, total_len, finished=True):
    h = cm.match_req(req).cuda_handle
    req.cache_handle = h
    req.cached_len = h.cached_len
    cm.lock(h)
    if h.cached_len > 0:  # prefill.py:57: the matched full locs enter the page table
        cm.page_table[req.table_idx, : h.cached_len].copy_(
            h.get_matched_indices().to(torch.int32))
    cm.free_swa_out_of_window_extend([req])
    cm.allocate_paged([req])
    req.complete_one()
    i = 0
    while req.cached_len < total_len:
        req.append_host(torch.tensor([9000 + req.cached_len % 97], dtype=torch.int32))
        req.decode_batch_idx = i + 1
        cm.maybe_free_swa_out_of_window([req], forward_iter=i)
        cm.allocate_paged([req])
        req.complete_one()
        i += 1
    if finished:
        cm.cache_req(req, finished=True)
    return req


def test_cold_then_hit_reference_shares_and_conserves():
    cm, pool, pt = _stack()
    prompt = torch.arange(1, 301, dtype=torch.int32)

    _lifecycle(cm, _req(0, prompt, n_decode=310), total_len=310)
    cm.check_integrity()

    # Same prompt HITS; the matched prefix is tree-owned; window slots resolve via translate.
    r2 = _req(1, prompt.clone(), n_decode=320)
    h2 = cm.match_req(r2).cuda_handle
    assert h2.cached_len == 256
    _lifecycle(cm, r2, total_len=320)
    cm.check_integrity()
    # the shared chain is tree-owned exactly once; both requests' pages fully reconciled
    assert cm.prefix_cache.full_evictable + cm.prefix_cache.full_protected > 0


@pytest.mark.parametrize("cache_type", ["swa_radix", "naive"])
def test_crash_a_residue_class_conserves(cache_type):
    # The crash-A signature: total length a multiple of P must not strand a page in EITHER tier.
    for total in (3 * P, 3 * P + 1, 3 * P + P - 1, 4 * P):
        cm, _, _ = _stack(cache_type=cache_type)
        _lifecycle(cm, _req(0, torch.arange(1, total - 3, dtype=torch.int32), n_decode=total),
                   total_len=total)
        cm.check_integrity()


def test_long_decode_recycles_window_pages():
    # Decode far past the window: the out-of-window driver must keep the row's live window
    # footprint bounded (~window + margin), returning pages to the pool as it slides.
    cm, pool, pt = _stack(num_pages=48, swa_ratio=0.25)   # small window tier
    import freetoken.scheduler.cache as C
    old = C._SWA_EVICTION_INTERVAL
    C._SWA_EVICTION_INTERVAL = 1                          # evict every forward (test cadence)
    try:
        _lifecycle(cm, _req(0, torch.arange(1, 200, dtype=torch.int32), n_decode=14 * P),
                   total_len=14 * P)
    finally:
        C._SWA_EVICTION_INTERVAL = old
    cm.check_integrity()


def test_chunked_prefill_conserves():
    cm, _, _ = _stack(num_pages=48)
    total = 6 * P + 5
    req = _req(0, torch.arange(1, total + 1, dtype=torch.int32), n_decode=1)
    h = cm.match_req(req).cuda_handle
    req.cache_handle = h
    req.cached_len = h.cached_len
    cm.lock(h)
    chunk = 2 * P + 7
    while req.cached_len < total:
        req.device_len = min(req.cached_len + chunk, total)
        cm.free_swa_out_of_window_extend([req])
        cm.allocate_paged([req])
        req.cached_len = req.device_len
        req.device_len += 1
    cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_capability_surface_matches_scheduler_expectations():
    cm, pool, _ = _stack()
    # Prefill batching is unconditional: ragged prefill carries a per-request start_pos, so
    # cold, radix-hit and chunk-continuation segments mix freely under radix AND naive. The
    # only pool capability the scheduler still picks up is the prefill chunk budget.
    assert not hasattr(pool, "prefill_single_request")
    assert not hasattr(pool, "prefill_cold_batch_only")
    assert cm.prefill_chunk_budget == pool.prefill_chunk_budget > 0
    assert pool.sliding_window_size == P


def test_prefill_chunk_budget_follows_the_pool_across_a_rebuild():
    """The manager must report the pool's LIVE chunk cap, not the one it had at construction.

    ``Scheduler.rebuild_cache`` recomputes ``prefill_budget`` from
    ``cache_manager.prefill_chunk_budget`` after a runtime resize, so a value snapshotted in
    ``CacheManager.__init__`` makes the rebuild write back the number it already had. Growing
    the pool then leaves prompts chunked against the old, smaller cap; shrinking it leaves a cap
    too LARGE for the pool, and the next long prompt is chunked past what ``_alloc_window`` can
    satisfy.

    ``test_rebuild_cache_refreshes_prefill_budget`` covers the scheduler half with a
    ``SimpleNamespace`` manager whose cap the test sets by hand, so it cannot see this: the gap
    is between a real manager and its real pool.
    """
    cm, pool, _ = _stack(num_pages=32)
    before = cm.prefill_chunk_budget
    assert before == pool.prefill_chunk_budget > 0

    # What POST /v1/cache/rebuild does: resize the window pool in place. _init_paged_state
    # recomputes the pool's cap from the new window-slot count.
    pool.rebuild(dsv4_pool_sizes(num_pages=96 + 1, args=_args(), swa_ratio=1.0, P=P))
    grown = pool.prefill_chunk_budget
    assert grown > before, "inert test: the rebuild did not move the pool's cap"
    assert cm.prefill_chunk_budget == grown

    # Shrink is the dangerous direction: a stale cap is then too large for the pool.
    pool.rebuild(dsv4_pool_sizes(num_pages=24 + 1, args=_args(), swa_ratio=1.0, P=P))
    shrunk = pool.prefill_chunk_budget
    assert shrunk < before, "inert test: the shrink did not move the pool's cap"
    assert cm.prefill_chunk_budget == shrunk


def test_chunk_boundaries_stay_page_aligned_under_unaligned_budget():
    """Chunk continuations resume the compressor carry, so every minted chunk must END
    page-aligned -- even when the binding cap is an unaligned token-budget leftover. A leftover
    below one page must refuse admission without leaking."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    def _managers():
        cm, pool, pt = _stack(num_pages=48)
        tm = TableManager(max_running_reqs=MRR, page_table=pt)
        return cm, tm, PrefillManager(cm, tm, DecodeManager(page_size=P))

    def _pending(uid, n):
        return PendingReq(uid=uid, input_ids=(torch.arange(n, dtype=torch.int32) % 131) + 1,
                          sampling_params=SamplingParams(max_tokens=1))

    # 300-token cold prompt leaves an unaligned 724-token leftover; the 2000-token prompt's
    # chunk must round down to 640 (a page multiple), not 724.
    cm, tm, pm = _managers()
    pm.pending_list = [_pending(1, 300), _pending(2, 2000)]
    batch = pm.schedule_next_batch(1024)
    chunked = [r for r in batch.reqs if isinstance(r, ChunkedReq)]
    assert len(batch.reqs) == 2 and len(chunked) == 1
    assert chunked[0].device_len == 640 and chunked[0].device_len % P == 0

    # 100-token leftover cannot mint a whole page: the big prompt must NOT be admitted, and the
    # bailed admission must leak neither its table row nor its cache-handle lock.
    cm, tm, pm = _managers()
    pm.pending_list = [_pending(1, 300), _pending(2, 2000)]
    free_tables = tm.available_size
    batch = pm.schedule_next_batch(400)
    assert [r.uid for r in batch.reqs] == [1]
    assert tm.available_size == free_tables - 1          # only req 1 holds a row
    assert any(p.uid == 2 for p in pm.pending_list)      # retried next pass
    cm.check_integrity()


def test_abort_anywhere_fuzz_conserves_and_isolates():
    """admit / extend / finish-or-abort at arbitrary points with shared prefixes + eviction
    pressure, on a deliberately small window tier. The generic conservation check (free + tree
    == capacity, both currencies) must hold after EVERY finish -- the ported DSV4 abort fuzz."""
    import random

    rng = random.Random(20260722)
    cm, pool, pt = _stack(num_pages=40, swa_ratio=0.5)
    bank = [
        torch.cat([
            torch.tensor([9000 + i] * 3, dtype=torch.int32),
            (torch.arange(240 + 17 * i, dtype=torch.int32) % 131) + 1,
        ])
        for i in range(5)
    ]
    live: dict[int, Req] = {}
    it = 0
    for _ in range(240):
        op = rng.random()
        free_tis = [ti for ti in range(MRR) if ti not in live]
        if op < 0.45 and free_tis:
            prompt = (rng.choice(bank).clone() if rng.random() < 0.6 else torch.cat([
                torch.tensor([rng.randint(20000, 60000)] * 3, dtype=torch.int32),
                torch.randint(1, 200, (rng.randint(200, 360),), dtype=torch.int32),
            ]))
            if cm.available_size < len(prompt) + 8:
                continue                          # scheduler defers when the full tier is short
            ti = rng.choice(free_tis)
            req = _req(ti, prompt, n_decode=64)
            h = cm.match_req(req).cuda_handle
            req.cache_handle = h
            req.cached_len = h.cached_len
            need_swa = min(max(req.input_len - h.cached_len, 1), P) + 1
            if cm.swa_available_size < ((need_swa + P - 1) // P) * P:   # PrefillAdder swa gate
                continue
            cm.lock(h)
            if h.cached_len > 0:
                pt[ti, : h.cached_len].copy_(h.get_matched_indices().to(torch.int32))
            cm.free_swa_out_of_window_extend([req])
            cm.allocate_paged([req])
            req.complete_one()
            live[ti] = req
        elif op < 0.75 and live:                  # decode a random live request
            ti = rng.choice(list(live))
            req = live[ti]
            if cm.available_size < 2 or cm.swa_available_size < 2 * P:
                continue
            req.append_host(torch.tensor([rng.randint(1, 200)], dtype=torch.int32))
            req.decode_batch_idx += 1
            cm.maybe_free_swa_out_of_window([req], forward_iter=it)
            cm.allocate_paged([req])
            req.complete_one()
        elif live:                                # finish / ABORT at whatever state it is in
            ti = rng.choice(list(live))
            cm.cache_req(live.pop(ti), finished=True)
            assert (pt[ti] >= 0).all()            # no negative locs ever enter the shared table
        it += 1
        if not live:                              # idle points: exact conservation, both tiers
            cm.check_integrity()
    for ti in list(live):
        cm.cache_req(live.pop(ti), finished=True)
    cm.check_integrity()
