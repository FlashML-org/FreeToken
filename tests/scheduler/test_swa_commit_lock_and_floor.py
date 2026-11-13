"""Two coupled properties of the prefill-boundary commit.

1. It must hand ``inc_lock`` ONE window. ``cache_req(finished=False)`` inserts the forwarded
   prompt as a single suffix node and locks it for the rest of decode (the decode driver floors
   its own frees at the committed length, so the request cannot reclaim it either). inc_lock is
   node-granular, so without a node boundary a window back it pins the entire last chunk's swa --
   and PrefillAdder sizes the chunk to fill the pool, so a long prompt hands its whole pool to
   the lock and the next decode step raises "SWA pool exhausted", which nothing handles. Whether
   it fires depends on the final chunk's length, i.e. on prompt % chunk.

2. The pool floor must then cover ``max_running_req`` of those footprints: admission gates only
   the incoming chunk and never reserves the decode growth of the requests already running.

CPU-only, no model: real PrefillManager/PrefillAdder chunking (so the cap is the shipped one)
in the scheduler's overlap order, then the decode driver.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.hybrid_swa_pool import (
    _swa_paged_num_tokens,
    _swa_per_req_swa_floor,
    _swa_pool_floor,
)
from freetoken.models.config import KVCacheGroupSpec
from freetoken.scheduler import cache as cache_mod
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

DEVICE = torch.device("cpu")
MAX_RUNNING = 4
UID = 7
TOKEN_BUDGET = 8192   # max_extend_tokens: never the binding cap here, the swa pool is
GAP = cache_mod._SWA_RETAIN_GAP

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


def _cfg(window: int, page_size: int = 1, max_running_req: int = 4, **kw):
    """The slice of EngineConfig the floor helpers read."""
    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                         sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                         sliding_window=window),
    )
    return SimpleNamespace(
        page_size=page_size, max_running_req=max_running_req,
        decode_rows_per_req=kw.get("rows", 1),
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: groups),
        swa_num_pages_override=kw.get("override"), swa_full_tokens_ratio=kw.get("ratio", 0.2),
        special_token_ckpt=kw.get("ckpt", False),
    )


def _managers(window: int, num_swa_tokens: int, ps: int = 1, num_pages: int = 4096, width=2048):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=ps))

    pool = HybridSWAKVCache(
        groups=_cfg(window, ps).model_config.kv_cache_group_specs(), num_layers=2,
        num_full_pages=num_pages, page_size=ps, dtype=torch.bfloat16, device=DEVICE,
        num_swa_tokens=num_swa_tokens,
    )
    pt = torch.zeros((MAX_RUNNING + 1, width), dtype=torch.int32, device=DEVICE)
    cm = CacheManager(num_pages=num_pages, page_size=ps, page_table=pt, type="swa_radix",
                      swa_pool=pool, sliding_window_size=window)
    assert cm.swa_paged and cm.is_swa
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    return cm, tm, PrefillManager(cm, tm, DecodeManager(page_size=ps))


def _prefill(cm, pm, prompt_len: int, n_decode: int, base: int = 1):
    """Real chunked prefill (chunk sizes from PrefillAdder) in the scheduler's overlap order:
    schedule+forward chunk N+1 before committing chunk N; only the final chunk commits.
    ``base`` shifts the token ids so concurrent requests share no prefix."""
    pm.pending_list = [PendingReq(uid=UID,
                                  input_ids=torch.arange(base, base + prompt_len,
                                                         dtype=torch.int32),
                                  sampling_params=SamplingParams(max_tokens=n_decode))]
    final = last_batch = None
    while pm.runnable or last_batch is not None:
        batch = pm.schedule_next_batch(TOKEN_BUDGET)
        if batch is not None:
            assert batch.reqs[0].extend_len > 0, "prefill stalled at a zero-length chunk"
            cm.free_swa_out_of_window_extend(batch.reqs)     # _prepare_batch
            cm.allocate_paged(batch.reqs)
            for r in batch.reqs:
                r.complete_one()
        if last_batch is not None:
            for r in last_batch.reqs:
                if not isinstance(r, ChunkedReq):
                    cm.cache_req(r, finished=False)          # scheduler.py:328
                    final = r
        last_batch = batch
    return final


def _decode(cm, reqs, n: int):
    """Shared decode loop in the scheduler's order: the out-of-window free runs on every
    decode step and frees once the request has slid out a full eviction interval."""
    for i in range(n):
        for r in reqs:
            r.append_host(torch.tensor([9999], dtype=torch.int32))
            r.decode_batch_idx = i + 1
        cm.maybe_free_swa_out_of_window(reqs)
        cm.allocate_paged(reqs)
        for r in reqs:
            r.complete_one()


# --------------------------------------------------------------- what the commit locks
@pytest.mark.parametrize("ps", [1, 8, 128])
def test_commit_locks_one_window_not_the_whole_extend(ps, monkeypatch):
    """The lock covers the retained window (page-rounded); the head stays live and unlocked, so
    ambient pressure can still reclaim it. ps == window == 128 is the DSV4 shape, where the
    boundary must also stay page-aligned for the pool's whole-page free path."""
    monkeypatch.setattr(cache_mod, "_SWA_EVICTION_INTERVAL", 1)
    window, prompt = (8, 200) if ps == 1 else (ps, 24 * ps)
    cm, _tm, pm = _managers(window, num_swa_tokens=64 * max(ps, 8) + 1, ps=ps,
                            num_pages=256 if ps > 1 else 4096, width=64 * max(ps, 32))
    _prefill(cm, pm, prompt, n_decode=1)

    retained = -(-(window + GAP) // ps) * ps
    assert cm.prefix_cache.swa_protected == retained
    assert cm.prefix_cache.swa_evictable == prompt - retained
    cm.prefix_cache.check_integrity()


def test_short_final_chunk_locks_no_more_than_a_window(monkeypatch):
    """The boundary is the retain gap, not the window: the committed live region is
    [L - c_last - window - 1, L), so keep_from (L - window - gap) only falls inside it once
    c_last > gap. Below that the split lands in the tombstoned head and does nothing -- and does
    not need to, the node is already short. Either way the lock stays <= window + gap."""
    monkeypatch.setattr(cache_mod, "_SWA_EVICTION_INTERVAL", 1)
    window, first_chunk, n_decode = 8, 100, 40
    for c_last in (1, 14, 15, 16, 17, 60):
        cm, tm, _pm = _managers(window, num_swa_tokens=4096)
        total = first_chunk + c_last
        req = Req(input_ids=torch.arange(1, total + 1, dtype=torch.int32), table_idx=0,
                  cached_len=0, output_len=n_decode, uid=UID,
                  sampling_params=SamplingParams(), cache_handle=None)
        req.input_len = total
        h = cm.match_req(req).cuda_handle
        req.cache_handle = h
        cm.lock(h)
        for end in (first_chunk, total):        # two explicit chunks -> c_last is exact
            req.device_len = end
            cm.free_swa_out_of_window_extend([req])
            cm.allocate_paged([req])
            req.cached_len = end
            req.device_len = end + 1
        live_node = total - req.swa_evicted_seqlen
        cm.cache_req(req, finished=False)

        assert cm.prefix_cache.swa_protected == min(live_node, window + GAP), f"c_last={c_last}"
        _decode(cm, [req], n_decode)
        cm.cache_req(req, finished=True)
        tm.free(req.table_idx)
        cm.check_integrity()


# --------------------------------------------------------------- what that buys at runtime
@pytest.mark.parametrize("interval", [1, 128])
def test_decode_survives_any_prompt_length(interval, monkeypatch):
    """The pool holds the window working set and nothing chunk-sized, at both an eviction every
    position and the shipped 128-position one (where it must also absorb a full interval of decode
    growth before the first reclaim). The prompt length -- hence the final chunk's -- must not
    decide whether decode completes."""
    monkeypatch.setattr(cache_mod, "_SWA_EVICTION_INTERVAL", interval)
    window, n_decode = 8, 300
    pool = 2 * window + GAP + interval + 8
    for prompt in (103, 129, 200, 400):
        cm, tm, pm = _managers(window, num_swa_tokens=pool + 1)
        req = _prefill(cm, pm, prompt, n_decode)
        _decode(cm, [req], n_decode)
        cm.cache_req(req, finished=True)
        tm.free(req.table_idx)
        cm.check_integrity()


@pytest.mark.parametrize("n_req", [1, 3])
def test_a_full_batch_decodes_on_a_pool_sized_to_the_declared_floor(n_req, monkeypatch):
    """Ties the sizing formula to the runtime: open the pool at exactly _swa_pool_floor and run
    max_running_req requests through prefill into a shared decode. All are committed before any
    decode, so every one holds its full non-evictable footprint at once."""
    monkeypatch.setattr(cache_mod, "_SWA_EVICTION_INTERVAL", 1)
    window, n_decode = 8, 200
    floor = _swa_pool_floor(_cfg(window, max_running_req=n_req))
    cm, tm, pm = _managers(window, num_swa_tokens=floor + 1)
    reqs = [_prefill(cm, pm, 120 + 7 * i, n_decode, base=1 + 10_000 * i) for i in range(n_req)]
    _decode(cm, reqs, n_decode)
    for r in reqs:
        cm.cache_req(r, finished=True)
        tm.free(r.table_idx)
    cm.check_integrity()


def test_a_per_request_sized_pool_cannot_hold_a_full_batch(monkeypatch):
    """Why the floor carries the max_running_req factor. Two manifestations, both fatal and
    neither handled: alloc_swa raising once the pool is dry, or -- reached first here --
    PrefillAdder's swa cap collapsing a continuation chunk to zero tokens (prefill.py:138),
    which builds a Req with device_len == cached_len. Continuations take the chunked branch of
    try_add_one, so the admission gate never sees them."""
    monkeypatch.setattr(cache_mod, "_SWA_EVICTION_INTERVAL", 1)
    window, n_decode = 8, 200
    per_req = _swa_per_req_swa_floor(_cfg(window, max_running_req=3))
    cm, _tm, pm = _managers(window, num_swa_tokens=per_req + 1)
    with pytest.raises((RuntimeError, AssertionError)):
        reqs = [_prefill(cm, pm, 120, n_decode, base=1 + 10_000 * i) for i in range(3)]
        _decode(cm, reqs, n_decode)


# --------------------------------------------------------------- the floor formula
def test_floor_terms_and_page_rounding():
    interval = cache_mod._SWA_EVICTION_INTERVAL
    for window in (8, 128, 1024):
        for ps in (1, 8, 64, 128):
            for rows in (1, 4):
                for ckpt in (False, True):
                    locked = -(-(window + GAP) // ps) * ps       # what the commit locks
                    # uncollected decode tail: an interval of forward rows + one step's rows
                    tail = window + 2 * ps + interval * rows + rows
                    # the special-token checkpoint holds one more window + gap of positions
                    extra = window + GAP + interval * rows if ckpt else 0
                    assert _swa_per_req_swa_floor(_cfg(window, ps, rows=rows, ckpt=ckpt)) == locked + tail + extra, (
                        window, ps, rows, ckpt)


def test_pool_floor_scales_with_max_running_req():
    for n in (1, 2, 4, 8, 32):
        cfg = _cfg(128, max_running_req=n)
        assert _swa_pool_floor(cfg) == n * _swa_per_req_swa_floor(cfg)


def test_ratio_and_override_never_dip_below_the_floor():
    cfg = _cfg(128, max_running_req=4)
    # Small full pool (ratio x full << floor) -> the floor, +1 slot-0 sentinel.
    assert _swa_paged_num_tokens(cfg, num_full_pages=64) == _swa_pool_floor(cfg) + 1
    # Generous full pool -> the ratio wins.
    assert _swa_paged_num_tokens(cfg, num_full_pages=1 << 20) == int(0.2 * (1 << 20)) + 1
    # A pinned window (the rebuild path only validates num_swa_pages > 0) is clamped up.
    pinned = _cfg(128, max_running_req=4, override=1)
    assert _swa_paged_num_tokens(pinned, num_full_pages=1024) == _swa_pool_floor(pinned) + 1
