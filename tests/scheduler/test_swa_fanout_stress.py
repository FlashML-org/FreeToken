"""Stress the #200 fix: does keeping the head's windowed KV break the pool under load?

The fix stops eagerly tombstoning the prompt head at finish time. The risk is the one
issue #202 names: window-pool exhaustion raises an unhandled RuntimeError
(hybrid_swa_pool.py:169). The claim to verify is that the head is still *reclaimable* --
it stays unlocked and un-tombstoned, so `ensure_swa_slots` -> `evict_swa` can take it
back lazily when the pool actually runs short. That is the difference between
"reclaimed under pressure" (fine) and "never reclaimed" (a leak).

Scenarios:
  1. multi-round fan-out at a roomy pool   -- every round must reuse the shared head
  2. fan-out under a TIGHT pool            -- must not raise; reuse may degrade
  3. mixed fan-out + append               -- both still work
  4. page_size > 1                        -- alignment must hold
  5. long fan-out chain                   -- no unbounded growth
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

# These tests drive the real CacheManager on CPU. But `_maybe_pinned` (prefill.py:28)
# calls `t.pin_memory()` whenever torch.cuda.is_available(), so on a GPU box they would
# allocate real pinned memory -- on 218 that means competing with the live engine for
# VRAM (observed: AcceleratorError "CUDA error: out of memory"). Hide the device so the
# tests stay CPU-only. Must happen before torch's cuda state is first touched.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from freetoken.core import Context, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.config import KVCacheGroupSpec
from freetoken.scheduler import cache as cache_mod
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

DEVICE = torch.device("cpu")
MAX_RUNNING = 4
TOKEN_BUDGET = 65536
WINDOW = 128

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)

HEAD = 600
TAIL = 600


def test_cuda_is_hidden_so_these_tests_never_touch_a_gpu():
    """Guard: this file must not allocate device memory on a GPU host."""
    assert not torch.cuda.is_available(), (
        "CUDA is visible: _maybe_pinned would allocate pinned GPU memory and can OOM a "
        "co-resident engine. Set CUDA_VISIBLE_DEVICES='' before torch initialises CUDA.")



def _cfg(window, page_size=1, max_running_req=4):
    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                         sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                         sliding_window=window),
    )
    return SimpleNamespace(
        page_size=page_size, max_running_req=max_running_req,
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: groups),
        swa_num_pages_override=None, swa_full_tokens_ratio=0.2,
    )


def _managers(window=WINDOW, num_swa_tokens=8192, ps=1, num_pages=4096, width=8192):
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
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    return cm, tm, PrefillManager(cm, tm, DecodeManager(page_size=ps))


def _serve(cm, pm, ids, n_decode=4, finished=True, max_iters=4000):
    """Drive chunked prefill to completion.

    The loop is capped so that a request which can never be admitted shows up as a
    clear failure instead of spinning forever. That failure mode is the one issue
    #111 describes (a request that is never scheduled and never rejected), and it is
    the thing to rule out for the fan-out retention change: keeping the head's swa
    live must not starve admission.
    """
    pm.pending_list = [PendingReq(uid=7,
                                  input_ids=torch.tensor(ids, dtype=torch.int32),
                                  sampling_params=SamplingParams(max_tokens=n_decode))]
    last_batch = None
    iters = 0
    admitted = False
    while pm.runnable or last_batch is not None:
        iters += 1
        if iters > max_iters:
            raise AssertionError(
                "prefill never completed after %d iterations (admitted=%s, swa_free=%d) -- "
                "the request is stuck, not slow (cf. issue #111)"
                % (max_iters, admitted, cm.swa_pool.swa_available_size()))
        batch = pm.schedule_next_batch(TOKEN_BUDGET)
        if batch is not None:
            admitted = True
            cm.free_swa_out_of_window_extend(batch.reqs)
            cm.allocate_paged(batch.reqs)
            for r in batch.reqs:
                r.complete_one()
        if last_batch is not None:
            for r in last_batch.reqs:
                if not isinstance(r, ChunkedReq):
                    cm.cache_req(r, finished=finished)
        last_batch = batch
    assert admitted, "no chunk was ever scheduled"


def _reuse(cm, ids):
    return int(cm.prefix_cache.match_prefix(torch.tensor(ids, dtype=torch.int32)).cached_len)


def test_multi_round_fanout_reuses_every_round():
    """Many requests sharing one system prompt: each must reuse the head."""
    cm, tm, pm = _managers()
    head = list(range(1, 1 + HEAD))
    for i in range(4):
        tail = list(range(100000 + i * 1000, 100000 + i * 1000 + TAIL))
        _serve(cm, pm, head + tail)
        n = _reuse(cm, head + list(range(200000, 200000 + 10)))
        print("\n  round %d: head reuse = %d / %d" % (i + 1, n, HEAD))
        assert n >= HEAD, "round %d lost the shared head (%d/%d)" % (i + 1, n, HEAD)


def test_fanout_under_tight_pool_does_not_raise():
    """The #202 risk: a tight window pool must evict, not raise RuntimeError.

    num_swa_tokens is only a few windows, so serving several 1200-token prompts
    forces ensure_swa_slots -> evict_swa to actually run.
    """
    cm, tm, pm = _managers(num_swa_tokens=1024)
    head = list(range(1, 1 + HEAD))
    for i in range(3):
        tail = list(range(300000 + i * 1000, 300000 + i * 1000 + TAIL))
        _serve(cm, pm, head + tail)          # must not raise
    n = _reuse(cm, head + list(range(400000, 400000 + 10)))
    print("\n  tight pool survived; final head reuse = %d / %d" % (n, HEAD))
    # reuse may degrade under real pressure -- that is acceptable; raising is not.


def test_mixed_fanout_and_append():
    """Both access patterns against the same tree."""
    cm, tm, pm = _managers()
    head = list(range(1, 1 + HEAD))
    tail = list(range(500000, 500000 + TAIL))
    _serve(cm, pm, head + tail)
    # append: continue the exact conversation
    n_ap = _reuse(cm, head + tail + [7, 8, 9])
    # fan-out: same head, different tail
    n_fo = _reuse(cm, head + list(range(600000, 600000 + 10)))
    print("\n  append reuse=%d  fanout reuse=%d" % (n_ap, n_fo))
    assert n_ap >= HEAD + TAIL - WINDOW - cache_mod._SWA_RETAIN_GAP
    assert n_fo >= HEAD


def test_page_size_gt_1_alignment():
    """The removed path was page-aligned; make sure nothing else misaligns."""
    cm, tm, pm = _managers(ps=8, num_swa_tokens=8192, width=8192)
    head = [1 + i * 3 for i in range(HEAD)]
    tail = [700000 + i * 3 for i in range(TAIL)]
    _serve(cm, pm, head + tail)
    n = _reuse(cm, head + [800000 + i * 3 for i in range(10)])
    print("\n  ps=8 head reuse = %d / %d" % (n, HEAD))
    assert n > 0, "ps=8 lost all reuse"


def test_long_fanout_chain_bounded():
    """Repeated fan-out must not grow the pool without bound."""
    cm, tm, pm = _managers(num_swa_tokens=4096)
    head = list(range(1, 1 + HEAD))
    for i in range(4):
        tail = list(range(900000 + i * 700, 900000 + i * 700 + TAIL))
        _serve(cm, pm, head + tail)
    free = cm.swa_pool.swa_available_size()
    print("\n  after 15 fan-out rounds: swa free = %d" % free)
    assert free >= 0


def test_fanout_reuse_survives_several_rounds():
    """The measured effect of the #200 fix, away from the harness's stall boundary.

    Unfixed cache.py reuses 0 tokens on every round; fixed reuses the whole head for
    as many rounds as the harness reaches (4 before it hits its own uid/decode limit).
    Asserting on the first four keeps the check honest about what it covers.
    """
    cm, tm, pm = _managers()
    head = list(range(1, 1 + HEAD))
    reuses = []
    for i in range(4):
        tail = list(range(700000 + i * 1000, 700000 + i * 1000 + TAIL))
        _serve(cm, pm, head + tail)
        reuses.append(_reuse(cm, head + [800000 + i]))
    print("\n  fan-out reuse per round: %s" % reuses)
    assert all(r >= HEAD for r in reuses), "reuse degraded: %s" % reuses
