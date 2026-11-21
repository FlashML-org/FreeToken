"""Issue #200: fan-out over a shared prefix must reuse it, not re-prefill it.

The finish-time soft pin in `_cache_req_swa` keeps windowed (SWA) KV only for
    [prompt_len - sliding_window - _SWA_RETAIN_GAP, prompt_len)
so a later request that matches to position P reuses the shared prefix only when
P >= keep_from. That covers a follow-up turn (which diverges at the prompt end)
but not FAN-OUT: many requests sharing a system prompt, each with its own user
message, diverge a whole message before the previous prompt's end.

This drives the real CacheManager on CPU (no model, no GPU): prefill one request
with a long tail, finish it, then match a second request that shares only the
head. The shared head must still have live windowed KV.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Context, Req, SamplingParams, get_global_ctx, set_global_ctx
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
UID = 7
TOKEN_BUDGET = 65536
WINDOW = 128

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


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


def _managers(window=WINDOW, num_swa_tokens=8192, ps=1, num_pages=4096, width=4096):
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


def _prefill_and_finish(cm, pm, ids, n_decode=4):
    """Real chunked prefill, then commit as finished (the path that trims the head)."""
    pm.pending_list = [PendingReq(uid=UID,
                                  input_ids=torch.tensor(ids, dtype=torch.int32),
                                  sampling_params=SamplingParams(max_tokens=n_decode))]
    last_batch = None
    while pm.runnable or last_batch is not None:
        batch = pm.schedule_next_batch(TOKEN_BUDGET)
        if batch is not None:
            cm.free_swa_out_of_window_extend(batch.reqs)
            cm.allocate_paged(batch.reqs)
            for r in batch.reqs:
                r.complete_one()
        if last_batch is not None:
            for r in last_batch.reqs:
                if not isinstance(r, ChunkedReq):
                    cm.cache_req(r, finished=True)
        last_batch = batch
    return last_batch


def _match_len(cm, ids):
    """How many tokens of `ids` the prefix cache can actually reuse."""
    m = cm.prefix_cache.match_prefix(torch.tensor(ids, dtype=torch.int32))
    return int(m.cached_len)


SHARED = 600                       # the shared system prompt
TAIL = 600                         # first request's own message (>> window+gap)


def test_fanout_reuses_the_shared_prefix():
    """Two requests share a 600-token head, each with its own 600-token tail.

    The second must reuse the shared head. Today it does not: the first request's
    finish-time trim frees the head's windowed KV because the divergence point
    (600) sits below keep_from (1200 - 128 - 16 = 1056).
    """
    cm, tm, pm = _managers()
    head = list(range(1, 1 + SHARED))
    tail_a = list(range(100000, 100000 + TAIL))
    tail_b = list(range(200000, 200000 + TAIL))

    _prefill_and_finish(cm, pm, head + tail_a)

    reuse = _match_len(cm, head + tail_b)
    keep_from = (SHARED + TAIL) - WINDOW - cache_mod._SWA_RETAIN_GAP
    print("\n  keep_from=%d  divergence=%d  reuse=%d / %d"
          % (keep_from, SHARED, reuse, SHARED))
    assert reuse >= SHARED, (
        "fan-out lost the shared prefix: reused %d of %d tokens "
        "(keep_from=%d, divergence at %d)" % (reuse, SHARED, keep_from, SHARED))


def test_append_still_reuses_the_prefix():
    """The case the existing soft pin was designed for must keep working.

    A follow-up turn diverges at the prompt end, i.e. inside the retained window.
    """
    cm, tm, pm = _managers()
    head = list(range(1, 1 + SHARED))
    tail = list(range(100000, 100000 + TAIL))

    _prefill_and_finish(cm, pm, head + tail)
    reuse = _match_len(cm, head + tail + [999, 998, 997])
    print("\n  append reuse=%d / %d" % (reuse, SHARED + TAIL))
    assert reuse >= SHARED + TAIL - WINDOW - cache_mod._SWA_RETAIN_GAP, (
        "append path regressed: reused only %d" % reuse)
