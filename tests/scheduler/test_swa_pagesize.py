"""SWA second-currency conservation at page_size > 1 (CPU, no model).

allocate_paged charges one swa slot per token of every WHOLE allocated page (alloc_swa over the
_page_to_token expansion), so every finish-path free must reach the page-CEIL bound -- freeing
only [..., cached_len) strands (-cached_len mod page_size) slots per request, permanently
(the next alloc_swa overwrites the full->swa mapping and the slot is unreachable). Sweeps a
full residue class of committed lengths so the boundary cases cannot be missed by luck.
Also: the hybrid GDN chunk-donate must skip a page-unaligned x64 boundary (insert would align
the key down and attach the state to a shorter node).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.config import KVCacheGroupSpec
from freetoken.scheduler.cache import CacheManager

DEVICE = torch.device("cpu")

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


def _swa_pool(ps: int, num_full_pages: int, num_swa_tokens: int):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                         sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                         sliding_window=ps),
    )
    return HybridSWAKVCache(
        groups=groups, num_layers=2, num_full_pages=num_full_pages, page_size=ps,
        dtype=torch.bfloat16, device=DEVICE, num_swa_tokens=num_swa_tokens,
    )


def _mgr(ps: int, cache_type: str = "swa_radix", num_pages: int = 24):
    pt = torch.zeros(5, num_pages * ps + 64, dtype=torch.int32)
    pool = _swa_pool(ps, num_full_pages=num_pages, num_swa_tokens=num_pages * ps)
    cm = CacheManager(num_pages=num_pages, page_size=ps, page_table=pt, type=cache_type,
                      swa_pool=pool, sliding_window_size=ps)
    assert cm.swa_paged
    return cm


def _req(prompt_len: int, n_decode: int) -> Req:
    r = Req(input_ids=torch.arange(1, prompt_len + 1, dtype=torch.int32), table_idx=0,
            cached_len=0, output_len=n_decode, uid=0,
            sampling_params=SamplingParams(), cache_handle=None)
    r.input_len = prompt_len
    return r


def _run_lifecycle(cm: CacheManager, total_len: int, n_decode: int = 3,
                   abort_after: int | None = None) -> None:
    """Faithful scheduler order: match -> lock -> [extend-free, allocate, complete_one]* ->
    cache_req(finished=True). ``abort_after`` finishes after that many decode steps instead."""
    req = _req(total_len - n_decode, n_decode)
    h = cm.match_req(req).cuda_handle
    req.cache_handle = h
    req.cached_len = h.cached_len
    cm.lock(h)

    cm.free_swa_out_of_window_extend([req])
    cm.allocate_paged([req])
    req.complete_one()
    steps = n_decode if abort_after is None else abort_after
    for i in range(steps):
        req.append_host(torch.tensor([9999], dtype=torch.int32))
        req.decode_batch_idx = i + 1
        cm.maybe_free_swa_out_of_window([req], forward_iter=i)
        cm.allocate_paged([req])
        req.complete_one()
    cm.cache_req(req, finished=True)
    cm.check_integrity()  # exact swa-slot + full-page conservation at idle


@pytest.mark.parametrize("ps", [1, 2, 8, 128])
@pytest.mark.parametrize("cache_type", ["swa_radix", "naive"])
def test_finish_conserves_swa_slots_across_a_full_residue_class(ps, cache_type):
    base = 3 * ps + 4
    for r in range(ps):
        cm = _mgr(ps, cache_type)
        _run_lifecycle(cm, total_len=base + r)


@pytest.mark.parametrize("ps", [2, 8, 128])
def test_abort_mid_decode_conserves_swa_slots(ps):
    for r in (0, 1, ps - 1):
        cm = _mgr(ps)
        _run_lifecycle(cm, total_len=3 * ps + 4 + r, n_decode=6, abort_after=2)


@pytest.mark.parametrize("ps", [8, 128])
def test_chunked_prefill_conserves_swa_slots(ps):
    # Multi-extend prompt (successive allocate_paged with a growing frontier), unaligned end.
    cm = _mgr(ps, num_pages=32)
    total = 6 * ps + 3
    req = _req(total, 1)
    h = cm.match_req(req).cuda_handle
    req.cache_handle = h
    req.cached_len = h.cached_len
    cm.lock(h)
    chunk = 2 * ps + 1
    while req.cached_len < total:
        req.device_len = min(req.cached_len + chunk, total)
        cm.free_swa_out_of_window_extend([req])
        cm.allocate_paged([req])
        req.cached_len = req.device_len
        req.device_len += 1
    cm.cache_req(req, finished=True)
    cm.check_integrity()


@pytest.mark.parametrize("ps", [8])
def test_hybrid_chunk_donate_skips_unaligned_boundary(ps):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    pool = LinearStatePool(group=g, num_slots=16, dtype=torch.bfloat16, device=DEVICE, tp_size=1)
    pt = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(16, ps, pt, "hybrid_radix", linear_state_pool=pool)

    req = _req(3 * ps + 3, 1)
    h = cm.match_req(SimpleNamespace(input_ids=req.input_ids, input_len=req.input_len,
                                     )).cuda_handle
    req.cache_handle = h
    req.cached_len = h.cached_len
    cm.lock(h)
    cm.allocate_paged([req])
    req.complete_one()
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = (pool.alloc(1)[0], pool.alloc(1)[0])
    req.mamba_next_track_idx = 0

    # Unaligned x64 boundary: the donate must be SKIPPED (state would attach to a shorter node).
    req.mamba_last_track_seqlen = 2 * ps + 3
    pp_before = req.mamba_ping_pong
    cm.cache_req(req, finished=False)
    assert req.mamba_last_track_seqlen is None
    assert req.mamba_ping_pong == pp_before          # frozen slot NOT replaced -> no donate
    assert req.cache_handle is h                     # handle NOT re-pointed -> donate skipped

    # Aligned boundary on the same request: the donate goes through.
    req.mamba_last_track_seqlen = 2 * ps
    cm.cache_req(req, finished=False)
    assert req.cache_handle is not h                 # re-matched + locked on the committed node
    assert req.cache_handle.cached_len == 2 * ps


def test_finish_retains_prompt_window_under_pressure():
    """Faithful scheduler order incl. the boundary commit: after finish + ambient evict_swa
    pressure, a follow-up request cutting exactly at the prompt end (the reasoning-dropped
    next turn) still reuses [0, P)."""
    ps = 1
    prompt_len, n_decode = 6, 12
    cm = _mgr(ps, "swa_radix", num_pages=64)
    req = _req(prompt_len, n_decode)
    h = cm.match_req(req).cuda_handle
    req.cache_handle = h
    req.cached_len = h.cached_len
    cm.lock(h)
    cm.free_swa_out_of_window_extend([req])
    cm.allocate_paged([req])
    req.complete_one()
    cm.cache_req(req, finished=False)              # prefill->decode boundary (scheduler.py:325)
    for i in range(n_decode):
        req.append_host(torch.tensor([9999], dtype=torch.int32))
        req.decode_batch_idx = i + 1
        cm.maybe_free_swa_out_of_window([req], forward_iter=0)  # window slides every step
        cm.allocate_paged([req])
        req.complete_one()
    cm.cache_req(req, finished=True)
    cm.check_integrity()

    ev = cm.prefix_cache.evict_swa(1)              # ambient pressure right after finish
    cm.swa_pool.free_swa(ev.swa_indices)
    if ev.kv_indices.numel():
        cm._free(ev.kv_indices)
    cm.check_integrity()
    probe = _req(prompt_len + 1, 1)  # prompt + the next turn's first divergent token
    assert cm.match_req(probe).cuda_handle.cached_len == prompt_len
