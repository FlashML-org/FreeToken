"""rollback_tail after a widened decode step: the request keeps exactly the pages of its committed tokens, the rest go back, and the next step allocates no page twice. On an swa_paged manager the swa slots of the dropped range go back too (a leak of ~k+1 per verify step otherwise)."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
from freetoken.models.config import KVCacheGroupSpec
from freetoken.scheduler.cache import CacheManager
from freetoken.utils import div_ceil

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)

K = 3
NUM_PAGES = 64


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cached", [8, 9, 11])
@pytest.mark.parametrize("n", range(1, K + 2))
def test_rollback_keeps_committed_pages_only(page_size, cached, n):
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, page_size, page_table, "naive")
    req = Req(input_ids=torch.zeros(cached, dtype=torch.int32), table_idx=0, cached_len=0, output_len=32,
              uid=0, sampling_params=SimpleNamespace(), cache_handle=None)
    cm.allocate_paged([req])
    req.complete_one()  # cached_len = cached, the pending token sits at position `cached`

    for _ in range(2):
        entry = req.cached_len
        req.widen_pending(K)
        cm.allocate_paged([req])
        req.commit_pending(n)
        with cm.lazy_free_region():
            cm.rollback_tail(req, req.cached_len, entry + K + 1)
        kept = div_ceil(req.cached_len, page_size)
        assert len(cm.free_slots) == NUM_PAGES - kept
        assert not set(cm.free_slots.tolist()) & set(cm.page_table[0, : kept * page_size : page_size].tolist())


ROUNDS = 4


@pytest.mark.parametrize("n", [1, 2, 3])
def test_rollback_tail_returns_swa_slots(n):
    """swa_paged verify steps: allocate_paged charges one swa slot per verify row, so a rollback
    that only frees full slots leaks the dropped rows' swa slots until alloc_swa raises
    "SWA pool exhausted". After the fix each round returns exactly the rolled-back range."""
    ps = 1  # SWA models are forced page_size == 1 (engine.py:1970-1973), so the range is exact-token
    cached = 9
    # Tight on purpose: pre-fix every round consumes K+1 swa slots, so the pool would raise by
    # the last round; post-fix only the n committed tokens per round stay held.
    num_swa_tokens = cached + ROUNDS * (K + 1)
    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                         sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                         sliding_window=4),
    )
    pool = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=NUM_PAGES, page_size=ps,
                            dtype=torch.bfloat16, device=torch.device("cpu"),
                            num_swa_tokens=num_swa_tokens)
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, ps, page_table, "naive", swa_pool=pool, sliding_window_size=4)
    assert cm.swa_paged
    req = Req(input_ids=torch.zeros(cached, dtype=torch.int32), table_idx=0, cached_len=0,
              output_len=32, uid=0, sampling_params=SimpleNamespace(), cache_handle=None)
    cm.allocate_paged([req])
    req.complete_one()

    for _ in range(ROUNDS):
        entry = req.cached_len
        req.widen_pending(K)
        cm.allocate_paged([req])
        req.commit_pending(n)
        start, end = req.cached_len, entry + K + 1
        before = pool.swa_available_size()
        with cm.lazy_free_region():
            cm.rollback_tail(req, req.cached_len, entry + K + 1)
        assert pool.swa_available_size() == before + (end - start)
        rolled = page_table[0, start:end].to(torch.int64)
        assert torch.all(pool.full_to_swa_index_mapping[rolled] == 0)
        assert torch.all(pool.full_to_swa_index_mapping[page_table[0, :start].to(torch.int64)] > 0)

    assert pool.swa_available_size() == num_swa_tokens - 1 - (cached + ROUNDS * n)

    # A later sweep over the stale range no-ops on the 0 sentinel. rollback_tail is not re-run over
    # it here: _free would push the already-freed full slots twice, and that combination never runs.
    before = pool.swa_available_size()
    cm._free_swa(page_table[0, start:end])
    assert pool.swa_available_size() == before
    assert len(cm.free_slots) == NUM_PAGES - div_ceil(req.cached_len, ps)
