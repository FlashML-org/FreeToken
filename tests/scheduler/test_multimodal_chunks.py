from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import MMItem, UserMsg
from freetoken.mm import mm_pad_value
from freetoken.mm.encoder_cache import EncoderCache
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.mm import plan_mm_batch
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq


def _manager(cache_type="radix", encoder_cache=None):
    table = TableManager(4, torch.zeros(4, 64, dtype=torch.int32))
    kwargs = {}
    page_size = 1
    if cache_type == "swa_radix":
        from freetoken.distributed import set_tp_info, try_get_tp_info
        from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
        from freetoken.models.config import KVCacheGroupSpec

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)
        page_size = 2
        groups = (
            KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                             sliding_window=None),
            KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                             sliding_window=4),
        )
        pool = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=32,
                                page_size=page_size, dtype=torch.bfloat16,
                                device=torch.device("cpu"), num_swa_tokens=64)
        kwargs = dict(swa_pool=pool, sliding_window_size=4)
    cache = CacheManager(64 // page_size, page_size, table.page_table, cache_type, **kwargs)
    manager = PrefillManager(cache, table, SimpleNamespace(inflight_tokens=0), encoder_cache=encoder_cache)
    return cache, manager


@pytest.mark.parametrize("cache_type", ["radix", "swa_radix"])
@pytest.mark.parametrize("payload_name", ["media", "mm_embeds"])
def test_media_survives_chunking_without_prefix_sharing(cache_type, payload_name):
    cache, manager = _manager(cache_type)
    ids = torch.tensor([4, 9, 9, 9, 9, 5, 6, 7, 8], dtype=torch.int32)
    media = [dict(start=1, types=torch.tensor([0, 1, 2, 3]), patches=torch.zeros(1, 3, 2, 2), n_vit_h=1, n_vit_w=1)]
    payload = media if payload_name == "media" else torch.arange(32).reshape(4, 8)
    manager.add_one_req(UserMsg(1, ids, SamplingParams(max_tokens=1), **{payload_name: payload}))
    original_free = len(cache.free_slots)
    original_swa = cache.swa_available_size if cache.swa_paged else None
    forwarded = 0
    finished = None
    while manager.runnable:
        batch = manager.schedule_next_batch(3)
        assert batch is not None
        [req] = batch.reqs
        assert getattr(req, payload_name) is payload
        assert req.cached_len == forwarded
        cache.free_swa_out_of_window_extend([req])
        cache.allocate_paged([req])
        forwarded += req.extend_len
        req.cached_len = req.device_len
        with cache.lazy_free_region():
            cache.cache_req(req, finished=False)
        assert cache.match_req(SimpleNamespace(input_ids=ids, input_len=len(ids), mm_embeds=None, media=media)).cuda_handle.cached_len == 0
        assert cache.match_req(SimpleNamespace(input_ids=ids, input_len=len(ids), mm_embeds=None)).cuda_handle.cached_len == 0
        if not isinstance(req, ChunkedReq):
            finished = req
    assert forwarded == len(ids)
    with cache.lazy_free_region():
        cache.cache_req(finished, finished=True)
    assert len(cache.free_slots) == original_free
    if cache.swa_paged:
        assert cache.swa_available_size == original_swa
    cache.check_integrity()


@pytest.mark.parametrize("cache_type", ["radix", "swa_radix"])
def test_hashed_image_chunks_preserve_encoder_claims_and_reuse_only_matching_content(cache_type):
    encoder = EncoderCache(storage="cpu")
    cache, manager = _manager(cache_type, encoder_cache=encoder)
    pad = mm_pad_value(7)
    ids = torch.tensor([4, pad, pad, pad, pad, 5, 6, 7], dtype=torch.int32)
    item = MMItem(modality="image", hash=7, pad_value=pad, offsets=[[1, 5]], feature=torch.zeros(1))
    items = [item]
    positions = torch.arange(24, dtype=torch.int32).reshape(3, 8)
    params = SamplingParams(max_tokens=1)
    manager.add_one_req(UserMsg(1, ids, params, mm_items=items, mrope_positions=positions, mrope_delta=5))
    embedding = torch.arange(32).reshape(4, 8)
    gathered = []
    encoded = 0
    while manager.runnable:
        batch = manager.schedule_next_batch(3)
        assert batch is not None
        [req] = batch.reqs
        assert req.mm_items is items and req.mrope_positions_full is positions
        assert req.mrope_delta == 5
        assert req.media is None and req.mm_embeds is None
        jobs, plan, rows = plan_mm_batch(batch.reqs, encoder)
        for job in jobs:
            encoded += 1
            encoder.put(job.hash, embedding)
        expected_rows = list(range(max(1, req.cached_len) - req.cached_len,
                                   min(5, req.device_len) - req.cached_len))
        assert rows == expected_rows
        for uid, h, lo, hi, _, _ in plan:
            gathered.append(encoder.get_slice(h, lo, hi, torch.device("cpu")))
            encoder.consume(h, uid, hi - lo)
        cache.free_swa_out_of_window_extend([req])
        cache.allocate_paged([req])
        req.cached_len = req.device_len
        if not isinstance(req, ChunkedReq):
            with cache.lazy_free_region():
                cache.cache_req(req, finished=True)

    assert encoded == 1 and torch.equal(torch.cat(gathered), embedding)
    assert encoder.stats() == (0, 0)
    same = PendingReq(2, ids, params, mm_items=items)
    matched_len = (len(ids) - 1) // cache.page_size * cache.page_size
    assert cache.match_req(same).cuda_handle.cached_len == matched_len
    changed = ids.clone()
    changed[1:5] = mm_pad_value(8)
    other = MMItem(modality="image", hash=8, pad_value=mm_pad_value(8), offsets=[[1, 5]], feature=torch.zeros(1))
    assert cache.match_req(PendingReq(3, changed, params, mm_items=[other])).cuda_handle.cached_len == 1 // cache.page_size
    assert cache.match_req(PendingReq(4, ids, params, media=[{}])).cuda_handle.cached_len == 0

    manager.add_one_req(UserMsg(2, ids, params, mm_items=items, mrope_positions=positions, mrope_delta=5))
    [reused] = manager.schedule_next_batch(3).reqs
    assert reused.cached_len == matched_len
    assert plan_mm_batch([reused], encoder) == ([], [], [])
    assert encoder.stats() == (0, 0)
    cache.allocate_paged([reused])
    reused.cached_len = reused.device_len
    with cache.lazy_free_region():
        cache.cache_req(reused, finished=True)
    cache.check_integrity()
