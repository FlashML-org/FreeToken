"""EncoderCache ownership: rows claimed at admission, freed when the last claim is gathered."""

from __future__ import annotations

import torch

from freetoken.mm.encoder_cache import EncoderCache

CPU = torch.device("cpu")


def test_entry_lives_until_every_claim_is_gathered():
    cache = EncoderCache(storage="cpu")
    emb = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    cache.register(7, 1, 6)
    cache.register(7, 2, 6)
    assert not cache.has(7)  # claimed, not encoded yet
    cache.put(7, emb)
    assert cache.has(7)
    assert torch.equal(cache.get_slice(7, 2, 4, CPU), emb[2:4])
    cache.consume(7, 1, 4)
    cache.consume(7, 1, 2)
    assert cache.has(7)  # request 2 still has rows to gather
    cache.consume(7, 2, 6)
    assert not cache.has(7) and cache.stats() == (0, 0)


def test_repeated_image_in_one_request_adds_up():
    cache = EncoderCache(storage="cpu")
    cache.register(7, 1, 4)
    cache.register(7, 1, 4)
    cache.put(7, torch.ones(4, 2))
    cache.consume(7, 1, 4)  # the first occurrence is done
    assert cache.has(7)
    cache.consume(7, 1, 4)
    assert not cache.has(7)


def test_release_drops_a_claim_even_before_the_tensor_arrives():
    cache = EncoderCache(storage="cpu")
    cache.register(5, 1, 3)
    cache.register(5, 2, 3)
    cache.release(1, [5, 99])  # unknown hash: no-op
    assert 5 in cache._entries
    cache.release(2, [5])
    assert cache._entries == {} and cache.stats() == (0, 0)


def test_replay_keeps_consumed_images_and_claims_prefix_hits_until_release():
    cache = EncoderCache(storage="cpu", retain_until_prefill_end=True)
    cache.register(7, 1, 4)
    cache.register(7, 2, 0)  # prefix hit: all unique rows cached, but a replay still reads them
    emb = torch.arange(8).reshape(4, 2)
    cache.put(7, emb)
    cache.consume(7, 1, 4)
    cache.consume(7, 1, 2)
    cache.release(1, [7])
    assert torch.equal(cache.get_slice(7, 1, 3, CPU), emb[1:3])
    cache.release(2, [7])
    assert cache.stats() == (0, 0)


def test_scheduler_releases_images_only_after_final_prefill_chunk():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.scheduler.scheduler import Scheduler

    cache = EncoderCache(retain_until_prefill_end=True)
    cache.register(7, 1, 184)
    cache.put(7, torch.ones(184, 2))
    scheduler = SimpleNamespace(
        engine=SimpleNamespace(encoder_cache=cache),
        cache_manager=SimpleNamespace(lazy_free_region=nullcontext, cache_req=lambda *a, **kw: None),
        finished_reqs=set(), eos_token_ids=set(), toolcall_anchor_id=None,
        decode_manager=SimpleNamespace(running_reqs=set()),
        prefill_manager=SimpleNamespace(pending_list=[]), config=SimpleNamespace(page_size=128),
        status_reporter=SimpleNamespace(report_batch=lambda *a, **kw: None),
        _kv_usage_pages=lambda: (0, 64), _mamba_slot_usage=lambda: None, _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0, send_result=lambda _: None,
    )
    for cls, length in ((ChunkedReq, 128), (Req, 210)):
        # The scheduler passes a sliced prompt to intermediate chunks, not the whole prompt.
        req = cls(torch.arange(length), 0, 0, 4, 1, SamplingParams(), None)
        req.mm_items = [SimpleNamespace(hash=7)]
        assert req.device_len == len(req.input_ids)
        req.complete_one()
        data = (SimpleNamespace(batch=Batch([req], 'prefill')),
                (None, torch.tensor([3]), SimpleNamespace(synchronize=lambda: None)))
        Scheduler._process_last_data(scheduler, data)
        assert cache.has(7) is (cls is ChunkedReq)
    assert cache.stats() == (0, 0)
