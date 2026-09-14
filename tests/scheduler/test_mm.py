"""Chunk-window planning over image spans: rows, jobs, cache hits, multi-span items, a repeated image across a cut."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.message import MMItem
from freetoken.mm.encoder_cache import EncoderCache
from freetoken.scheduler.mm import gather_legacy_mm_batch, mm_rows_after, plan_mm_batch, plan_mm_chunk

CPU = torch.device("cpu")


def _item(h, offsets):
    return MMItem(modality="image", hash=h, pad_value=h, offsets=offsets, feature=torch.zeros(1))


class _Cache:
    def __init__(self, *hashes):
        self._h = set(hashes)

    def has(self, h):
        return h in self._h


def test_single_span_split_across_two_chunks():
    item = _item(7, [[10, 20]])
    jobs, plan = plan_mm_chunk(1, [item], 0, 16, None)
    assert jobs == [item] and plan == [(1, 7, 0, 6, 10, 10)]
    jobs, plan = plan_mm_chunk(1, [item], 16, 32, None)
    assert jobs == [item] and plan == [(1, 7, 6, 10, 10, 0)]  # row_hi == num_tokens: last rows


def test_cached_prefix_and_cache_hit_produce_no_job():
    item = _item(7, [[10, 20]])
    assert plan_mm_chunk(1, [item], 20, 40, None) == ([], [])  # span entirely in the cached prefix
    jobs, plan = plan_mm_chunk(1, [item], 0, 40, _Cache(7))
    assert jobs == [] and plan == [(1, 7, 0, 10, 10, 10)]  # gather only


def test_multi_span_item_maps_to_consecutive_rows():
    item = _item(9, [[2, 5], [8, 12]])  # 3 + 4 rows
    jobs, plan = plan_mm_chunk(3, [item], 0, 10, None)
    assert jobs == [item]  # one job per item even when two spans overlap the window
    assert plan == [(3, 9, 0, 3, 7, 2), (3, 9, 3, 5, 7, 8)]
    jobs, plan = plan_mm_chunk(3, [item], 10, 20, None)
    assert jobs == [item] and plan == [(3, 9, 5, 7, 7, 0)]


def test_rows_after_the_cached_prefix():
    item = _item(1, [[2, 5], [8, 12]])
    assert mm_rows_after(item, 0) == 7
    assert mm_rows_after(item, 4) == 1 + 4  # the hit ends inside the first span
    assert mm_rows_after(item, 12) == 0


def test_repeated_image_survives_a_chunk_cut_between_its_occurrences():
    cache = EncoderCache(storage="cpu")
    items = [_item(7, [[0, 4]]), _item(7, [[5, 9]])]
    for item in items:
        cache.register(item.hash, 1, mm_rows_after(item, 0))
    encoded = []

    def run_chunk(lo, hi):
        jobs, plan = plan_mm_chunk(1, items, lo, hi, cache)
        for item in jobs:
            if not cache.has(item.hash):
                assert item.feature is not None
                encoded.append(item.hash)
                cache.put(item.hash, torch.ones(4, 8))
            item.feature = None
        for uid, h, row_lo, row_hi, _, _ in plan:
            cache.get_slice(h, row_lo, row_hi, CPU)
            cache.consume(h, uid, row_hi - row_lo)
        return jobs

    assert len(run_chunk(0, 6)) == 1  # both occurrences overlap the window: one job
    assert cache.has(7)  # the second occurrence still has rows to gather
    assert run_chunk(6, 10) == []  # nothing to encode, both features already gone
    assert encoded == [7] and not cache.has(7)


def test_batch_rows_follow_the_reqs_in_batch_order():
    from types import SimpleNamespace

    # req 1: 6 tokens, image on [2, 5); req 2: chunk [4, 10) of a prompt whose image spans [3, 8)
    a = SimpleNamespace(uid=1, mm_items=[_item(7, [[2, 5]])], cached_len=0, device_len=6, extend_len=6)
    b = SimpleNamespace(uid=2, mm_items=[_item(8, [[3, 8]])], cached_len=4, device_len=10, extend_len=6)
    jobs, plan, rows = plan_mm_batch([a, b], None)
    assert [j.hash for j in jobs] == [7, 8]
    assert plan == [(1, 7, 0, 3, 3, 2), (2, 8, 1, 5, 5, 0)]
    assert rows == [2, 3, 4, 6, 7, 8, 9]  # req 2 starts at batch row 6; its image rows 1..5 land on its first four tokens


def test_legacy_soft_tokens_slice_at_chunk_boundaries():
    embeds = torch.arange(24).reshape(3, 8)
    prefix_req = SimpleNamespace(extend_len=3)
    req = SimpleNamespace(input_ids=torch.tensor([9, 1, 9, 9, 2]), mm_embeds=embeds)
    gathered = []
    for lo, hi, expected_rows in ((0, 2, [3]), (2, 4, [3, 4]), (4, 5, [])):
        req.cached_len, req.device_len, req.extend_len = lo, hi, hi - lo
        parts, rows = gather_legacy_mm_batch([prefix_req, req], image_token_id=9)
        assert rows == expected_rows
        gathered.extend(parts)
    assert torch.equal(torch.cat(gathered), embeds)


@pytest.mark.parametrize("image_token_id,embeds,message", [
    (None, torch.zeros(1, 8), "require an image_token_id"),
    (9, torch.zeros(8), "must have shape"),
    (9, torch.zeros(0, 8), "slots exceed"),
])
def test_legacy_soft_tokens_reject_incompatible_metadata(image_token_id, embeds, message):
    req = SimpleNamespace(input_ids=torch.tensor([9]), mm_embeds=embeds,
                          cached_len=0, device_len=1, extend_len=1)
    with pytest.raises(ValueError, match=message):
        gather_legacy_mm_batch([req], image_token_id)


def test_scheduler_appends_legacy_rows_after_canonical_rows():
    from freetoken.scheduler.scheduler import Scheduler

    embeds = torch.arange(8).reshape(1, 8)
    legacy = SimpleNamespace(uid=1, mm_items=None, input_ids=torch.tensor([9, 5]),
                             mm_embeds=embeds, cached_len=0, device_len=2, extend_len=2)
    canonical = SimpleNamespace(uid=2, mm_items=[_item(7, [[1, 3]])],
                                cached_len=0, device_len=3, extend_len=3)
    batch = SimpleNamespace(padded_reqs=[legacy, canonical], mm_embeds=None, mm_rows=None)
    scheduler = SimpleNamespace(engine=SimpleNamespace(encoder_cache=None), device=CPU,
                                config=SimpleNamespace(model_config=SimpleNamespace(image_token_id=9)))

    Scheduler._gather_multimodal(scheduler, batch)

    assert batch.mm_rows.tolist() == [3, 4, 0]
    assert torch.equal(batch.mm_embeds, embeds)
    assert batch.mm_encoder_jobs == canonical.mm_items
    assert batch.mm_gather_plan == [(2, 7, 0, 2, 2, 1)]
