"""Chunk-window planning over image spans: rows, jobs, cache hits, multi-span items, a repeated image across a cut."""

from __future__ import annotations

import torch

from freetoken.message import MMItem
from freetoken.mm.encoder_cache import EncoderCache
from freetoken.scheduler.mm import cut_image_spans, mm_chunk_end, mm_rows_after, plan_mm_batch, plan_mm_chunk

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
    jobs, plan, rows, block_ends = plan_mm_batch([a, b], None)
    assert [j.hash for j in jobs] == [7, 8]
    assert plan == [(1, 7, 0, 3, 3, 2), (2, 8, 1, 5, 5, 0)]
    assert rows == [2, 3, 4, 6, 7, 8, 9]  # req 2 starts at batch row 6; its image rows 1..5 land on its first four tokens
    assert block_ends == [0, 0, 5, 5, 5, 0, 8, 8, 8, 8, 0, 0]  # every image row carries its span's end in request positions


def test_chunk_end_never_lands_inside_an_image_span():
    items = [_item(1, [[10, 20]]), _item(2, [[22, 40]])]
    assert mm_chunk_end(items, 0, 8, 1) == 8  # before the images: untouched
    assert mm_chunk_end(items, 0, 15, 1) == 10  # inside the first image: back to its start
    assert mm_chunk_end(items, 0, 30, 1) == 22  # inside the second
    assert mm_chunk_end(items, 0, 30, 8) == 8  # the aligned start 16 lands in the first image, so back again
    assert mm_chunk_end(items, 0, 20, 1) == 20 and mm_chunk_end(items, 0, 40, 1) == 40  # ends on a boundary are fine
    assert mm_chunk_end(items, 22, 30, 1) == 30  # the image begins at the chunk start: longer than the budget, split it
    assert mm_chunk_end(items, 12, 15, 1) == 15  # an image already split by an earlier chunk keeps going


def test_cut_image_spans_reports_the_image_a_chunk_ends_inside():
    from types import SimpleNamespace

    image = MMItem(modality="image", hash=1, pad_value=1, offsets=[[10, 20], [22, 40]], feature=torch.zeros(1))
    req = lambda device_len: SimpleNamespace(mm_items=[image], device_len=device_len)
    text = SimpleNamespace(mm_items=None, device_len=5)
    assert cut_image_spans([text, req(30)]) == [(22, 40)]
    assert cut_image_spans([req(8)]) == cut_image_spans([req(20)]) == cut_image_spans([req(40)]) == []  # outside every span or on a boundary
