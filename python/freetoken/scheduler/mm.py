"""Scheduler-side multimodal planning: the embedding rows a chunk gathers and the items it encodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from freetoken.utils import align_down

if TYPE_CHECKING:
    from freetoken.message import MMItem
    from freetoken.mm.encoder_cache import EncoderCache


def mm_rows_after(item: MMItem, cached_len: int) -> int:
    """Embedding rows of the item that prefill will gather once the first cached_len tokens are skipped."""
    return sum(max(0, span_hi - max(span_lo, cached_len)) for span_lo, span_hi in item.offsets)


def mm_chunk_end(items: List[MMItem], chunk_lo: int, end: int, align: int) -> int:
    """The chunk end pulled back to an align multiple before any image span it would cut; end itself when that span already begins at or before chunk_lo (the image is longer than the budget)."""
    cut = end
    for span_lo, span_hi in sorted((lo, hi) for item in items for lo, hi in item.offsets)[::-1]:
        if span_lo < cut < span_hi:
            if span_lo <= chunk_lo:
                return end
            cut = align_down(span_lo, align)
            if cut <= chunk_lo:
                return end
    return cut


def cut_image_spans(reqs) -> List[Tuple[int, int]]:
    """The image spans the reqs' current chunks end inside of; mm_chunk_end leaves such a cut only for an image longer than the chunk."""
    return [(lo, hi) for req in reqs if req.mm_items for item in req.mm_items for lo, hi in item.offsets if lo < req.device_len < hi]


def plan_mm_chunk(
    uid: int,
    items: List[MMItem],
    window_lo: int,
    window_hi: int,
    encoder_cache: EncoderCache | None,
) -> tuple[List[MMItem], List[Tuple[int, int, int, int, int, int]]]:
    """Encoder jobs and the embedding-row gather plan for the items overlapping the chunk window [window_lo, window_hi).

    Plan rows are (uid, hash, row_lo, row_hi, num_tokens, pos): embedding rows [row_lo, row_hi) land at chunk position pos. A repeated image is one job.
    """
    jobs: List[MMItem] = []
    queued: set[int] = set()
    plan: List[Tuple[int, int, int, int, int, int]] = []
    for item in items:
        num_tokens = item.num_tokens
        needs_encode = item.hash not in queued and (encoder_cache is None or not encoder_cache.has(item.hash))
        row_base = 0
        for span_lo, span_hi in item.offsets:
            lo, hi = max(span_lo, window_lo), min(span_hi, window_hi)
            if lo < hi:
                if needs_encode:
                    jobs.append(item)
                    queued.add(item.hash)
                    needs_encode = False
                plan.append((uid, item.hash, row_base + lo - span_lo, row_base + hi - span_lo, num_tokens, lo - window_lo))
            row_base += span_hi - span_lo
    return jobs, plan


def plan_mm_batch(reqs, encoder_cache: EncoderCache | None, *, starts: List[int] | None = None) -> tuple[List[MMItem], List[Tuple[int, int, int, int, int, int]], List[int], List[int]]:
    """Jobs, gather plan, embedding rows and image-span ends over each [start, device_len); starts default to cached_len."""
    jobs: List[MMItem] = []
    plan: List[Tuple[int, int, int, int, int, int]] = []
    rows: List[int] = []
    block_ends: List[int] = []
    offset = 0
    starts = starts if starts is not None else [r.cached_len for r in reqs]
    assert len(starts) == len(reqs)
    for req, start in zip(reqs, starts):
        if req.mm_items:
            req_jobs, req_plan = plan_mm_chunk(req.uid, req.mm_items, start, req.device_len, encoder_cache)
            jobs.extend(req_jobs)
            plan.extend(req_plan)
            for _, _, row_lo, row_hi, _, pos in req_plan:
                rows.extend(range(offset + pos, offset + pos + row_hi - row_lo))
            if not block_ends:
                block_ends = [0] * sum(r.device_len - s for r, s in zip(reqs, starts))
            for item in req.mm_items:
                for span_lo, span_hi in item.offsets:
                    for i in range(max(span_lo, start), min(span_hi, req.device_len)):
                        block_ends[offset + i - start] = span_hi
        offset += req.device_len - start
    return jobs, plan, rows, block_ends


__all__ = ["cut_image_spans", "mm_chunk_end", "mm_rows_after", "plan_mm_batch", "plan_mm_chunk"]
