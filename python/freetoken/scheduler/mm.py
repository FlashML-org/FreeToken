"""Scheduler-side multimodal planning: the embedding rows a chunk gathers and the items it encodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from freetoken.message import MMItem
    from freetoken.mm.encoder_cache import EncoderCache


def mm_rows_after(item: MMItem, cached_len: int) -> int:
    """Embedding rows of the item that prefill will gather once the first cached_len tokens are skipped."""
    return sum(max(0, span_hi - max(span_lo, cached_len)) for span_lo, span_hi in item.offsets)


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


def plan_mm_batch(reqs, encoder_cache: EncoderCache | None) -> tuple[List[MMItem], List[Tuple[int, int, int, int, int, int]], List[int]]:
    """Jobs, plan and the batch rows of every gathered embedding row, over the reqs in batch order (each spans [cached_len, device_len))."""
    jobs: List[MMItem] = []
    plan: List[Tuple[int, int, int, int, int, int]] = []
    rows: List[int] = []
    offset = 0
    for req in reqs:
        if req.mm_items:
            req_jobs, req_plan = plan_mm_chunk(req.uid, req.mm_items, req.cached_len, req.device_len, encoder_cache)
            jobs.extend(req_jobs)
            plan.extend(req_plan)
            for _, _, row_lo, row_hi, _, pos in req_plan:
                rows.extend(range(offset + pos, offset + pos + row_hi - row_lo))
        offset += req.extend_len
    return jobs, plan, rows


__all__ = ["mm_rows_after", "plan_mm_batch", "plan_mm_chunk"]
