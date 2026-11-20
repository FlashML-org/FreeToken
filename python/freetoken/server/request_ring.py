"""In-memory ring of recent API requests for the desktop Logs tab.

A bounded deque + a monotonic all-time cursor: clients pull incrementally with
``?since=<next_cursor>``. Records are appended by an HTTP middleware; p95 (for /v1/stats)
reads the same ring. Purely in-process — request_logger.py still owns the on-disk JSONL."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass


@dataclass
class RequestRecord:
    ts: str
    method: str
    path: str
    status: int
    model: str | None
    duration_ms: int
    ttft_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    stream: bool | None
    error: str | None
    # prompt_tokens served from the prefix cache (whatever the scheduler reported on the
    # terminal ack). None where the path cannot know it (the HTTP middleware, pre-totals).
    cached_tokens: int | None = None


class RequestRing:
    def __init__(self, capacity: int = 512) -> None:
        self._buf: "deque[tuple[int, RequestRecord]]" = deque(maxlen=capacity)
        self._next = 0  # all-time monotonic cursor (survives eviction)

    def add(self, rec: RequestRecord) -> None:
        self._buf.append((self._next, rec))
        self._next += 1

    def since(self, cursor: int, limit: int) -> tuple[list[dict], int]:
        """Return records with id >= cursor (up to limit) and the next cursor to poll with.
        If limit truncates the result set, next_cursor is the cursor of the last returned record.
        Only when all matched records are returned may next_cursor be self._next (the all-time count)."""
        matched = [(idx, rec) for idx, rec in self._buf if idx >= cursor]
        out = [asdict(rec) for idx, rec in matched[:limit]]

        # If we got fewer results than matched, we were truncated by limit
        if len(out) < len(matched):
            # next_cursor is the cursor of the last returned record + 1
            last_returned_idx = matched[len(out) - 1][0]
            next_cursor = last_returned_idx + 1
        else:
            # All matched records were returned
            next_cursor = self._next

        return out, next_cursor

    def p95_ms(self) -> int:
        durs = sorted(rec.duration_ms for _idx, rec in self._buf)
        if not durs:
            return 0
        k = max(0, math.ceil(0.95 * len(durs)) - 1)
        return int(durs[k])

    def ttft_mean_ms(self) -> int:
        """Mean TTFT over the records that have one."""
        vals = [rec.ttft_ms for _idx, rec in self._buf if rec.ttft_ms is not None]
        if not vals:
            return 0
        return int(round(sum(vals) / len(vals)))

    def count(self) -> int:
        return self._next


# ------------------------------------------------------------------ module singleton
_RING = RequestRing()


def record_request(rec: RequestRecord) -> None:
    _RING.add(rec)


def requests_since(cursor: int, limit: int) -> tuple[list[dict], int]:
    return _RING.since(cursor, limit)


def requests_p95_ms() -> int:
    return _RING.p95_ms()


def requests_ttft_mean_ms() -> int:
    return _RING.ttft_mean_ms()


def requests_count() -> int:
    return _RING.count()


def reset() -> None:
    """Test helper: clear the singleton ring."""
    global _RING
    _RING = RequestRing()
