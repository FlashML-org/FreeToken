"""Frontend backpressure: opt-in outstanding-request capacity, not a token rate limit."""

from __future__ import annotations

import math
from typing import Any

from . import request_ring


class AdmissionThrottledError(RuntimeError):
    """No frontend slot is available; no uid, queue or accounting was allocated."""

    def __init__(self) -> None:
        super().__init__("server is busy: maximum concurrent requests reached")
        # Use the existing bounded ring's request p95 as a retry estimate. It is
        # deliberately only a hint, not a promise about when the next slot frees.
        p95_ms = request_ring.requests_p95_ms()
        self.retry_after = max(1, math.ceil(p95_ms / 1000)) if p95_ms > 0 else 1


def admission_limit(config: Any) -> int:
    """0 is unlimited; -1 explicitly opts in to the scheduler's running-request cap."""
    limit = getattr(config, "max_concurrent_requests", 0)
    if limit == -1:
        limit = getattr(config, "max_running_req", 0)
        if limit <= 0:
            raise ValueError("automatic admission limit requires a positive max_running_req")
    if limit < 0:
        raise ValueError("max_concurrent_requests must be -1, 0 or a positive integer")
    return limit
