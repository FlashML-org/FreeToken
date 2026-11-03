from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch

from freetoken.core import ReqStats

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_items: list | None = None
    mrope_positions_full: torch.Tensor | None = None
    mrope_delta: int = 0
    # timing/draft counters; add_one_req stamps enqueued_at on it, and every chunk
    # Req built from this pending request shares the same instance
    stats: ReqStats = field(default_factory=ReqStats)

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
