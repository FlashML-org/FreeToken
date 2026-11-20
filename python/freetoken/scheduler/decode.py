from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from freetoken.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    # slots a decode step may hold past each request's committed length (provisional rows)
    token_headroom: int = 0
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        # each request's partly filled page, plus the provisional rows a decode step may add
        tokens_reserved = (self.page_size - 1 + self.token_headroom) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self, hold: Iterable[Req] = ()) -> Batch | None:
        """The next decode step over every running request but those in ``hold``, which sit this one out."""
        reqs = sorted(self.running_reqs.difference(hold), key=lambda req: req.uid)
        return Batch(reqs=reqs, phase="decode") if reqs else None

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
