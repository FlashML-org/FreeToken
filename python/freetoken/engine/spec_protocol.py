"""The speculative-decoding runtime as the engine and scheduler see it; they import these types only, never ``freetoken.spec``."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Protocol

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.models.blocks import FeatureSink

SPEC_RUNNER = "freetoken.spec.runner:SpecRunner"


class SpecForwardOutput(NamedTuple):
    """Drop-in for ForwardOutput at the drain: row i committed ``next_tokens[i, : emitted[i] + 1]``."""

    next_tokens_gpu: torch.Tensor        # [bs, k+1]
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    emitted_cpu: torch.Tensor            # [bs] accepted drafts, 0..k


class SpecRuntime(Protocol):
    radix_lookahead: bool

    @classmethod
    def adjust_config(cls, config, override) -> None: ...

    @classmethod
    def budget(cls, config) -> tuple[int, int]: ...

    def attach(self, engine) -> None: ...

    def detach(self, *, keep_kv: bool) -> None: ...

    def capture_graphs(self, bs_list: List[int], pool) -> None: ...

    def feature_sink(self, batch: Batch) -> FeatureSink | None: ...

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None: ...

    def propose_into(self, reqs: List[Req], token_pool: torch.Tensor) -> None: ...

    def verify_step(self, batch: Batch, sample_args: BatchSamplingArgs) -> SpecForwardOutput: ...

    def release(self, req: Req) -> None: ...
