"""The speculative-decoding runtime as the engine and scheduler see it; they import these types only, never ``freetoken.spec``."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Protocol

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.models.blocks import FeatureSink
    from freetoken.models.config import SlotStateSpec

SPEC_RUNNER = "freetoken.spec.runner:SpecRunner"


class SpecProposal(NamedTuple):
    tokens: torch.Tensor                 # [bs, k] int64 target-vocab ids
    cand: torch.Tensor | None = None     # [bs, k, K] target-vocab ids when drafts are sampled
    q: torch.Tensor | None = None        # [bs, k, K] draft probabilities over cand


class SpecForwardOutput(NamedTuple):
    """Drop-in for ForwardOutput at the drain: row i committed ``next_tokens[i, : emitted[i] + 1]``."""

    next_tokens_gpu: torch.Tensor        # [bs, k+1]
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    emitted_cpu: torch.Tensor            # [bs] accepted drafts, 0..k


class SpecRuntime(Protocol):
    rows_per_req: int                    # k + 1
    radix_lookahead: int

    def propose_into(self, reqs: List[Req], token_pool: torch.Tensor) -> None: ...

    def verify_step(self, batch: Batch, sample_args: BatchSamplingArgs, token_pool: torch.Tensor) -> SpecForwardOutput: ...

    def commit(self, rows: List[int], reqs: List[Req], entry_cached: List[int], counts: List[int],
               slots: List[int], token_pool: torch.Tensor) -> None: ...

    def release(self, req: Req) -> None: ...

    def prefill_sink(self, batch: Batch) -> FeatureSink | None: ...

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None: ...

    def verify_sink(self, rows: int) -> FeatureSink | None: ...

    def capture_graphs(self, bs_list: List[int], pool) -> None: ...

    def detach(self, *, keep_kv: bool) -> None: ...

    @classmethod
    def default_k(cls, config) -> int: ...

    @classmethod
    def validate(cls, config) -> None: ...

    @classmethod
    def budget(cls, config) -> tuple[int, int]: ...

    @classmethod
    def slot_states(cls, config) -> tuple[SlotStateSpec, ...]: ...
