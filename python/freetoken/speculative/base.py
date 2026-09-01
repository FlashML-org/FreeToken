from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.engine.engine import Engine
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.models import BaseLLMModel


class SpecAlgorithm(Enum):
    """Builtin speculative decoding algorithms."""

    NONE = "none"
    DFLASH = "dflash"

    @classmethod
    def from_string(cls, name: Optional[str]) -> SpecAlgorithm:
        if name is None:
            return cls.NONE
        return cls(name.lower())


class BaseSpecWorker(ABC):
    """Abstract interface for speculative decoding workers.

    The engine calls through this interface; concrete implementations
    (DFlash, future EAGLE/ngram) live under ``speculative/<algo>/``.
    """

    block_size: int
    target_layer_ids: set[int]

    @abstractmethod
    def draft(
        self,
        hidden_states: list[torch.Tensor],
        base_token_id: torch.Tensor,
        position: int,
        *,
        sampling_args=None,
    ) -> torch.Tensor:
        """Generate ``block_size`` draft tokens in parallel."""

    @abstractmethod
    def store_hidden_states(self, hidden_states: list[torch.Tensor]) -> None:
        """Store target hidden states as draft context."""

    @abstractmethod
    def reset_context(self) -> None:
        """Clear context when a request finishes."""

    @abstractmethod
    def target_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project target hidden states to logits (all positions, no prefill slicing)."""

    @abstractmethod
    def forward(self, batch: Batch, args: BatchSamplingArgs):
        """Full spec-decode forward: base + draft + verify + commit.

        Returns a ``ForwardOutput`` with ``num_tokens > 1``.
        """


def create_spec_worker(
    algorithm: SpecAlgorithm,
    *,
    target_model: BaseLLMModel,
    engine: Engine,
    device: torch.device,
    draft_model_path: str,
    block_size: int,
) -> Optional[BaseSpecWorker]:
    """Factory: create a spec worker by algorithm name, or None if disabled."""
    if algorithm == SpecAlgorithm.NONE:
        return None
    if algorithm == SpecAlgorithm.DFLASH:
        from freetoken.speculative.dflash.worker import DFlashWorker

        return DFlashWorker(
            draft_model_path=draft_model_path,
            target_model=target_model,
            engine=engine,
            device=device,
            block_size=block_size,
        )
    raise ValueError(f"Unknown speculative algorithm: {algorithm}")
