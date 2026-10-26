"""The LM head a draft predicts with; every kind returns target-vocab ids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple

import torch

if TYPE_CHECKING:
    from freetoken.layers import ParallelLMHead


class DraftHead(Protocol):
    vocab_size: int

    def argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """Target-vocab id of each row's top logit."""
        ...

    def topk(self, hidden: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(logits [rows, k], target-vocab ids [rows, k]) of each row's top k."""
        ...


class SharedTargetHead:
    """The target's own LM head over the full vocabulary."""

    def __init__(self, head: ParallelLMHead) -> None:
        self.head = head
        self.vocab_size = head.num_embeddings

    def argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head.logits(hidden).argmax(dim=-1)

    def topk(self, hidden: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head.logits(hidden).topk(k, dim=-1)
