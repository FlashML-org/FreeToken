"""The embedding side: per-layer config, the kernel ABC and the Method base. The method owns the vocab table: it looks rows up, and a tied LM head multiplies by the same table through it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ..method import QuantMethod
from ..scheme import QuantScheme


@dataclass(frozen=True)
class EmbeddingConfig:
    """The TP-local table shape, the dtype a lookup returns, and the table's scheme."""

    num_embeddings: int
    embedding_dim: int
    tp_size: int
    dtype: torch.dtype
    scheme: QuantScheme | None = None

    @classmethod
    def from_layer(cls, layer: Any, scheme: QuantScheme | None) -> "EmbeddingConfig":
        return cls(layer.num_embeddings_tp, layer.embedding_dim, layer.tp_size, torch.get_default_dtype(), scheme)


class EmbeddingKernel(ABC):
    """One backend for one table kind: the row lookup, and ``x @ table^T`` for a tied LM head."""

    name: ClassVar[str]

    def unusable_reason(self, cfg: EmbeddingConfig) -> str | None:
        return None

    def worth_it(self, cfg: EmbeddingConfig) -> bool:
        return True

    def finalize(self, layer: Any) -> None:
        pass

    @abstractmethod
    def embed(self, layer: Any, ids: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def logits(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor: ...


class EmbeddingMethod(QuantMethod):
    """Declares the table for its kind; the lookup and the tied head's product go to the kernel."""

    @abstractmethod
    def create_weights(self, layer: Any) -> None: ...

    def finalize(self, layer: Any) -> None:
        self.kernel.finalize(layer)

    def embed(self, layer: Any, ids: torch.Tensor) -> torch.Tensor:
        return self.kernel.embed(layer, ids)

    def logits(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.kernel.logits(layer, x, bias)
