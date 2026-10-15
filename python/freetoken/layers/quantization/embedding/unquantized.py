"""Dense tables: the indexing kernel looks rows up; a tied head runs the unquantized linear kernel on the same table."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from ..linear.unquantized import TorchLinearKernel
from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import EmbeddingKernel, EmbeddingMethod


class IndexingEmbeddingKernel(EmbeddingKernel):
    name = "indexing"

    def embed(self, layer: Any, ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        return indexing(weights=layer.weight, indices=ids, vocab_range=layer.vocab_range if layer.tp_size > 1 else None)

    def logits(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return TorchLinearKernel().apply(SimpleNamespace(weight=layer.weight, bias=bias), x)


@register_method(QuantKind.NONE, LayerKind.EMBEDDING)
class UnquantizedEmbeddingMethod(EmbeddingMethod):
    candidates = (IndexingEmbeddingKernel,)

    def create_weights(self, layer: Any) -> None:
        layer.weight = torch.empty(self.cfg.num_embeddings, self.cfg.embedding_dim)
