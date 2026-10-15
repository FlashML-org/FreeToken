"""GGUF tables: packed ggml rows, decoded per looked-up row; a tied head runs the GGUF linear kernel on the same rows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from ..ggml_types import Segment, pad_rows, row_bytes
from ..linear.gguf import GgmlLinearKernel
from ..registry import LayerKind, register_method
from ..scheme import QuantKind, gguf_types
from .base import EmbeddingConfig, EmbeddingKernel, EmbeddingMethod


class GgmlEmbeddingKernel(EmbeddingKernel):
    name = "ggml"

    def unusable_reason(self, cfg: EmbeddingConfig) -> str | None:
        return "GGUF tables are not sharded over TP" if cfg.tp_size > 1 else None

    def embed(self, layer: Any, ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import ggml

        cfg = layer.quant_method.cfg
        (ggml_type,) = gguf_types(cfg.scheme)
        rows = ggml.get_rows(layer.weight, ggml_type, cfg.embedding_dim, ids.reshape(-1).to(torch.int32), cfg.dtype)
        return rows.view(*ids.shape, cfg.embedding_dim)

    def logits(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        cfg = layer.quant_method.cfg
        (ggml_type,) = gguf_types(cfg.scheme)
        table = SimpleNamespace(_ggml_segments=[Segment(ggml_type, 0, 1, 0, cfg.num_embeddings)], weight=layer.weight, bias=bias)
        return GgmlLinearKernel().apply(table, x)


@register_method(QuantKind.GGUF, LayerKind.EMBEDDING)
class GGUFEmbeddingMethod(EmbeddingMethod):
    candidates = (GgmlEmbeddingKernel,)

    def create_weights(self, layer: Any) -> None:
        (ggml_type,) = gguf_types(self.cfg.scheme)
        h = self.cfg.embedding_dim
        layer.weight = torch.empty(self.cfg.num_embeddings + pad_rows(h), row_bytes(ggml_type, h), dtype=torch.uint8)
