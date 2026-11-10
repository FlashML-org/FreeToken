"""DF11 tables: the bf16 table compressed losslessly at load into row-contiguous codes, so a lookup decodes only the rows it gathers."""

from __future__ import annotations

from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import EmbeddingConfig, EmbeddingKernel, EmbeddingMethod

DF11_KEYS = ("low8", "bitstream", "chunk_start", "lut")


class Df11EmbeddingKernel(EmbeddingKernel):
    name = "triton"

    def unusable_reason(self, cfg: EmbeddingConfig) -> str | None:
        if cfg.tp_size > 1:
            return "DF11 tables are not sharded over TP"
        return None if cfg.dtype == torch.bfloat16 else f"DF11 compresses bf16 tables, the model runs {cfg.dtype}"

    def finalize(self, layer: Any) -> None:
        from freetoken.kernel.triton.df11 import df11_compress_rows

        c = df11_compress_rows(layer.weight.cuda())
        del layer.weight
        for key in DF11_KEYS:
            setattr(layer, key, c[key])
        layer.df11_meta = c["meta"]

    def embed(self, layer: Any, ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.df11_decode import df11_gather_decode

        bundle = {key: getattr(layer, key) for key in DF11_KEYS}
        bundle["meta"] = layer.df11_meta
        return df11_gather_decode(bundle, ids).view(*ids.shape, layer.embedding_dim)

    def logits(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError("a DF11 table cannot serve a tied LM head")


@register_method(QuantKind.DF11, LayerKind.EMBEDDING)
class Df11EmbeddingMethod(EmbeddingMethod):
    candidates = (Df11EmbeddingKernel,)
    online = True

    def create_weights(self, layer: Any) -> None:
        layer.weight = torch.empty(self.cfg.num_embeddings, self.cfg.embedding_dim)
