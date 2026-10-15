"""LFM2-MoE model: hybrid short-conv / GQA decoder with a dense MLP prefix and routed experts.

HF's ``x = x + mixer(operator_norm(x)); x = x + feed_forward(ffn_norm(x))`` is carried in the
residual-fused form (``RMSNormFused.forward(x, residual)``), one add+norm kernel per sublayer
instead of two. The norms are PLAIN RMSNorm (HF's ``Lfm2MoeRMSNorm`` scales by the raw
weight), the final norm is ``embedding_norm`` and the head is tied to the embedding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.utils import nvtx_annotate

from .attention import Lfm2Attention
from .conv import Lfm2ShortConv
from .moe import Lfm2DenseMLP, Lfm2MoeSparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Lfm2MoeDecoderLayer(BaseOP):
    """Pre-norm hybrid block: the mixer is a short conv or GQA attention, the FFN dense or MoE."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "") -> None:
        self._layer_id = layer_id
        self._is_conv = config.is_short_conv_layer(layer_id)
        if self._is_conv:
            self.conv = Lfm2ShortConv(config, layer_id, prefix=f"{prefix}.conv")
        else:
            self.self_attn = Lfm2Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.feed_forward = (
            Lfm2MoeSparseBlock(config, layer_id, prefix=f"{prefix}.feed_forward")
            if layer_id >= config.first_k_dense_replace
            else Lfm2DenseMLP(config, prefix=f"{prefix}.feed_forward")
        )
        self.operator_norm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.operator_norm.forward(x, residual)
        x = self.conv.forward(x) if self._is_conv else self.self_attn.forward(x)
        x, residual = self.ffn_norm.forward(x, residual)
        x = self.feed_forward.forward(x)
        return x, residual


class Lfm2MoeModel(BaseOP):
    def __init__(self, config: "ModelConfig", *, prefix: str = "model") -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Lfm2MoeDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.embedding_norm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.embedding_norm.forward(x, residual)[0]


class Lfm2MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig") -> None:
        self.model = Lfm2MoeModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

        # GGUF checkpoints carry native block-quantized weights: swap the dense projections
        # and the embedding (also the tied head) for GGUF-quant ops; the routed experts stay
        # on the offload cache as native K-quant banks.
        from .gguf import convert_lfm2_moe_to_gguf, is_gguf_model

        if is_gguf_model(config):
            convert_lfm2_moe_to_gguf(self, config)

    def forward(self) -> torch.Tensor:
        return self.lm_head.forward(self.model.forward(get_global_ctx().batch.input_ids))


__all__ = ["Lfm2MoeForCausalLM", "Lfm2MoeModel"]
