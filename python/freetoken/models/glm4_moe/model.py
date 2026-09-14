from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Glm4MoeAttention
from .df11_embedding import EmbeddingDF11
from .mlp import GlmGatedMLP
from .moe import Glm4MoeSparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Glm4MoeDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.self_attn = Glm4MoeAttention(config, layer_id)
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = Glm4MoeSparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = GlmGatedMLP(
                config.hidden_size, config.intermediate_size, quant_config=config.quant, prefix=f"{prefix}.mlp"
            )
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Glm4MoeModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = EmbeddingDF11(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Glm4MoeDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Glm4MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Glm4MoeModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Glm4MoeForCausalLM"]
