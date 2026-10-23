from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, GatedMLP
from freetoken.models.config import ModelConfig
from freetoken.utils import nvtx_annotate

from .attention import FlintAttention
from .moe import FlintMoE


class FlintDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self.self_attn = FlintAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = (
            GatedMLP(config, quant_config=config.quant, prefix=f"{prefix}.mlp")
            if layer_id < config.first_k_dense_replace
            else FlintMoE(config, layer_id, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        return self.mlp.forward(x), residual


class FlintModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, quant_config=config.quant, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = OPList([
            FlintDecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in range(config.num_layers)
        ])
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x, residual = self.embed_tokens.forward(input_ids), None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class FlintForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = FlintModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant, prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)
