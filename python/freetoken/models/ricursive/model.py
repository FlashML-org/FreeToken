from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    silu_and_mul,
    swiglu_clamp_and_mul,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.config import ModelConfig
from freetoken.utils import nvtx_annotate

from .attention import RicursiveAttention


class RicursiveMLP(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [config.intermediate_size, config.intermediate_size],
            has_bias=False, quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearReplicated(
            config.intermediate_size, config.hidden_size,
            has_bias=False, quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )
        self.limit = config.swiglu_limit

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        if self.limit is None:
            return self.down_proj.forward(silu_and_mul(gate_up))
        return self.down_proj.forward(swiglu_clamp_and_mul(gate_up, alpha=1.0, limit=self.limit))


class RicursiveDecoderLayer(BaseOP):
    """Four-norm block: x += post_attn(attn(pre_attn(x))); x += post_ffn(mlp(pre_ffn(x)))."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self.self_attn = RicursiveAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = RicursiveMLP(config, prefix=f"{prefix}.mlp")
        H, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)
        self.pre_feedforward_layernorm = RMSNormFused(H, eps)
        self.post_feedforward_layernorm = RMSNorm(H, eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, value_residual: torch.Tensor) -> torch.Tensor:
        h = self.self_attn.forward(self.input_layernorm.forward(x), value_residual)
        h = self.post_attention_layernorm.forward(h)
        # Fused add + norm: x becomes x + h in place, h its norm.
        h, x = self.pre_feedforward_layernorm.forward(h, x)
        return x + self.post_feedforward_layernorm.forward(self.mlp.forward(h))


class RicursiveModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.embedding_scale,
            quant_config=config.quant, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = OPList([
            RicursiveDecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Layers add into x in place; every layer's V still reads the scaled embeddings.
        value_residual = self.embed_tokens.forward(input_ids)
        x = value_residual.clone()
        for layer in self.layers.op_list:
            x = layer.forward(x, value_residual)
        return self.norm.forward(x)


class RicursiveForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        if get_tp_info().size > 1:
            raise NotImplementedError("Ricursive supports TP=1 only")
        self.model = RicursiveModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant, prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        return self.lm_head.forward(self.model.forward(get_global_ctx().batch.input_ids))
