from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaPlusOneRMSNormFused,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import MiniMaxM3Attention
from .mlp import MiniMaxM3MLP
from .moe import MiniMaxM3SparseMoeBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class MiniMaxM3DecoderLayer(BaseOP):
    """Dense layers store the FFN under ``mlp``; MoE layers under
    ``block_sparse_moe`` -- matching the checkpoint's naming. All layernorms are
    Gemma-style (1+w) RMSNorm (``use_gemma_norm``)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.m3_args
        self.self_attn = MiniMaxM3Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.is_moe_layer = layer_id in args.moe_layer_ids
        if self.is_moe_layer:
            self.block_sparse_moe: BaseOP = MiniMaxM3SparseMoeBlock(
                config, layer_id, prefix=f"{prefix}.block_sparse_moe"
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config.hidden_size,
                args.dense_intermediate_size,
                alpha=args.swiglu_alpha,
                limit=args.swiglu_limit,
                quant_config=config.quant,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = GemmaPlusOneRMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaPlusOneRMSNormFused(
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
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        x = ffn.forward(x)
        return x, residual


class MiniMaxM3Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                MiniMaxM3DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaPlusOneRMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class MiniMaxM3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = MiniMaxM3Model(config)
        # lm_head is BF16 in the checkpoint (excluded from quantization), untied.
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


__all__ = ["MiniMaxM3ForCausalLM"]
