from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, GatedMLP
from freetoken.utils import nvtx_annotate

from .attention import LagunaAttention
from .moe import LagunaSparseMoeBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LagunaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self.self_attn = LagunaAttention(config, layer_id)
        # Determine dense vs MoE. Laguna S 2.1: mlp_only_layers [0] → layer 0 dense,
        # rest MoE sparse. No decoder_sparse_step concept beyond that.
        # We also respect generic first_k_dense_replace if set.
        is_dense = False
        # Check mlp_only_layers style: we didn't store it on config yet, but
        # config.intermediate_size path for dense can be used: if layer 0 or
        # first_k_dense_replace covers this layer.
        # For Laguna we know mlp_only_layers=[0]; encode it via a model-config
        # field would be ideal, but we can inline the check here via num_layers/type.
        # Since we don't have the list on ModelConfig, treat layer 0 as dense when
        # model_type is laguna and num_experts>0.
        if config.model_type == "laguna":
            mlp_only = {0}
            if layer_id in mlp_only:
                is_dense = True
        if getattr(config, "first_k_dense_replace", 0) > layer_id:
            is_dense = True
        if not config.moe_enabled:
            is_dense = True
        if is_dense:
            # Dense GatedMLP uses intermediate_size (12288).
            self.mlp = GatedMLP(config)
        else:
            self.mlp = LagunaSparseMoeBlock(config, layer_id)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class LagunaModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList([LagunaDecoderLayer(config, lid) for lid in range(config.num_layers)])
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class LagunaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = LagunaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["LagunaForCausalLM", "LagunaModel", "LagunaDecoderLayer"]
