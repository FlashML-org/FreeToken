from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import GlmMoeDsaAttention
from .mlp import GlmDsaGatedMLP
from .moe import GlmMoeDsaSparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class GlmMoeDsaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.self_attn = GlmMoeDsaAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = GlmMoeDsaSparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = GlmDsaGatedMLP(
                config.hidden_size, config.intermediate_size,
                quant_config=config.quant, prefix=f"{prefix}.mlp",
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


class GlmMoeDsaModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                GlmMoeDsaDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
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


class GlmMoeDsaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = GlmMoeDsaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook (engine calls it before the pool family's solve_num_pages):
        materialize every layer's bmm-ready kv_b split and free the checkpoint-layout
        originals, so the ~2.2 GiB repack is measured by the sizing pass instead of
        overcommitting the KV budget on the first forward (gpt_oss precedent)."""
        import torch

        for layer in self.model.layers.op_list:
            layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["GlmMoeDsaForCausalLM"]
