from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.glm4_moe.mlp import GlmGatedMLP
from freetoken.utils import nvtx_annotate

from .attention import MiMoV2Attention
from .moe import MiMoV2SparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class MiMoV2DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.self_attn = MiMoV2Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = MiMoV2SparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        else:
            # moe_layer_freq[0] == 0: layer 0 is a plain dense SwiGLU MLP (no gate/bias).
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


class MiMoV2Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                MiMoV2DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
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


class MiMoV2ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        tp_size = get_tp_info().size
        if tp_size != 1:
            raise ValueError(f"MiMo v2 supports tensor parallel size 1 only; got TP={tp_size}")
        self.model = MiMoV2Model(config)
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


__all__ = ["MiMoV2ForCausalLM"]
