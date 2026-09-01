"""FreeToken inference model for the Kimi-K3 text tower."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel

from .layers import KimiDecoderLayer, apply_attention_residual

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class KimiK3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        args = config.kimi_k3_args
        if args is None:
            raise ValueError("KimiK3Model requires ModelConfig.kimi_k3_args")
        if get_tp_info().size != 1:
            raise NotImplementedError(
                "Kimi-K3 text inference currently supports tensor parallel size 1"
            )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = OPList(
            [
                KimiDecoderLayer(config, layer_id)
                for layer_id in range(config.num_layers)
            ]
        )
        self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_proj = LinearReplicated(
            config.hidden_size, 1, has_bias=False
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens.forward(input_ids)
        block_residual = hidden_states.new_empty(
            hidden_states.shape[0], 0, hidden_states.shape[1]
        )
        for layer in self.layers.op_list:
            hidden_states, block_residual = layer.forward(hidden_states, block_residual)
        hidden_states = apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        return self.norm.forward(hidden_states)


class KimiK3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = KimiK3Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens
            if config.tie_word_embeddings
            else None,
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        for layer in self.model.layers.op_list:
            if not layer.is_linear_attn:
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["KimiK3ForCausalLM", "KimiK3Model"]
