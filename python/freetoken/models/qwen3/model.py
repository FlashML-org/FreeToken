from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.utils import nvtx_annotate

from freetoken.models.blocks import BaseLLMModel, GatedMLP as Qwen3MLP

from .attention import Qwen3Attention as Qwen3Attn

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


from . import probe_state as _ps


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        if _ps.PROBE_LAYERS and _ps.CURRENT_POSITIONS is not None:
            _ps.record_layer(self._layer_id, _ps.CURRENT_POSITIONS, x)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        positions = getattr(batch, "positions", None)
        if _ps.PROBE_LAYERS:
            _ps.reset_if_new_request(positions)
            _ps.CURRENT_PHASE = "prefill" if getattr(batch, "is_prefill", False) else "decode"
            if positions is not None:
                _ps.record_forward_meta(_ps.CURRENT_PHASE, positions)
                _ps.CURRENT_POSITIONS = positions
        x = self.embed_tokens.forward(input_ids)
        if _ps.PROBE_LAYERS and positions is not None:
            _ps.record_embedding(positions, x)
            if _ps.CURRENT_PHASE == "prefill":
                _ps.record_input_ids(input_ids)
        residual: torch.Tensor | None = None
        for i, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
        if _ps.PROBE_LAYERS:
            try:
                _ps.finalize("/tmp/ft_probe.npz")
            except Exception:
                pass
            _ps.CURRENT_POSITIONS = None
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3ForCausalLM"]
