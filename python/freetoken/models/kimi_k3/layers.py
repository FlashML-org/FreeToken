"""Kimi-K3 decoder layer and attention-residual aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.utils import nvtx_annotate

from .attention import KimiDeltaAttention, KimiMLAAttention
from .mlp import KimiMLP
from .moe import KimiSparseMoeBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: LinearReplicated,
    norm: RMSNorm,
) -> torch.Tensor:
    """Softmax-weighted aggregation from the official Kimi attention residual."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    vf = values.float()
    normalized = vf * torch.rsqrt(vf.square().mean(-1, keepdim=True) + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    probs = (normalized * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probs, vf).squeeze(1).to(values.dtype)


class KimiDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.kimi_k3_args
        if args is None:
            raise ValueError("KimiDecoderLayer requires ModelConfig.kimi_k3_args")
        self._layer_id = layer_id
        self.layer_id = layer_id
        self.is_linear_attn = config.is_linear_layer(layer_id)
        self.self_attn: BaseOP = (
            KimiDeltaAttention(config, layer_id)
            if self.is_linear_attn
            else KimiMLAAttention(config, layer_id)
        )
        if layer_id >= config.first_k_dense_replace:
            self.block_sparse_moe = KimiSparseMoeBlock(config, layer_id)
        else:
            self.mlp = KimiMLP(
                config.hidden_size,
                config.intermediate_size,
                beta=args.situ_beta,
                linear_beta=args.situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attention_res_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = LinearReplicated(
            config.hidden_size, 1, has_bias=False
        )
        self.mlp_res_proj = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self.attn_res_block_size = args.attn_res_block_size

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, hidden_states: torch.Tensor, block_residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = apply_attention_residual(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )
        if self.layer_id % self.attn_res_block_size == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm.forward(hidden_states)
        mixed = self.self_attn.forward(hidden_states)
        prefix_sum = mixed if prefix_sum is None else prefix_sum + mixed

        hidden_states = apply_attention_residual(
            prefix_sum, block_residual, self.mlp_res_proj, self.mlp_res_norm
        )
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe.forward(hidden_states)
        else:
            hidden_states = self.mlp.forward(hidden_states)
        return prefix_sum + hidden_states, block_residual


__all__ = ["KimiDecoderLayer", "apply_attention_residual"]
