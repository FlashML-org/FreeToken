"""Latent routed Kimi-K3 MoE with noaux_tc sigmoid routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, RMSNorm, make_moe_layer
from freetoken.moe import is_offload_moe_backend

from .mlp import KimiMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = tuple[torch.Tensor, torch.Tensor]


class KimiSparseMoeBlock(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.kimi_k3_args
        if args is None:
            raise ValueError("KimiSparseMoeBlock requires ModelConfig.kimi_k3_args")
        if config.expert_quant == "mxfp4" and not is_offload_moe_backend(
            config.moe_backend
        ):
            raise ValueError(
                "Kimi-K3 MXFP4 experts require an offload-family MoE backend"
            )
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.renormalize = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False
        )
        # This routing-only correction stays FP32 in the official checkpoint.
        self.e_score_correction_bias = torch.empty(
            config.num_experts, dtype=torch.float32
        )

        self.routed_expert_down_proj = LinearReplicated(
            config.hidden_size, args.routed_expert_hidden_size, has_bias=False
        )
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            activation="situ",
            hidden_size=args.routed_expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            extra_attrs={
                "hidden_act_alpha": args.situ_beta,
                "swiglu_limit": args.situ_linear_beta,
            },
        )
        self.routed_expert_norm = RMSNorm(
            args.routed_expert_hidden_size, eps=config.rms_norm_eps
        )
        self.routed_expert_up_proj = LinearReplicated(
            args.routed_expert_hidden_size, config.hidden_size, has_bias=False
        )
        self.shared_experts = KimiMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            beta=args.situ_beta,
            linear_beta=args.situ_linear_beta,
        )

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        choice = scores + self.e_score_correction_bias.float()
        if self.n_group > 1 and self.n_group > self.topk_group:
            n, e = choice.shape
            group_scores = (
                choice.view(n, self.n_group, e // self.n_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_ids = group_scores.topk(self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, group_ids, True)
            choice = choice.masked_fill(
                ~group_mask.unsqueeze(-1)
                .expand(n, self.n_group, e // self.n_group)
                .reshape(n, e),
                float("-inf"),
            )
        topk_ids = choice.topk(self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
        if self.top_k > 1 and self.renormalize:
            topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.float().contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        topk_weights, topk_ids = self._route(hidden_states)
        latent = self.routed_expert_down_proj.forward(hidden_states)
        routed = self.experts.routed_forward(latent, topk_weights, topk_ids)
        routed = self.routed_expert_norm.forward(routed)
        routed = self.routed_expert_up_proj.forward(routed)
        return routed + self.shared_experts.forward(identity)


__all__ = ["KimiSparseMoeBlock"]
