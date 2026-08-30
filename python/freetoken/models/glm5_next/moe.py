from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .mlp import Glm5NextMLP


class Glm5NextSparseBlock(BaseOP):
    def __init__(self, config, layer_id: int) -> None:
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False
        )
        self.e_score_correction_bias = torch.empty(config.num_experts)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            extra_attrs={"swiglu_limit": config.swiglu_limit},
        )
        self.shared_experts = Glm5NextMLP(
            config.hidden_size,
            config.moe_intermediate_size * max(1, config.n_shared_experts),
            config.swiglu_limit,
        )

    def _route(self, x: torch.Tensor):
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        choice = scores + self.e_score_correction_bias.float()
        if self.n_group > 1:
            m, e, g = x.shape[0], self.num_experts, self.n_group
            group_scores = choice.view(m, g, e // g).topk(2, dim=-1)[0].sum(-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1, sorted=False)[1]
            mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1).bool()
            choice = choice.masked_fill(
                ~mask.unsqueeze(-1).expand(m, g, e // g).reshape(m, e), float("-inf")
            )
        ids = choice.topk(self.top_k, dim=-1)[1]
        weights = scores.gather(-1, ids)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return (
            weights * self.routed_scaling_factor
        ).float().contiguous(), ids.int().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, shape[-1])
        weights, ids = self._route(x)
        routed = self.experts.routed_forward(x, weights, ids)
        return (routed + self.shared_experts.forward(x)).view(shape)


__all__ = ["Glm5NextSparseBlock"]
