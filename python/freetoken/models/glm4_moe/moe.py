from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .mlp import GlmGatedMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class Glm4MoeSparseBlock(BaseOP):
    """GLM-4 sparse MoE block: top-k routed experts + one always-on shared expert.

    Routing matches HF ``Glm4MoeMoE.route_tokens_to_experts``: sigmoid scores, add the
    ``e_score_correction_bias`` for *selection only*, (optional) group-limited top-k,
    gather the unbiased sigmoid scores, renormalize, then scale by
    ``routed_scaling_factor``. The shared expert is applied to the same input and added.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # DeepSeek-style selection bias; a registered buffer in HF (kept fp32 there). We
        # store it in the model's bf16 dtype and upcast at use, which is exact enough for
        # the argmax-style top-k selection.
        self.e_score_correction_bias = torch.empty(config.num_experts)

        # The offload cache indexes experts by *MoE* layer (global layer minus
        # first_k_dense_replace), matching how the loader packs the expert banks. The
        # per-expert compute is the unified OffloadMoELayer, shared with MiniMax-M2.
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = GlmGatedMLP(
            config.hidden_size,
            config.moe_intermediate_size * max(1, config.n_shared_experts),
            quant_config=config.quant,
            prefix=f"{prefix}.shared_experts",
        )

    def _group_limited(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        m = scores_for_choice.shape[0]
        e, g = self.num_experts, self.n_group
        group_scores = scores_for_choice.view(m, g, e // g).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(-1).expand(m, g, e // g).reshape(m, e)
        return scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        # HF computes the router logits in fp32; the gate is tiny so we match it exactly.
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.float()
        if self.n_group > 1:
            scores_for_choice = self._group_limited(scores_for_choice)
        _, topk_ids = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        out = out + self.shared_experts.forward(hidden_states)
        return out.view(num_tokens, hidden_dim)


__all__ = ["Glm4MoeSparseBlock"]
