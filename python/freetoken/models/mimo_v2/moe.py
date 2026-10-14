from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class MiMoV2SparseBlock(BaseOP):
    """MiMo v2 sparse MoE block: top-k routed experts only (``n_shared_experts`` is None).

    Uses fp32 gate GEMM, sigmoid scores, correction bias for selection only,
    unbiased contribution scores, optional normalization, and routed scaling.
    The supplied external ``MiMoV2MoEGate`` is provenance, not an exact-equivalence
    oracle: it requests unsorted expert top-k whereas this port uses sorted top-k.
    Synthetic routing tests do not establish full-checkpoint numerical equivalence.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # Keep selection bias in fp32: rounding can change the selected experts.
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)

        # The offload cache indexes experts by *MoE* layer (global layer minus
        # first_k_dense_replace), matching how the loader packs the expert banks. The
        # per-expert compute is the unified OffloadMoELayer, shared with GLM-4 MoE.
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
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
        # Keep logits in fp32 to avoid rounding before sigmoid and expert selection.
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.float()
        if self.n_group > 1:
            scores_for_choice = self._group_limited(scores_for_choice)
        _, topk_ids = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob and self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        return out.view(num_tokens, hidden_dim)


__all__ = ["MiMoV2SparseBlock"]
