"""GLM-5.2 sparse MoE block.

Identical routing to glm4_moe (``Glm4MoeSparseBlock``): sigmoid scores + a selection-only
``e_score_correction_bias``, optional group-limited top-k, gather unbiased scores,
renormalize, scale by ``routed_scaling_factor``. The routed experts go through
``make_moe_layer`` (the offload family for NVFP4) and the always-on shared expert is
added. The only GLM-5.2 difference from glm4_moe is the shared expert is bf16
(unquantized) rather than NVFP4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import os as _os

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .mlp import GlmDsaGatedMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class GlmMoeDsaSparseBlock(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # DeepSeek-style selection bias; kept fp32 in HF, stored in the model dtype and
        # upcast at use (exact enough for the argmax-style top-k selection).
        self.e_score_correction_bias = torch.empty(config.num_experts)

        # The offload cache indexes experts by *MoE* layer (global layer minus
        # first_k_dense_replace), matching how the loader packs the expert banks.
        # Subclasses (glm5_next: resident split) override _make_experts.
        self.experts = self._make_experts(config, layer_id)
        self.shared_experts = GlmDsaGatedMLP(
            config.hidden_size,
            config.moe_intermediate_size * max(1, config.n_shared_experts),
            quant=config.dense_quant,
        )

    def _make_experts(self, config: ModelConfig, layer_id: int):
        return make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            # GLM-5.3 experts clamp swiglu at config.swiglu_limit (GLM-5.2: None -> silu).
            # Only the Triton NVFP4 kernels honor the limit, so backend auto resolves there
            # (marlin/b12x hard-code plain silu and are rejected for this kind).
            activation="swiglu_clamp" if getattr(config, "swiglu_limit", None) else "silu",
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
        if self.n_group <= 1 and _os.environ.get("FREETOKEN_FUSED_ROUTE", "1") == "1":
            # tail-fusion (2026-08-28): sigmoid/+bias/topk/gather/renorm/cast chain
            # (~8 launches) -> one kernel. Same math; see fused_route docstring.
            from freetoken.kernel.triton.fused_route import fused_route

            return fused_route(
                logits, self.e_score_correction_bias, self.top_k,
                self.norm_topk_prob, self.routed_scaling_factor,
            )
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
        import os as _os
        if _os.environ.get("FREETOKEN_GLM5_STUB_MOE", "0") == "1":  # ablation timing only
            return torch.zeros_like(hidden_states)
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        out = out + self.shared_experts.forward(hidden_states)
        return out.view(num_tokens, hidden_dim)


__all__ = ["GlmMoeDsaSparseBlock"]
