"""MiniMax-M3 sparse MoE block.

Routing follows the DeepSeek/GLM family (``GlmMoeDsaSparseBlock``): fp32 sigmoid
scores + a selection-only ``e_score_correction_bias``, gather the unbiased scores,
renormalize, scale by ``routed_scaling_factor`` (2.0). No group-limited top-k
(n_group == 1). The routed experts (NVFP4, swigluoai) go through ``make_moe_layer``
(the offload family; the swigluoai alpha/limit are the layer's ``alpha`` /
``limit``) and the always-on shared expert (MXFP8, swigluoai) is added.

The checkpoint keeps ``e_score_correction_bias`` fp32 and the reference applies it
fp32; the bias sits on the top-k boundary, so it is stored fp32 here too (unlike
GLM's bf16 storage, whose checkpoint ships it fp32 but consumes it off a bf16 cast).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .mlp import MiniMaxM3MLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class MiniMaxM3SparseMoeBlock(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.m3_args
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        # Router weights kept fp32 (vLLM stores them fp32 too): both the gate and
        # the bias sit on the top-4 selection boundary, where a bf16 round can
        # flip near-tie picks away from the reference. 128 x 6144 x 4 B = ~3.1 MB
        # per layer, ~180 MB across the 57 MoE layers -- accepted for selection
        # fidelity.
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.weight = torch.empty(
            config.num_experts, config.hidden_size, dtype=torch.float32
        )
        # DeepSeek-style selection bias; fp32 in the checkpoint AND here.
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)

        # The offload cache indexes experts by *MoE* layer (global layer minus
        # first_k_dense_replace), matching how the loader packs the expert banks.
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            activation="swigluoai",
            alpha=args.swiglu_alpha,
            limit=args.swiglu_limit,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = MiniMaxM3MLP(
            config.hidden_size,
            config.shared_expert_intermediate_size * max(1, config.n_shared_experts),
            alpha=args.swiglu_alpha,
            limit=args.swiglu_limit,
            quant_config=config.quant,
            prefix=f"{prefix}.shared_experts",
        )

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        # HF/vLLM compute the router logits in fp32; the gate is stored fp32.
        logits = F.linear(hidden_states.float(), self.gate.weight)
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
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


__all__ = ["MiniMaxM3SparseMoeBlock"]
