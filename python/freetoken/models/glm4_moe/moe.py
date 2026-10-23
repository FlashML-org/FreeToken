from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
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
        # Keep selection bias in fp32: rounding can change the selected experts.
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)

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

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        logits = torch.mm(hidden_states, self.gate.weight.t(), out_dtype=torch.float32)
        return fused_topk_sigmoid(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.norm_topk_prob,
            routed_scale=self.routed_scaling_factor,
            n_group=self.n_group,
            topk_group=self.topk_group,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(hidden_states)
        # Routed kernels may overwrite hidden_states; shared experts need the original input.
        shared = self.shared_experts.forward(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        out = out + shared
        return out.view(num_tokens, hidden_dim)


__all__ = ["Glm4MoeSparseBlock"]
