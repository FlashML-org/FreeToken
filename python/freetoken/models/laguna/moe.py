from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer, silu_and_mul

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class LagunaMLP(BaseOP):
    """Plain SwiGLU MLP used by Laguna dense and shared experts."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearReplicated(hidden_size, 2 * intermediate_size, has_bias=False)
        self.down_proj = LinearReplicated(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = silu_and_mul(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class LagunaSparseMoeBlock(BaseOP):
    """Mixture-of-experts block for Laguna."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        # Router weights are fp32; this is required for exact tie-breaking at the top-k boundary.
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)

        moe_layer_id = layer_id - config.first_k_dense_replace
        # Mixed-type GGUF banks ("gguf" offload format): the kernels need this
        # layer's ggml types and output-row geometry (see fused_experts_gguf).
        extra_attrs = None
        if config.gguf_expert_types is not None:
            gu_t, dn_t = config.gguf_expert_types[moe_layer_id]
            extra_attrs = {
                "gguf_gate_up_type": gu_t,
                "gguf_down_type": dn_t,
                "gguf_gate_up_rows": 2 * config.moe_intermediate_size,
                "gguf_down_rows": config.hidden_size,
            }
        self.experts = make_moe_layer(
            config,
            layer_id=moe_layer_id,
            renormalize=config.norm_topk_prob,
            activation="silu",
            extra_attrs=extra_attrs,
        )
        self.shared_experts = LagunaMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size * max(1, config.n_shared_experts),
        )

    def _route(self, hidden_states: torch.Tensor) -> TopK:
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


__all__ = ["LagunaMLP", "LagunaSparseMoeBlock"]
