from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, make_moe_layer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LagunaGate(BaseOP):
    """Gate with bias for HF-faithful keys mlp.gate.weight + mlp.gate.e_score_correction_bias."""

    def __init__(self, hidden_size: int, num_experts: int):
        self.weight = torch.empty(num_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class LagunaSparseMoeBlock(BaseOP):
    """Laguna MoE: sigmoid-routed experts + shared expert.

    Uses the offload/resident seam ``make_moe_layer(...).routed_forward`` so
    the existing MoE kernels are reused unchanged. Keys are
    ``mlp.gate.weight`` + ``mlp.gate.e_score_correction_bias`` (HF-faithful).
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = int(layer_id)
        first_dense = int(getattr(config, "first_k_dense_replace", 0))
        offload_id = layer_id - first_dense
        if offload_id < 0:
            offload_id = 0
        self._offload_id = offload_id
        hidden = int(config.hidden_size)
        self.gate = LagunaGate(hidden, int(config.num_experts))
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.experts = make_moe_layer(
            config,
            layer_id=offload_id,
            # ``_route`` below already renormalized, and ``routed_forward`` bypasses the
            # layer's internal router entirely -- so the flag must not re-apply it.
            renormalize=False,
        )
        shared_inter = int(config.shared_expert_intermediate_size or config.moe_intermediate_size)
        self.shared_expert = _SharedExpert(config.hidden_size, shared_inter, config.hidden_act, config.rms_norm_eps)

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sigmoid router (modeling_laguna.py LagunaTopKRouter.forward).

        The bias shifts SELECTION only; the returned weights are gathered from the
        unbiased scores, then renormalized and scaled -- the DeepSeek/GLM ``noaux_tc``
        convention (cf. ``glm4_moe/moe.py``).
        """
        logits = self.gate.forward(hidden).float()
        scores = torch.sigmoid(logits)
        scores_for_selection = scores + self.gate.e_score_correction_bias.float()
        _, topk_ids = torch.topk(scores_for_selection, self.top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = hidden_states.view(-1, hidden_states.shape[-1])
        topk_weights, topk_ids = self._route(hidden)
        routed = self.experts.routed_forward(hidden, topk_weights, topk_ids.clone())
        shared = self.shared_expert.forward(hidden)
        out = routed + shared
        return out.view(hidden_states.shape)


class _SharedExpert(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, eps: float):
        from freetoken.layers import LinearColParallelMerged, LinearRowParallel, silu_and_mul, gelu_and_mul, gelu_tanh_and_mul

        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        act_map = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
        self.act_fn = act_map.get(hidden_act, silu_and_mul)
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.down_proj.forward(self.act_fn(self.gate_up_proj.forward(x)))


__all__ = ["LagunaGate", "LagunaSparseMoeBlock"]
