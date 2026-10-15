"""LFM2-MoE feed-forward: the dense gated MLP of the leading layers and the sigmoid-routed MoE.

Router (HF ``Lfm2MoeSparseMoeBlock.route_tokens_to_experts``, transcribed):

    scores = sigmoid(gate(x))                     # model dtype, like HF
    selected = topk(scores + expert_bias)         # the fp32 bias shifts SELECTION only
    weights = scores[selected] * routed_scaling_factor, renormalized over the top-k

``expert_bias`` never enters the returned weights; the renormalization eps is 1e-6. The
transcription is needed because the engine's fused routers are softmax-only (``fused_topk``)
or softmax-over-top-k (``gpt_oss_fused_routing``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router renormalization epsilon (HF Lfm2MoeSparseMoeBlock; NOT glm4_moe's 1e-20).
_NORM_TOPK_EPS = 1e-6


class Lfm2DenseMLP(BaseOP):
    """SwiGLU MLP of the leading ``num_dense_layers`` layers: ``w2(silu(w1(x)) * w3(x))``.

    Separate checkpoint projections, no fusion (same reasoning as the attention projections).
    """

    def __init__(self, config: "ModelConfig", *, prefix: str = "") -> None:
        self.w1 = LinearReplicated(
            config.hidden_size, config.intermediate_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.w1",
        )
        self.w3 = LinearReplicated(
            config.hidden_size, config.intermediate_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.w3",
        )
        self.w2 = LinearReplicated(
            config.intermediate_size, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(F.silu(self.w1.forward(x)) * self.w3.forward(x))


class Lfm2MoeSparseBlock(BaseOP):
    """Sigmoid-routed experts with a selection-only bias; no shared expert."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "") -> None:
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.use_expert_bias = config.use_expert_bias
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # a non-underscore tensor attribute is a persistent state-dict entry (the checkpoint's
        # F32 feed_forward.expert_bias); fp32 because the rounding would change the top-k
        self.expert_bias = (
            torch.empty(config.num_experts, dtype=torch.float32) if config.use_expert_bias else None
        )
        # The expert banks are indexed by MoE-layer id: the leading dense layers own no experts.
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )

    def _route(self, hidden_states: torch.Tensor):
        logits = self.gate.forward(hidden_states)
        scores = logits.sigmoid()
        scores_for_routing = scores + self.expert_bias if self.use_expert_bias else scores
        _, topk_ids = torch.topk(scores_for_routing, self.top_k, dim=-1)
        topk_weights = torch.gather(scores, 1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + _NORM_TOPK_EPS)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # No shared branch to protect: the routed kernel may overwrite its input.
        topk_weights, topk_ids = self._route(hidden_states)
        return self.experts.routed_forward(hidden_states, topk_weights, topk_ids)


__all__ = ["Lfm2DenseMLP", "Lfm2MoeSparseBlock"]
