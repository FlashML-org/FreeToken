from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class MiniMaxM2SparseMoeBlock(BaseOP):
    """MiniMax-M2 sparse MoE block: sigmoid routing with a selection bias over the
    NVFP4 expert banks in the unified offload cache (the expert GEMM backend follows
    ``cache.quant_format``; see :meth:`OffloadMoELayer.routed_forward`)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        # DeepSeek-style selection bias; not an nn.Parameter in HF (a registered buffer).
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        logits = F.linear(hidden_states.float(), self.gate.weight)
        return fused_topk_sigmoid(logits, self.e_score_correction_bias, self.top_k, self.norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        return out.view(num_tokens, hidden_dim)


__all__ = ["MiniMaxM2SparseMoeBlock"]
