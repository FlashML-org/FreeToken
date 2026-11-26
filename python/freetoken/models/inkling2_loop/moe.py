"""Inkling 2 sparse MoE: fp32 sigmoid router with a selection bias, top-k weights renormalized and scaled by
``route_scale``, no shared expert, softcapped SwiGLU experts behind FreeToken's MoE layer (resident or offload)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, make_moe_layer
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def route(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, top_k: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_weights fp32, topk_ids int32)``: sigmoid scores pick by ``score + bias``, weigh by ``score``."""
    scores = torch.sigmoid(F.linear(x.float(), weight))
    ids = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    w = scores.gather(1, ids)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * scale
    return w, ids.to(torch.int32)


class Inkling2MoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.inkling2_args
        self.gate_weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.gate_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self._top_k = config.num_experts_per_tok
        self._scale = args.route_scale
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            activation="swiglu_softcap",
            alpha=args.gate_softcap,
            limit=args.up_softcap,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )

    @nvtx_annotate("MoE")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, ids = route(x, self.gate_weight, self.gate_bias, self._top_k, self._scale)
        return self.experts.routed_forward(x, w, ids)


__all__ = ["Inkling2MoE", "route"]
