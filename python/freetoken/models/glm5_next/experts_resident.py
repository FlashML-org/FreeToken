"""GPU-resident NVFP4 routed experts for GLM-5.3's hotness-driven VRAM/host split.

A resident layer keeps its FULL packed expert banks on the GPU (no host pin, no
offload cache slots, no PCIe on miss) and computes through the same inline-dequant
Triton kernels the offload path uses for materialized full-layer views -- raw expert
ids, position == expert id. Which layers are resident is chosen by measured fetch
hotness (FREETOKEN_GLM5_RESIDENT_LAYERS); the host banks for these layers are never
built (weight.py's spec skips them), which is what makes the 45-layer model fit in
host RAM.
"""

from __future__ import annotations

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP
from freetoken.models.config import ModelConfig

FP8 = torch.float8_e4m3fn


class ResidentNvfp4Experts(BaseOP):
    """Same ``routed_forward`` contract as OffloadMoELayer, minus the cache."""

    def __init__(self, config: ModelConfig, activation: str, layer_id: int = -1):
        self.layer_id = layer_id
        e = config.num_experts
        h = config.hidden_size
        i = config.moe_intermediate_size
        self.num_experts = e
        self.activation = activation
        self.apply_router_weight_on_input = False
        self.hidden_act_alpha = config.hidden_act_alpha
        self.swiglu_limit = config.swiglu_limit
        # The six native ModelOpt banks (position == expert id), loaded by weight.py.
        self.gate_up_packed = torch.empty(e, 2 * i, h // 2, dtype=torch.uint8)
        self.gate_up_scale = torch.empty(e, 2 * i, h // 16, dtype=FP8)
        self.gate_up_global = torch.empty(e, 2 * i, dtype=torch.float16)
        self.down_packed = torch.empty(e, h, i // 2, dtype=torch.uint8)
        self.down_scale = torch.empty(e, h, i // 16, dtype=FP8)
        self.down_global = torch.empty(e, h, dtype=torch.float16)

    def _views(self):
        return (
            self.gate_up_packed, self.gate_up_scale, self.gate_up_global,
            self.down_packed, self.down_scale, self.down_global,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        act_limit = self.swiglu_limit
        act_limit = float("inf") if act_limit is None else float(act_limit)
        if get_global_ctx().batch.is_prefill:
            from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

            return fused_experts_nvfp4(
                hidden_states, *self._views(), topk_weights, topk_ids,
                self.num_experts, self.activation,
                self.apply_router_weight_on_input, self.hidden_act_alpha, act_limit,
            )
        from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

        # Marlin-style GEMV path: CUDA-graph safe (fixed shapes, no host sync).
        return fused_experts_decode_nvfp4_marlin(
            hidden_states, *self._views(), topk_weights, topk_ids,
            self.activation, self.apply_router_weight_on_input,
            self.hidden_act_alpha, act_limit,
        )


__all__ = ["ResidentNvfp4Experts"]
