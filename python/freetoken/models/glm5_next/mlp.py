"""Clamped-SwiGLU MLP for GLM-5.3-Flash's leading dense layers and shared experts.

Same shape as glm_moe_dsa's GlmDsaGatedMLP (bf16 in the NVFP4 checkpoint;
optional W8A16 fp8-at-load via ``ModelConfig.dense_quant``), but the activation
is the GLM-5.3 clamped SwiGLU (``swiglu_limit``):
``clamp(gate, max=L) * sigmoid(gate_clamped) * clamp(up, +-L)``.
"""

from __future__ import annotations


import torch
from freetoken.layers import BaseOP, swiglu_clamp_and_mul
from freetoken.utils import nvtx_annotate

from .attention import _make_proj


class Glm5NextGatedMLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: str = "none",
        swiglu_limit: float | None = None,
    ):
        self.gate_proj = _make_proj(quant, hidden_size, intermediate_size)
        self.up_proj = _make_proj(quant, hidden_size, intermediate_size)
        self.down_proj = _make_proj(quant, intermediate_size, hidden_size)
        self.swiglu_limit = swiglu_limit

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        if self.swiglu_limit is None:
            import torch.nn.functional as F

            return self.down_proj.forward(F.silu(gate) * up)
        gated = swiglu_clamp_and_mul(
            torch.cat([gate, up], dim=-1), alpha=1.0, limit=self.swiglu_limit
        )
        return self.down_proj.forward(gated)


__all__ = ["Glm5NextGatedMLP"]
