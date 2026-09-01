"""Clamped-SwiGLU MLP for GLM-5.3's dense layers and shared experts.

HF reference (Glm5NextTextMLP / shared experts): ``silu(clamp(gate, max=limit)) *
clamp(up, -limit, limit)``. Serves through the dsv4 fused kernel, which is bit-exact
to that formula (verified). Weight names match GlmDsaGatedMLP (gate/up/down_proj).
"""

from __future__ import annotations

import torch
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.models.glm_moe_dsa.mlp import GlmDsaGatedMLP
from freetoken.utils import nvtx_annotate


class Glm5ClampedMLP(GlmDsaGatedMLP):
    def __init__(self, hidden_size: int, intermediate_size: int, quant: str = "none",
                 limit: float = 10.0):
        super().__init__(hidden_size, intermediate_size, quant=quant)
        self.__dict__["_limit"] = float(limit)  # plain attr, excluded from state_dict walk

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        return self.down_proj.forward(fused_swiglu(gate, up, self._limit, x.dtype))


__all__ = ["Glm5ClampedMLP"]
