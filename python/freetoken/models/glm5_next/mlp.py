"""Clamped-SwiGLU MLP for GLM-5.3-Flash's leading dense layers, shared experts and vision tower.

Same shape as glm_moe_dsa's GlmDsaGatedMLP (every projection built from the QuantConfig), but the activation
is the GLM-5.3 clamped SwiGLU (``swiglu_limit``):
``clamp(gate, max=L) * sigmoid(gate_clamped) * clamp(up, +-L)``.
"""

from __future__ import annotations


import torch
from freetoken.layers import BaseOP, swiglu_clamp_and_mul
from freetoken.utils import nvtx_annotate

from freetoken.layers import LinearColParallelMerged, LinearReplicated, LinearRowParallel


class Glm5NextGatedMLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float | None = None,
        *,
        has_bias: bool = False,
        tensor_parallel: bool = False,
        quant_config=None,
        prefix: str = "",
    ):
        # The decoder MLPs split their intermediate dim under TP (down all-reduces); the vision
        # tower runs whole on every rank, so its projections stay replicated.
        kw = dict(has_bias=has_bias, quant_config=quant_config)
        if tensor_parallel:
            self.gate_proj = LinearColParallelMerged(hidden_size, [intermediate_size], prefix=f"{prefix}.gate_proj", **kw)
            self.up_proj = LinearColParallelMerged(hidden_size, [intermediate_size], prefix=f"{prefix}.up_proj", **kw)
            self.down_proj = LinearRowParallel(intermediate_size, hidden_size, prefix=f"{prefix}.down_proj", **kw)
        else:
            self.gate_proj = LinearReplicated(hidden_size, intermediate_size, prefix=f"{prefix}.gate_proj", **kw)
            self.up_proj = LinearReplicated(hidden_size, intermediate_size, prefix=f"{prefix}.up_proj", **kw)
            self.down_proj = LinearReplicated(intermediate_size, hidden_size, prefix=f"{prefix}.down_proj", **kw)
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
