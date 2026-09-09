from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch

    from freetoken.layers.quantization import QuantConfig


class GlmGatedMLP(BaseOP):
    """SwiGLU MLP for the leading dense layers (``intermediate_size``) and each MoE layer's
    always-on shared expert (``moe_intermediate_size``); the projections take whatever the
    checkpoint stores for their prefix (NVFP4 in the GLM-4 releases)."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, *, quant_config: QuantConfig | None = None, prefix: str = ""
    ):
        self.gate_proj = LinearReplicated(hidden_size, intermediate_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = LinearReplicated(hidden_size, intermediate_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = LinearReplicated(intermediate_size, hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        return self.down_proj.forward(F.silu(gate) * up)


__all__ = ["GlmGatedMLP"]
