"""SwiGLU MLP for GLM-5.2's leading dense layers and per-layer shared experts.

Every projection is built from the QuantConfig, so it serves whatever precision the checkpoint stores (bf16 under NVIDIA's NVFP4 recipe, which only quantizes the routed experts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from freetoken.layers import BaseOP
from freetoken.utils import nvtx_annotate

from freetoken.layers import LinearReplicated

if TYPE_CHECKING:
    import torch


class GlmDsaGatedMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int, *, quant_config=None, prefix: str = ""):
        self.gate_proj = LinearReplicated(hidden_size, intermediate_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = LinearReplicated(hidden_size, intermediate_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = LinearReplicated(intermediate_size, hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        return self.down_proj.forward(F.silu(gate) * up)


__all__ = ["GlmDsaGatedMLP"]
