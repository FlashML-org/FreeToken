"""Clamped-SwiGLU MLP for GLM-5.3-Flash's leading dense layers, shared experts and vision tower.

Same shape as glm_moe_dsa's GlmDsaGatedMLP (every projection built from the QuantConfig), but the activation
is the GLM-5.3 clamped SwiGLU (``swiglu_limit``):
``clamp(gate, max=L) * sigmoid(gate_clamped) * clamp(up, +-L)``.

With ``tp`` > 1 the decoder's MLPs split column/row (multi.py): each rank holds ``intermediate_size / tp`` of the gate
and up rows and the matching down columns, and the partial outputs sum. The vision tower keeps ``tp`` 1.
"""

from __future__ import annotations


import torch
from freetoken.distributed import DistributedCommunicator
from freetoken.layers import BaseOP, swiglu_clamp_and_mul
from freetoken.utils import nvtx_annotate

from freetoken.layers import LinearReplicated


class Glm5NextGatedMLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float | None = None,
        *,
        has_bias: bool = False,
        tp: int = 1,
        quant_config=None,
        prefix: str = "",
    ):
        assert not (has_bias and tp > 1), "a split MLP would add the down bias once per rank"
        inter = intermediate_size // tp
        self.gate_proj = LinearReplicated(hidden_size, inter, has_bias=has_bias, quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = LinearReplicated(hidden_size, inter, has_bias=has_bias, quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = LinearReplicated(inter, hidden_size, has_bias=has_bias, quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.swiglu_limit = swiglu_limit
        self.tp = tp
        self._comm = DistributedCommunicator()

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        """``reduce=False`` returns this rank's partial sum under ``tp`` > 1, for a caller that folds it into a later
        all-reduce."""
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        if self.swiglu_limit is None:
            import torch.nn.functional as F

            out = self.down_proj.forward(F.silu(gate) * up)
        else:
            gated = swiglu_clamp_and_mul(
                torch.cat([gate, up], dim=-1), alpha=1.0, limit=self.swiglu_limit
            )
            out = self.down_proj.forward(gated)
        return self._comm.all_reduce(out) if reduce and self.tp > 1 else out


__all__ = ["Glm5NextGatedMLP"]
