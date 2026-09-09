"""SwiGLU-OAI MLP for MiniMax-M3's leading dense layers and per-layer shared experts.

The checkpoint ships these MXFP8 (fp8-e4m3 weight + uint8 e8m0 block-32
``weight_scale_inv``) and they are served native W8A16 through the checkpoint's
QuantConfig -- decode reads every shared expert each token, so this halves the
resident-MLP weight traffic and frees the same VRAM for expert-cache slots.

gate/up are stored merged (``gate_up_proj``, [gate; up] halves -- the loader
concatenates the two checkpoint projections output-wise; MXFP8 scales are per-output-
row so the fusion is exact), which feeds the same uninterleaved ``swigluoai_and_mul``
kernel the NVFP4 expert path uses:
``clamp(gate, max=limit) * sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.layers import BaseOP, LinearReplicated, swigluoai_and_mul
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class MiniMaxM3MLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        alpha: float,
        limit: float,
        quant_config=None,
        prefix: str = "",
    ):
        self.gate_up_proj = LinearReplicated(
            hidden_size, 2 * intermediate_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearReplicated(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.down_proj",
        )
        self._alpha = alpha
        self._limit = limit

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = swigluoai_and_mul(gate_up, alpha=self._alpha, limit=self._limit)
        del gate_up
        return self.down_proj.forward(y)


__all__ = ["MiniMaxM3MLP", "make_proj"]
