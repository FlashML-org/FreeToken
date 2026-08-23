"""Dense and shared SiTU MLPs for Kimi-K3."""

from __future__ import annotations

import torch

from freetoken.kernel.triton.activation import situ_and_mul
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearRowParallel
from freetoken.utils import nvtx_annotate


class KimiMLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        beta: float = 4.0,
        linear_beta: float = 25.0,
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False
        )
        self.beta = beta
        self.linear_beta = linear_beta

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        return self.down_proj.forward(
            situ_and_mul(gate_up, beta=self.beta, linear_beta=self.linear_beta)
        )


__all__ = ["KimiMLP"]
