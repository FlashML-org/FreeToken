from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated


class Glm5NextMLP(BaseOP):
    def __init__(
        self, hidden_size: int, intermediate_size: int, limit: float | None
    ) -> None:
        self.gate_proj = LinearReplicated(
            hidden_size, intermediate_size, has_bias=False
        )
        self.up_proj = LinearReplicated(hidden_size, intermediate_size, has_bias=False)
        self.down_proj = LinearReplicated(
            intermediate_size, hidden_size, has_bias=False
        )
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.down_proj.forward(F.silu(gate) * up)


__all__ = ["Glm5NextMLP"]
