from __future__ import annotations

from dataclasses import replace

import torch

from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.models.blocks import GatedMLP
from freetoken.models.config import ModelConfig


class FlintMoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.expert_bias = torch.zeros(config.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(
            config, layer_id=layer_id - config.first_k_dense_replace,
            quant_config=config.quant, prefix=f"{prefix}.experts",
        )
        shared_config = replace(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert = GatedMLP(shared_config, quant_config=config.quant, prefix=f"{prefix}.shared_expert")

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The reference projects in model dtype, then computes sigmoid in fp32.
        return fused_topk_sigmoid(
            self.gate.forward(x).float(), self.gate.expert_bias,
            self.top_k, self.renormalize,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, ids = self._route(x)
        # Some routed kernels overwrite their input; the shared branch needs it first.
        shared = self.shared_expert.forward(x)
        return self.experts.routed_forward(x, weights, ids) + shared
