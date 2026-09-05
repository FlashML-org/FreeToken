from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import LinearRowParallel, silu_and_mul
from freetoken.layers.moe import OffloadMoELayer, make_moe_layer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.

    TP: the offload experts are sharded along the intermediate axis and the bf16 shared expert is
    row-parallel, so both produce partial sums; ``routed + gate * shared`` is linear in them and
    is reduced once (one all-reduce per MoE layer instead of two).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None) -> None:
        self._comm = DistributedCommunicator()
        self._tp_size = get_tp_info().size
        if getattr(config, "expert_quant", "none") != "fp8_block":
            super().__init__(config, layer_id=layer_id)
            return
        # Qwen3.8's block-fp8 checkpoint quantizes only the routed experts; the shared
        # expert stays bf16, so hide expert_quant from _SharedExpert's fp8 branch and
        # rebuild the routed experts with the fp8_block bank layout.
        super().__init__(replace(config, expert_quant="none"), layer_id=layer_id)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        se, ex = self.shared_expert, self.experts
        if (
            self._tp_size > 1
            and isinstance(se.down_proj, LinearRowParallel)
            and isinstance(ex, OffloadMoELayer)
        ):
            shared = F.linear(silu_and_mul(se.gate_up_proj.forward(hidden_states)), se.down_proj.weight)
            if get_global_ctx().batch.is_prefill:
                routed = ex.prefill_forward(hidden_states, router_logits)
            else:
                routed = ex.decode_forward(hidden_states, router_logits)
            out = self._comm.all_reduce(shared_gate_mul_add(routed, shared, gate))
            return out.view(num_tokens, hidden_dim)
        shared = se.forward(hidden_states)
        routed = ex.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
