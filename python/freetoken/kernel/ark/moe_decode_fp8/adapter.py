"""Drop-in for freetoken.moe.fused_fp8_block.fused_experts_decode_fp8_block."""

import torch
from freetoken.kernel.ark.common.trace import record
from freetoken.kernel.ark.common.policy import allowed, key
from freetoken.kernel.ark.common.contracts import moe_supported, scales_supported, plain_silu
from freetoken.kernel.ark.common.native import load


def fused_experts_decode_fp8_block(
    hidden_states,
    gate_up,
    gate_up_scale,
    down,
    down_scale,
    topk_weights,
    topk_ids,
    activation="silu",
    apply_router_weight_on_input=False,
    act_alpha=1.0,
    act_limit=float("inf"),
):
    if not (
        hidden_states.ndim == 2
        and gate_up.ndim == 3
        and down.ndim == 3
        and topk_ids.ndim == 2
        and allowed(
            "moe_decode_fp8", key(hidden_states.shape[0], hidden_states.shape[1], down.shape[2], gate_up.shape[0], topk_ids.shape[1])
        )
        and plain_silu(activation, apply_router_weight_on_input, act_alpha, act_limit)
        and moe_supported(hidden_states, gate_up, down, topk_weights, topk_ids)
        and scales_supported(gate_up_scale, down_scale, gate_up, down)
    ):
        record(
            "moe_decode_fp8",
            key(hidden_states.shape[0], hidden_states.shape[1], down.shape[2], gate_up.shape[0], topk_ids.shape[1]),
            "freetoken",
        )
        from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block as baseline

        return baseline(
            hidden_states,
            gate_up,
            gate_up_scale,
            down,
            down_scale,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            act_alpha,
            act_limit,
        )
    record("moe_decode_fp8", key(hidden_states.shape[0], hidden_states.shape[1], down.shape[2], gate_up.shape[0], topk_ids.shape[1]), "ark")
    load()
    return torch.ops.freetoken_ark.moe_decode(hidden_states, gate_up, down, topk_weights, topk_ids, gate_up_scale, down_scale)
