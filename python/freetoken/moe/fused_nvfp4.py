"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import torch

from freetoken.kernel import moe_sum_reduce_triton, nvfp4_moe
from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4
from freetoken.layers import gated_act_and_mul
from freetoken.moe.fused import moe_align_block_size


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = nvfp4_moe.prefill_config(M, top_k, num_experts, hidden_states.device.index)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    nvfp4_moe.prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    nvfp4_moe.prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_nvfp4_w4a4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    gate_up_input_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill NVFP4 MoE with NVFP4 activations (W4A4) where the GPU and the batch favour it, else :func:`fused_experts_nvfp4`.

    ``*_input_scale`` are the per-expert static activation scales of the ``num_experts`` banks; each GEMM input is quantized under their max, as vLLM does.
    """
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    inter = gate_up_packed.shape[1] // 2
    cfg = nvfp4_moe.w4a4_prefill_config(M, top_k, num_experts, H, inter, activation, hidden_states.dtype, hidden_states.device.index)
    if cfg is None:
        return fused_experts_nvfp4(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, down_packed, down_scale, down_global,
            topk_weights, topk_ids, num_experts, activation, False, act_alpha, act_limit,
        )
    dev = hidden_states.device
    g = torch.stack((gate_up_input_scale[:num_experts], down_input_scale[:num_experts])).float().amax(dim=1)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    xq, xs = quantize_nvfp4(hidden_states, g[0:1])
    q = torch.empty((num_valid, inter // 2), device=dev, dtype=torch.uint8)
    qs = torch.empty((num_valid, inter // 16), device=dev, dtype=torch.float8_e4m3fn)
    nvfp4_moe.w4a4_gate_up_gemm(
        xq, xs, g[0:1], gate_up_packed, gate_up_scale, gate_up_global, q, qs, g[1:2], sorted_ids, expert_ids, ntpp,
        num_valid, top_k, activation, act_alpha, act_limit, cfg["gate_up"],
    )
    out = torch.zeros_like(hidden_states)
    nvfp4_moe.w4a4_down_gemm(
        q, qs, g[1:2], down_packed, down_scale, down_global, out, tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        cfg["down"],
    )
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
    "fused_experts_nvfp4_w4a4",
]
