"""MXFP8 routed experts (W8A16): fp8-e4m3 weights with one e8m0 code per 32 inputs of a row, bf16 activations.

``weight [S, N, K]`` e4m3 and ``scale [S, N, K // 32]`` uint8 codes, ``w = e4m3 * 2**(code - 127)``; S is the expert or cache slot axis.
Both kernels read the routed rows in place and dequantize in the K loop. ``e4m3 * 2**k`` is exact in bf16, so only the accumulation order differs from a bf16 copy of the experts.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language import target_info
from triton.runtime.jit import constexpr_function

from freetoken.kernel import backend
from freetoken.kernel.triton.e4m3_compat import FORCE_EMU, e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _e8m0_f32(codes):
    """e8m0 codes -> 2**(code - 127) as fp32; code 0 gives 0 and 255 gives inf."""
    return (codes.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@constexpr_function
def _bf16x2_cvt_native():
    # ptxas takes cvt.rn.bf16x2.e4m3x2 from sm_100 on
    return not FORCE_EMU and target_info.cuda_capability_geq(10, 0)


@triton.jit
def _e4m3_scaled_bf16(w, codes):
    """e4m3 weights times 2**(code - 127) as bf16, two values per instruction."""
    # code << 7 is the bf16 bit pattern of the scale
    # an output register may alias an input, so the asm writes its outputs last
    r = tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; .reg .b32 t, u; mov.b32 {lo, hi}, $2; cvt.rn.bf16x2.e4m3x2 t, lo; cvt.rn.bf16x2.e4m3x2 u, hi; mul.rn.bf16x2 t, t, $3; mul.rn.bf16x2 u, u, $4; mov.b32 $0, t; mov.b32 $1, u; }",
        "=r,=r,r,r,r", [w, codes.to(tl.uint16) << 7], dtype=tl.uint16, is_pure=True, pack=4)
    return r.to(tl.bfloat16, bitcast=True)


@triton.jit
def _decode_mxfp8_moe_kernel(
    a_ptr,        # [M, K] or [M * TOP_K, K] activations
    w_ptr,        # [S, N, K] e4m3 (uint8 view before sm_89)
    s_ptr,        # [S, N, K // 32] uint8
    c_ptr,        # [M, TOP_K, N] output
    topk_weights_ptr, topk_ids_ptr,
    total_routes, N, K,
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_se, stride_sn, stride_sk,
    stride_cm, stride_ck, stride_cn,
    stride_twm, stride_twk, stride_tidm, stride_tidk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TOP_K: tl.constexpr, A_ROW_IS_ROUTE: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    KG: tl.constexpr = BLOCK_K // 32
    route_id = tl.program_id(0)
    n_block = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_g = tl.arange(0, KG)

    slot = tl.load(topk_ids_ptr + token_id * stride_tidm + route_k * stride_tidk).to(tl.int64)
    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    if slot < 0:
        # Both GEMMs must clear inactive routes before activation/reduction, including graph replay.
        tl.store(c_ptrs, 0, mask=(route_id < total_routes) & n_mask)
        return
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am
    w_slot = w_ptr + slot * stride_we
    s_slot = s_ptr + slot * stride_se

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kb in range(tl.cdiv(K, BLOCK_K)):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        if e4m3_native_cx():
            w = tl.load(
                w_slot + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
        else:
            w = e4m3_u8_to_f32(tl.load(
                w_slot + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :], other=0,
            ))
        g = kb * KG + offs_g
        codes = tl.load(
            s_slot + offs_n[:, None] * stride_sn + g[None, :] * stride_sk,
            mask=n_mask[:, None] & (g[None, :] * 32 < K), other=0,
        )
        # load the activation row at the weight tile shape so it gets the weight tile layout (no shared-memory layout conversion)
        a_ptrs = tl.broadcast_to(a_base + offs_k[None, :] * stride_ak, (BLOCK_N, BLOCK_K))
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
        prod = tl.reshape(w * a, (BLOCK_N, KG, 32))
        acc += tl.sum(tl.sum(prod, axis=2) * _e8m0_f32(codes), axis=1)

    if MUL_ROUTED_WEIGHT:
        acc *= tl.load(topk_weights_ptr + token_id * stride_twm + route_k * stride_twk)
    tl.store(c_ptrs, acc.to(compute_type), mask=(route_id < total_routes) & n_mask)


def _decode_gemm(a, w, s, c, topk_weights, topk_ids, mul_routed_weight, a_row_is_route):
    M, top_k = topk_ids.shape
    N, K = w.shape[1], w.shape[2]
    total_routes = M * top_k
    w = e4m3_kernel_view(w)
    BLOCK_N = 16
    grid = (total_routes, triton.cdiv(N, BLOCK_N))
    _decode_mxfp8_moe_kernel[grid](
        a, w, s, c, topk_weights, topk_ids, total_routes, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1), topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=128, TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route, MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_TL.get(c.dtype, tl.bfloat16), num_warps=4,
    )


def fused_experts_decode_mxfp8(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, activation="silu", act_alpha=1.0, act_limit=float("inf"),
) -> torch.Tensor:
    """Decode: one GEMV per route. ``topk_ids`` index rows of the banks (expert id or cache slot); negative slots produce zero."""
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.layers import gated_act_and_mul

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _decode_gemm(hidden_states, gate_up, gate_up_scale, ic1, topk_weights, topk_ids, False, False)
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _decode_gemm(ic2, down, down_scale, ic3, topk_weights, topk_ids, True, True)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


@triton.jit
def _prefill_mxfp8_moe_kernel(
    a_ptr, w_ptr, s_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_se, stride_sn, stride_sk,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    # out^T = W @ X^T: the dequantized weights are the MMA operand that comes from registers, and a 16-token block still fills the MMA.
    KG: tl.constexpr = BLOCK_SIZE_K // 32
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_g = tl.arange(0, KG)
    xt_ptrs = a_ptr + (offs_token[None, :] // top_k * stride_am + offs_k[:, None] * stride_ak)
    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    w_ptrs = w_ptr + slot * stride_we + offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk
    s_ptrs = s_ptr + slot * stride_se + offs_bn[:, None] * stride_sn + offs_g[None, :] * stride_sk

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = kb * BLOCK_SIZE_K + offs_k < K
        g_mask = ((kb * KG + offs_g) * 32 < K)[None, :]
        if _bf16x2_cvt_native():
            codes = tl.broadcast_to(tl.load(s_ptrs, mask=g_mask, other=0)[:, :, None], (BLOCK_SIZE_N, KG, 32))
            w = _e4m3_scaled_bf16(tl.load(w_ptrs, mask=k_mask[None, :], other=0.0), tl.reshape(codes, (BLOCK_SIZE_N, BLOCK_SIZE_K)))
        else:
            if e4m3_native_cx():
                w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=k_mask[None, :], other=0))
            codes = tl.load(s_ptrs, mask=g_mask, other=0)
            w = tl.reshape(tl.reshape(w, (BLOCK_SIZE_N, KG, 32)) * _e8m0_f32(codes)[:, :, None], (BLOCK_SIZE_N, BLOCK_SIZE_K))
        xt = tl.load(xt_ptrs, mask=k_mask[:, None] & token_mask[None, :], other=0.0)
        acc = tl.dot(w.to(xt.dtype), xt, acc)
        xt_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk
        s_ptrs += KG * stride_sk

    if MUL_ROUTED_WEIGHT:
        mw = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0.0)
        acc = acc * mw[None, :]
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
    tl.store(c_ptrs, acc.to(compute_type), mask=token_mask[None, :] & (offs_cn[:, None] < N))


def prefill_config(M: int, top_k: int, num_experts: int) -> dict:
    # moe_align_block_size pads each expert's token count to a multiple of BLOCK_SIZE_M, so the block follows the tokens per expert
    per_expert = M * top_k / num_experts
    block_m = 64 if per_expert >= 80 else 32 if per_expert >= 24 else 16
    # on sm_10x the 128-token tile spills registers in the routed-weight GEMM
    if per_expert >= 384 and backend.device_capability()[0] != 10:
        block_m = 128
    # the 64-token and 128-token tiles spill registers unless the K step shrinks with them
    block_k = {64: 64, 128: 32}.get(block_m, 128)
    return dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=128, BLOCK_SIZE_K=block_k, GROUP_SIZE_M=8, num_warps=4, num_stages=3)


def _prefill_gemm(a, w, s, c, tw, sorted_ids, expert_ids, ntpp, num_valid, kernel_top_k, mul_routed_weight, cfg):
    N, K = w.shape[1], w.shape[2]
    EM = sorted_ids.shape[0]
    w = e4m3_kernel_view(w)
    grid = (triton.cdiv(EM, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    _prefill_mxfp8_moe_kernel[grid](
        a, w, s, c, tw, sorted_ids, expert_ids, ntpp, N, K, EM, num_valid,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        c.stride(1), c.stride(2), tw.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=kernel_top_k,
        compute_type=_TL.get(c.dtype, tl.bfloat16), **cfg,
    )


def fused_experts_mxfp8(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, num_experts, activation="silu", act_alpha=1.0, act_limit=float("inf"),
) -> torch.Tensor:
    """Prefill: grouped GEMM over the materialized layer banks; ``topk_ids`` are expert ids in [0, num_experts)."""
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.layers import gated_act_and_mul
    from freetoken.moe.fused import moe_align_block_size

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = prefill_config(M, top_k, num_experts)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(hidden_states, gate_up, gate_up_scale, ic1, tw, sorted_ids, expert_ids, ntpp, num_valid, top_k, False, cfg)
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(ic2, down, down_scale, ic3, tw, sorted_ids, expert_ids, ntpp, num_valid, 1, True, cfg)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = ["fused_experts_decode_mxfp8", "fused_experts_mxfp8"]
