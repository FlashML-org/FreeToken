"""Inline-dequant NVFP4 fused-MoE Triton kernels.

These kernels read the NVFP4 expert cache (packed FP4 codes + fp8 block scales +
fp16 per-row global scale) *directly* and dequantize inside the GEMM K-loop, so the
grouped MoE GEMM never materializes a BF16 copy of the experts (no separate dequant
pass, no HBM round-trip of the 4x larger BF16 weights).

Layout, for an expert weight ``W[N, K]``:
  - ``packed[slot, n, k//2]`` uint8: two FP4 codes per byte, low nibble = even k.
  - ``scale[slot, n, k//16]`` fp8-e4m3: per 16-wide block scale.
  - ``global[slot, n]`` fp16: per-output-row scale (``weight_scale_2``).
  - ``W[n, k] = E2M1[code] * scale[n, k//16] * global[n]``.

The global scale is constant along K, so it is applied once after the K-loop.

Decode (M=1) is HBM-bandwidth bound. :func:`_decode_nvfp4_marlin_kernel` is the production
decode GEMV (int32 wide loads + deferred K reduction); :func:`_decode_nvfp4_moe_kernel` is
the original LUT-gather variant kept for A/B comparison. The marlin-style wide load lifts
MiniMax-M2 decode toward the RTX 5090 read-bandwidth ceiling (~87%); the residual gap is FP4
dequant ALU, which a swizzled-layout tensor-core path (marlin / flashinfer b12x) closes but
those need sm_80-99 / CUDA>=13 respectively.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32
from freetoken.kernel.triton.nvfp4_linear import _nvfp4_pair_f32

_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


@functools.lru_cache(maxsize=None)
def _e2m1_lut(device_index: int) -> torch.Tensor:
    return torch.tensor(
        _E2M1_VALUES, dtype=torch.float32, device=torch.device("cuda", device_index)
    )


@triton.jit
def _decode_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    lut_ptr,           # [16] fp32
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,  # bytes processed per K-iter (covers 2*KB k-values)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Original LUT-gather serial-K decode GEMV. Kept as the A/B baseline; the production
    decode path is :func:`_decode_nvfp4_marlin_kernel`."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    K_BYTES = K // 2
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kb_start in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb_start * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_slot + offs_n[None, :] * stride_pn + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(
            p_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0
        ).to(tl.int32)
        lo = bytes_ & 0xF
        hi = (bytes_ >> 4) & 0xF
        b_lo = tl.load(lut_ptr + lo)
        b_hi = tl.load(lut_ptr + hi)

        sblk = byte_idx // 8
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + sblk[:, None] * stride_sblk
        s_mask = byte_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))
        b_lo = b_lo * scale
        b_hi = b_hi * scale

        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        accumulator += tl.sum(a_lo[:, None] * b_lo, axis=0)
        accumulator += tl.sum(a_hi[:, None] * b_hi, axis=0)

    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _bf16x4_to_f32(v):
    """Four bf16 packed in one int64 -> four fp32 (a bf16 is the high half of its fp32)."""
    lo = v.to(tl.int32)
    hi = (v >> 32).to(tl.int32)
    return (
        (lo << 16).to(tl.float32, bitcast=True),
        (lo & -65536).to(tl.float32, bitcast=True),
        (hi << 16).to(tl.float32, bitcast=True),
        (hi & -65536).to(tl.float32, bitcast=True),
    )


@triton.jit
def _f16x4_to_f32(v):
    lo = v.to(tl.int32)
    hi = (v >> 32).to(tl.int32)
    return (
        lo.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32),
        (lo >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32),
        hi.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32),
        (hi >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32),
    )


@triton.jit
def _decode_nvfp4_marlin_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 8] int32 (8 fp4 codes per word, nibble j -> k=8*w+j)
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,  # int32 words per tile (covers 8*KW k-values)
    UNROLL: tl.constexpr,         # tiles per loop trip
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    A_WIDE: tl.constexpr,         # activation rows are contiguous bf16/fp16, 16-byte aligned
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Marlin-style NVFP4 decode GEMV: wide int32 weight loads + deferred reduction.

    Versus :func:`_decode_nvfp4_moe_kernel` (which loads the packed codes one byte at a
    time and runs a cross-lane reduction *every* K-iter) this:

      * Loads the FP4 codes as **int32 words** (8 codes / 4 bytes per element) so the HBM
        read issues wide coalesced transactions -- the single biggest lever for the
        gate/up GEMM, whose mem-only ceiling jumps from ~65% to ~85% of peak read BW.
      * **Defers** the K reduction: the ``[BLOCK_N, BLOCK_KW]`` partial accumulates across
        *all* K-iters and is reduced once at the end, removing the per-iter ``tl.sum``
        barrier so the weight loads of successive iters pipeline.

    Every load in the K-loop is shaped so the compiler keeps the whole loop in the word
    tile's register layout (no shared-memory layout conversions): the block scales are
    loaded as a half tile and interleaved in registers (one e4m3 scale per 16 k-values
    == 2 words), the e2m1 codes are decoded with the fp16 bit trick of
    :func:`_nvfp4_pair_f32` (2^14 folded into the scale), and with ``A_WIDE`` the 8
    activations of a word are read as two int64 and unpacked by shifts. Mirrors the fast
    kernel's route/tile mapping and epilogue so it is a drop-in decode GEMV (CUDA-graph
    safe: fixed shapes, no host sync)."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    offs_kh = tl.arange(0, BLOCK_SIZE_KW // 2)
    K_WORDS = K // 8
    partial = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_KW), dtype=tl.float32)

    p_base = packed_ptr + slot * stride_pe + offs_n[:, None] * stride_pn
    s_base = scale_ptr + slot * stride_se + offs_n[:, None] * stride_sn
    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW), UNROLL):
        for u in tl.static_range(UNROLL):
            widx = (kw_start + u) * BLOCK_SIZE_KW + offs_kw
            w_mask = widx < K_WORDS
            word = tl.load(
                p_base + widx[None, :] * stride_pkw,
                mask=n_mask[:, None] & w_mask[None, :], other=0,
            )
            hidx = (kw_start + u) * (BLOCK_SIZE_KW // 2) + offs_kh
            s_ptrs = s_base + hidx[None, :] * stride_sblk
            s_mask = n_mask[:, None] & (hidx[None, :] < K_WORDS // 2)
            if e4m3_native_cx():
                s_half = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
            else:
                s_half = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))
            s_half = s_half * 16384.0
            scale = tl.interleave(s_half, s_half)

            if A_WIDE:
                a64 = a_base.to(tl.pointer_type(tl.int64), bitcast=True)
                v = tl.load(
                    a64 + widx[:, None] * 2 + tl.arange(0, 2)[None, :],
                    mask=w_mask[:, None], other=0,
                )
                v0, v1 = tl.split(v)
                if a_ptr.dtype.element_ty == tl.bfloat16:
                    a0, a1, a2, a3 = _bf16x4_to_f32(v0)
                    a4, a5, a6, a7 = _bf16x4_to_f32(v1)
                else:
                    a0, a1, a2, a3 = _f16x4_to_f32(v0)
                    a4, a5, a6, a7 = _f16x4_to_f32(v1)
            else:
                kbase = 8 * widx
                a0 = tl.load(a_base + kbase * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                a1 = tl.load(a_base + (kbase + 1) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                a2 = tl.load(a_base + (kbase + 2) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                a3 = tl.load(a_base + (kbase + 3) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                a4 = tl.load(a_base + (kbase + 4) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                a5 = tl.load(a_base + (kbase + 5) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                a6 = tl.load(a_base + (kbase + 6) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                a7 = tl.load(a_base + (kbase + 7) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)

            acc_w = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_KW), dtype=tl.float32)
            b0, b4 = _nvfp4_pair_f32(word)
            b1, b5 = _nvfp4_pair_f32(word >> 4)
            b2, b6 = _nvfp4_pair_f32(word >> 8)
            b3, b7 = _nvfp4_pair_f32(word >> 12)
            acc_w += a0[None, :] * b0
            acc_w += a1[None, :] * b1
            acc_w += a2[None, :] * b2
            acc_w += a3[None, :] * b3
            acc_w += a4[None, :] * b4
            acc_w += a5[None, :] * b5
            acc_w += a6[None, :] * b6
            acc_w += a7[None, :] * b7
            partial += acc_w * scale

    accumulator = tl.sum(partial, axis=1)
    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _prefill_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [num_valid_tokens, N] output (flat over M*top_k)
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,    # cache slot per M-block
    num_tokens_post_padded_ptr,
    lut_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
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
    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    a_ptrs_lo = a_ptr + (offs_token[:, None] // top_k * stride_am + (2 * offs_kb)[None, :] * stride_ak)
    a_ptrs_hi = a_ptr + (offs_token[:, None] // top_k * stride_am + (2 * offs_kb + 1)[None, :] * stride_ak)

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    packed_base = packed_ptr + slot * stride_pe + offs_bn[None, :] * stride_pn
    scale_base = scale_ptr + slot * stride_se + offs_bn[None, :] * stride_sn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_BYTES = K // 2
    for kb in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_base + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(p_ptrs, mask=byte_mask[:, None], other=0).to(tl.int32)
        lo = bytes_ & 0xF
        hi = (bytes_ >> 4) & 0xF
        sblk = byte_idx // 8
        s_ptrs = scale_base + sblk[:, None] * stride_sblk
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=byte_mask[:, None], other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=byte_mask[:, None], other=0))
        b_lo = tl.load(lut_ptr + lo) * scale  # [BLOCK_KB, BLOCK_N]
        b_hi = tl.load(lut_ptr + hi) * scale

        a_lo = tl.load(a_ptrs_lo, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        a_hi = tl.load(a_ptrs_hi, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        accumulator += tl.dot(a_lo, b_lo.to(a_lo.dtype))
        accumulator += tl.dot(a_hi, b_hi.to(a_hi.dtype))

        a_ptrs_lo += BLOCK_SIZE_KB * 2 * stride_ak
        a_ptrs_hi += BLOCK_SIZE_KB * 2 * stride_ak

    g = tl.load(global_ptr + slot * stride_ge + offs_bn * stride_gn).to(tl.float32)
    accumulator = accumulator * g[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


__all__ = [
    "_decode_nvfp4_moe_kernel",
    "_decode_nvfp4_marlin_kernel",
    "_prefill_nvfp4_moe_kernel",
    "_e2m1_lut",
]
