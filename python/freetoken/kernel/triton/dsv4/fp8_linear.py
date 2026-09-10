"""Block-scaled FP8 (e4m3) linear for DeepSeek-V4, matching the reference numerics.

The reference (``inference/model.py`` ``linear`` + ``inference/kernel.py``
``act_quant``/``fp8_gemm``) quantizes the *activation* to FP8 with a per-128 block
power-of-two (ue8m0) scale, then runs an FP8xFP8 block-scaled GEMM against the FP8
weight (which carries its own 128x128 ue8m0 block scale). Both operands' scales are
applied per 128-K block to a separate FP32 accumulator. This module reproduces that:

  ``y = fp8_gemm(act_quant(x, 128, ue8m0), weight_fp8, weight_scale_e8m0)``

``act_quant`` (reference): per block ``s = 2**ceil(log2(max(|x|,1e-4)/448))`` (exact
via IEEE bit ops -> matches ``fast_round_scale``), ``x_fp8 = round_e4m3(clamp(x/s,
+-448))``, scale stored e8m0. The GEMM accumulates ``sum_k (A_fp8 @ B_fp8) * s_a * s_b``
per 128-K block in FP32.

Also provides ``act_quant_fp8_inplace`` -- the fused FP8 quant+dequant round-trip the
reference applies in-place to the window / compressor KV (``act_quant(..., 64, ...,
inplace=True)``), returning BF16.

Assumes ``K % 128 == 0`` and ``N % 128 == 0`` (true for every DeepSeek-V4 projection).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_act_dtype,
    e4m3_kernel_view,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    round_e4m3,
)

FP8 = torch.float8_e4m3fn
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


# ======================================================================================
# Activation FP8 quantization (ue8m0 power-of-two scale), matching reference act_quant.
# ======================================================================================
@triton.jit
def _log2_ceil(v):
    """Exact ceil(log2(v)) for v > 0 via IEEE-754 bit ops (matches fast_log2_ceil)."""
    bits = v.to(tl.uint32, bitcast=True)
    exp = ((bits >> 23) & 0xFF).to(tl.int32)
    man = (bits & 0x7FFFFF).to(tl.int32)
    return exp - 127 + tl.where(man != 0, 1, 0)


@triton.jit
def _act_quant_fp8_kernel(
    x_ptr, y_ptr, s_ptr, M, N,
    stride_xm, stride_xn, stride_ym, stride_yn, stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK: tl.constexpr,
):
    """Per-row, per-``BLOCK`` FP8 quant with ue8m0 (pow2) scale. ``s`` holds e8m0 codes."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK + tl.arange(0, BLOCK)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-4)
    e = _log2_ceil(amax * (1.0 / 448.0))                # [BLOCK_M]
    s = tl.exp2(e.to(tl.float32))
    y = tl.clamp(x / s[:, None], -448.0, 448.0)
    # See fp8_block_linear: triton's float8e4nv downcast double-rounds one-sided
    # (toward zero), so round onto the grid in fp32 first and let the (now exact)
    # downcast follow.
    y = round_e4m3(y)  # emu path keeps these grid values in the wrapper's bf16 buffer
    if e4m3_native_cx():
        y = y.to(tl.float8e4nv)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yn,
        y, mask=m_mask[:, None],
    )
    code = (e + 127).to(tl.uint8)
    tl.store(s_ptr + offs_m * stride_sm + pid_n * stride_sn, code, mask=m_mask)


def act_quant_fp8(x: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``act_quant`` (ue8m0): returns ``(x_fp8 [M,K], scale_codes [M,K//block])``
    where ``scale = 2**(code-127)``. Without native fp8 the quantized values are the
    same e4m3-grid points held in bf16 (exactly representable)."""
    *lead, K = x.shape
    assert K % block == 0, (K, block)
    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    y = torch.empty((M, K), dtype=e4m3_act_dtype(), device=x.device)
    s = torch.empty((M, K // block), dtype=torch.uint8, device=x.device)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), K // block)
    _act_quant_fp8_kernel[grid](
        x2d, y, s, M, K,
        x2d.stride(0), x2d.stride(1), y.stride(0), y.stride(1), s.stride(0), s.stride(1),
        BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return y, s


@triton.jit
def _act_quant_inplace_kernel(
    x_ptr, o_ptr, M, N, stride_m, stride_n, stride_om, stride_on,
    FP8_MIN, FP8_MAX, INV_MAX, FP4: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fused quant+dequant round-trip (reference ``inplace=True``), written to ``o_ptr`` as
    the input dtype (``o_ptr==x_ptr`` for true in-place; a distinct out buffer fuses the
    copy for callers that must not clobber the input). ``FP4=False`` -> FP8 e4m3 (block 64);
    ``FP4=True`` -> FP4 e2m1 (block 32)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK + tl.arange(0, BLOCK)
    m_mask = offs_m < M
    ptrs = x_ptr + offs_m[:, None] * stride_m + offs_k[None, :] * stride_n
    x = tl.load(ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    if FP4:
        amax = tl.maximum(amax, 6.0 * (2.0 ** -126))
    else:
        amax = tl.maximum(amax, 1e-4)
    e = _log2_ceil(amax * INV_MAX)
    s = tl.exp2(e.to(tl.float32))
    q = tl.clamp(x / s[:, None], FP8_MIN, FP8_MAX)
    if FP4:
        q = _round_fp4(q)
    else:
        # No fp8 round-trip on the native path: it only ever *re*-quantized an
        # already-grid value, and triton's downcast double-rounds rather than RNE.
        q = round_e4m3(q)
    optrs = o_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    y = (q * s[:, None]).to(optrs.dtype.element_ty)
    tl.store(optrs, y, mask=m_mask[:, None])


@triton.jit
def _round_fp4(x):
    """Round to nearest float4_e2m1fn value in {0,.5,1,1.5,2,3,4,6} (signed), in FP32.

    Matches the hardware ``float4_e2m1fn`` cast: round-to-nearest, ties-to-even on the
    grid magnitudes. Even-magnitude grid points are {0, 1.0, 2.0, 4.0} (even mantissa
    bit), so the odd-magnitude midpoints (0.75, 1.75, 3.5) round UP to the even neighbor
    while the even-magnitude midpoints (0.25, 1.25, 2.5, 5.0) round toward the even one.
    Verified against the tilelang reference fp4 cast (probe: 0.75->1, 1.75->2, 3.5->4)."""
    sign = tl.where(x < 0, -1.0, 1.0)
    a = tl.abs(x)
    r = tl.where(
        a <= 0.25, 0.0,          # 0.25 tie -> 0.0 (even)
        tl.where(a < 0.75, 0.5,  # 0.75 tie -> 1.0 (even)
        tl.where(a <= 1.25, 1.0, # 1.25 tie -> 1.0 (even)
        tl.where(a < 1.75, 1.5,  # 1.75 tie -> 2.0 (even)
        tl.where(a <= 2.5, 2.0,  # 2.5 tie -> 2.0 (even)
        tl.where(a < 3.5, 3.0,   # 3.5 tie -> 4.0 (even)
        tl.where(a <= 5.0, 4.0, 6.0)))))))
    return sign * r


def act_quant_fp8_inplace(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Reference ``act_quant(x, block, ue8m0, e8m0, inplace=True)``: FP8 quant+dequant
    round-trip written back into ``x`` (BF16). Operates on the (possibly strided) view."""
    *lead, N = x.shape
    assert N % block == 0, (N, block)
    x2d = x.reshape(-1, N)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _act_quant_inplace_kernel[grid](
        x2d, x2d, M, N, x2d.stride(0), x2d.stride(1), x2d.stride(0), x2d.stride(1),
        -448.0, 448.0, 1.0 / 448.0, False, BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return x


def act_quant_fp8_roundtrip(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 quant+dequant round-trip into a fresh contiguous BF16 tensor (fuses the copy --
    for callers that must keep ``x`` intact, e.g. the MoE expert input shared with the
    gate / shared expert). Numerically identical to ``act_quant_fp8_inplace(x.clone())``."""
    *lead, N = x.shape
    assert N % block == 0, (N, block)
    x2d = x.reshape(-1, N)
    out = torch.empty_like(x2d)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _act_quant_inplace_kernel[grid](
        x2d, out, M, N, x2d.stride(0), x2d.stride(1), out.stride(0), out.stride(1),
        -448.0, 448.0, 1.0 / 448.0, False, BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return out.reshape(x.shape)


def fp4_act_quant_inplace(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Reference ``fp4_act_quant(x, block, inplace=True)``: FP4 quant+dequant round-trip
    written back into ``x`` (BF16)."""
    *lead, N = x.shape
    assert N % block == 0, (N, block)
    x2d = x.reshape(-1, N)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _act_quant_inplace_kernel[grid](
        x2d, x2d, M, N, x2d.stride(0), x2d.stride(1), x2d.stride(0), x2d.stride(1),
        -6.0, 6.0, 1.0 / 6.0, True, BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return x


# ======================================================================================
# FP8 (act) x FP8 (weight) block-scaled GEMM / GEMV.
# ======================================================================================
@triton.jit
def _fp8_act_gemm_kernel(
    a_ptr,            # [M, K] float8_e4m3fn (quantized activation)
    w_ptr,            # [N, K] float8_e4m3fn
    sa_ptr,           # [M, K//128] uint8 (e8m0 act codes)
    sb_ptr,           # [N//128, K//128] uint8 (e8m0 weight codes)
    c_ptr,            # [M, N] compute dtype
    M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk,
    stride_sam, stride_sak, stride_sbn, stride_sbk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for k in range(num_k):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        if e4m3_native_cx():
            p = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        else:
            # bf16 dot on the same e4m3 grid: operands exact in bf16, fp32 acc
            p = tl.dot(a, tl.trans(e4m3_u8_to_f32(w).to(tl.bfloat16)), out_dtype=tl.float32)
        sa_code = tl.load(sa_ptr + offs_m * stride_sam + k * stride_sak, mask=m_mask, other=0)
        sca = tl.exp2(sa_code.to(tl.float32) - 127.0)            # [BLOCK_M]
        sb_code = tl.load(sb_ptr + pid_n * stride_sbn + k * stride_sbk)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)            # scalar (one 128-N block)
        acc += p * sca[:, None] * scb
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None])


@triton.jit
def _fp8_act_gemv_splitk_kernel(
    a_ptr,            # [K] float8_e4m3fn
    sa_ptr,           # [K//128] uint8 (e8m0 act codes)
    w_ptr,            # [N, K] float8_e4m3fn
    sb_ptr,           # [N//128, K//128] uint8 (e8m0 weight codes)
    part_ptr,         # [SPLIT_K, N] fp32
    N, K,
    stride_ak, stride_wn, stride_wk, stride_sbn, stride_sbk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    sn = offs_n // 128
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, k_per, 128):
        offs_k = k_start + k0 + tl.arange(0, 128)
        a = tl.load(a_ptr + offs_k * stride_ak).to(tl.float32)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=n_mask[:, None], other=0))
        kb = (k_start + k0) // 128
        sb_code = tl.load(sb_ptr + sn * stride_sbn + kb * stride_sbk, mask=n_mask, other=0)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)
        sa_code = tl.load(sa_ptr + kb)
        sca = tl.exp2(sa_code.to(tl.float32) - 127.0)
        acc += tl.sum(w * a[None, :], axis=1) * scb * sca
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(part_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
                          stride_pk, stride_pn, BLOCK: tl.constexpr, OUT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(OUT), mask=mask)


# Swept-best decode GEMV config per (N, K) -> (BLOCK_N, SPLIT_K, num_warps).
_DECODE_FP8_CFG = {
    (1024, 4096): (16, 16, 1),   # wq_a
    (32768, 1024): (16, 1, 2),   # wq_b
    (512, 4096): (16, 16, 1),    # attn wkv
    (4096, 8192): (16, 8, 1),    # wo_b
    (2048, 4096): (16, 8, 1),    # shared w1 / w3
    (4096, 2048): (16, 8, 1),    # shared w2
    (8192, 1024): (16, 2, 4),    # indexer wq_b
}


def _decode_cfg(N: int, K: int) -> tuple[int, int, int]:
    cfg = _DECODE_FP8_CFG.get((N, K))
    if cfg is not None:
        bn, sk, nw = cfg
        return bn, max(1, min(sk, K // 128)), nw
    bn = 16
    n_tiles = triton.cdiv(N, bn)
    sk = max(1, 1536 // n_tiles)
    sk = 1 << (sk.bit_length() - 1)
    return bn, max(1, min(sk, K // 128)), 1


def _fp8_act_gemv(a_fp8: torch.Tensor, sa: torch.Tensor, weight: torch.Tensor,
                  sb: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    N, K = weight.shape
    BLOCK_N, split_k, num_warps = _decode_cfg(N, K)
    n_tiles = triton.cdiv(N, BLOCK_N)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a_fp8.device)
    _fp8_act_gemv_splitk_kernel[(n_tiles, split_k)](
        a_fp8, sa, weight, sb, part, N, K,
        a_fp8.stride(0), weight.stride(0), weight.stride(1),
        sb.stride(0), sb.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, SPLIT_K=split_k, num_warps=num_warps,
    )
    out = torch.empty(N, dtype=out_dtype, device=a_fp8.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype], num_warps=2,
    )
    return out


def block_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = act_quant(x) @ weight^T`` (reference FP8 path).

    ``x``: ``[..., K]`` bf16; ``weight``: ``[N, K]`` float8_e4m3fn; ``scale``:
    ``[N//128, K//128]`` float8_e8m0fnu (weight block scale). Activation is quantized
    to FP8 with a per-128 ue8m0 scale; the GEMM applies both scales per 128-K block.
    """
    assert weight.dtype == FP8
    *lead, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert K % 128 == 0 and N % 128 == 0, (N, K)
    compute_dtype = x.dtype if x.dtype in _TL_DTYPE else torch.bfloat16
    sb = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    sb = sb.contiguous()
    w = e4m3_kernel_view(weight)

    a_fp8, sa = act_quant_fp8(x, 128)  # [M,K] fp8, [M,K//128] e8m0 codes
    M = a_fp8.shape[0]

    if M == 1:
        out = _fp8_act_gemv(a_fp8[0], sa[0], w, sb, compute_dtype).reshape(*lead, N)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    out = torch.empty((M, N), dtype=compute_dtype, device=x.device)
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_act_gemm_kernel[grid](
        a_fp8, w, sa, sb, out,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1), w.stride(0), w.stride(1),
        sa.stride(0), sa.stride(1), sb.stride(0), sb.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        compute_type=_TL_DTYPE[compute_dtype], num_warps=4, num_stages=3,
    )
    out = out.reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


__all__ = ["block_fp8_linear", "act_quant_fp8", "act_quant_fp8_inplace", "fp4_act_quant_inplace"]
