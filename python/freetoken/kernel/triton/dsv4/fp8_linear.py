"""Block-scaled FP8 (e4m3) linear for DeepSeek-V4 / V4.1, matching the reference numerics.

The reference (``inference/model.py`` ``linear`` + ``inference/kernel.py``
``act_quant``/``fp8_gemm``) quantizes the *activation* to FP8 with a per-``block``
power-of-two (ue8m0) scale, then runs an FP8xFP8 block-scaled GEMM against the FP8
weight (which carries its own ``block x block`` ue8m0 scale). Both operands' scales are
applied per ``block``-K slab to a separate FP32 accumulator. This module reproduces that:

  ``y = fp8_gemm(act_quant(x, block, ue8m0), weight_fp8, weight_scale_e8m0)``

``block`` is the checkpoint's ``weight_block_size``: 128 for DeepSeek-V4, 32 for V4.1.

``act_quant`` (reference): per block ``s = 2**ceil(log2(max(|x|,1e-4)/448))`` (exact
via IEEE bit ops -> matches ``fast_round_scale``), ``x_fp8 = round_e4m3(clamp(x/s,
+-448))``, scale stored e8m0. The GEMM accumulates ``sum_k (A_fp8 @ B_fp8) * s_a * s_b``
per ``block``-K slab in FP32.

Also provides ``act_quant_fp8_inplace`` -- the fused FP8 quant+dequant round-trip the
reference applies in-place to the window / compressor KV (``act_quant(..., 64, ...,
inplace=True)``), returning BF16.

Assumes ``K % block == 0`` and ``N % block == 0`` (true for every DeepSeek-V4 / V4.1 projection).
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
    if e4m3_native_cx():
        y = y.to(tl.float8e4nv)
    else:
        y = round_e4m3(y)  # e4m3-grid values into the wrapper's bf16 buffer
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
    elif e4m3_native_cx():
        q = q.to(tl.float8e4nv).to(tl.float32)
    else:
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
    sa_ptr,           # [M, K//BLOCK_K] uint8 (e8m0 act codes)
    sb_ptr,           # [N//BLOCK_K, K//BLOCK_K] uint8 (e8m0 weight codes; square BLOCK_K blocks)
    c_ptr,            # [M, N] compute dtype
    M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk,
    stride_sam, stride_sak, stride_sbn, stride_sbk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    """``BLOCK_K`` is the weight's square block edge, so each K step maps to exactly one scale
    column. When the N tile IS one scale row (``BLOCK_N == BLOCK_K``, the DeepSeek-V4 block-128
    case) the weight scale is a scalar per step and the N masks vanish -- the original kernel;
    otherwise (block 32) the tile spans ``BLOCK_N // BLOCK_K`` scale rows and N is masked."""
    SCALAR_SCALE: tl.constexpr = BLOCK_N == BLOCK_K
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
    if SCALAR_SCALE:
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
            scb = tl.exp2(sb_code.to(tl.float32) - 127.0)            # scalar (one N block)
            acc += p * sca[:, None] * scb
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None])
    else:
        n_mask = offs_n < N  # N is a multiple of BLOCK_K, not necessarily of BLOCK_N
        sn = offs_n // BLOCK_K
        for k in range(num_k):
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            if e4m3_native_cx():
                p = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
            else:
                p = tl.dot(a, tl.trans(e4m3_u8_to_f32(w).to(tl.bfloat16)), out_dtype=tl.float32)
            sa_code = tl.load(sa_ptr + offs_m * stride_sam + k * stride_sak, mask=m_mask, other=0)
            sca = tl.exp2(sa_code.to(tl.float32) - 127.0)            # [BLOCK_M]
            sb_code = tl.load(sb_ptr + sn * stride_sbn + k * stride_sbk, mask=n_mask, other=0)
            scb = tl.exp2(sb_code.to(tl.float32) - 127.0)            # [BLOCK_N]
            acc += p * sca[:, None] * scb[None, :]
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _fp8_gemv_kernel(
    x_ptr,            # [M, G, K] activation in the compute dtype (bf16); QUANT_ACT: quantized in-kernel per SLAB
    w_ptr,            # [G * N, K] float8_e4m3fn (or its uint8 view on pre-Ada parts)
    sb_ptr,           # [G * N // SLAB, K // SLAB] uint8 (e8m0 weight codes)
    out_ptr,          # [M, G * N] OUT dtype (SPLIT_K == 1)
    part_ptr,         # [SPLIT_K, M, G * N] fp32 partials (SPLIT_K > 1)
    M, N, K,
    stride_xm, stride_xg, stride_wn, stride_wk, stride_sbn, stride_sbk, stride_om, stride_pk, stride_pm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SLAB: tl.constexpr, SPLIT_K: tl.constexpr,
    QUANT_ACT: tl.constexpr, OUT: tl.constexpr,
):
    """Small-M (decode) block-scaled fp8 GEMV/GEMM: one pass over the weight at near HBM bandwidth.

    Each program owns ``BLOCK_N`` rows and ``K // SPLIT_K`` contiguous K, walked ``BLOCK_K`` (several
    ``SLAB``-wide scale slabs) at a time: a wide tile keeps enough bytes in flight per SM to saturate
    the memory system, which a slab-at-a-time walk (512 B per load at the V4.1 slab of 32) cannot.
    Every weight tile is applied to all ``M`` (<= ``BLOCK_M``) activation rows, so a decode batch
    costs one weight read like a single token does (the tensor-core GEMM path is 3-6x slower at
    M = 2..8: too few tiles, tiny K steps). ``QUANT_ACT`` quantizes the activation here rather than
    by a separate ``act_quant`` launch: every program recomputes the ue8m0 scale and the e4m3 rounding
    of the slabs it touches (the reference formula, bit-identical to ``act_quant_fp8``) from the
    L2-resident bf16 rows, so the path is one kernel (plus a small reduce when ``SPLIT_K > 1``).
    Without it the activation stays bf16 (W8A16: the reference's bf16 einsum over a dequantized
    weight, e.g. ``wo_a``). The third grid axis is an independent group (block-diagonal weights:
    group ``g`` reads ``x[:, g]`` and rows ``[g * N, (g + 1) * N)``)."""
    tl.static_assert(BLOCK_K % SLAB == 0, "the K tile holds whole scale slabs")
    NSLAB: tl.constexpr = BLOCK_K // SLAB
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64)
    x_ptr += g * stride_xg
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    sn = (g * N + offs_n) // SLAB
    offs_n = g * N + offs_n
    rows = tl.arange(0, BLOCK_M)
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k_per, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        # --- the weight tile, dequantized per slab (read once for every activation row) ---
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=n_mask[:, None], other=0))
        kb = (k_start + k0) // SLAB + tl.arange(0, NSLAB)
        sb_code = tl.load(sb_ptr + sn[:, None] * stride_sbn + kb[None, :] * stride_sbk, mask=n_mask[:, None], other=0)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)  # [BLOCK_N, NSLAB]
        for m in tl.static_range(BLOCK_M):
            if m < M:
                xs = tl.reshape(tl.load(x_ptr + m * stride_xm + offs_k).to(tl.float32), (NSLAB, SLAB))
                if QUANT_ACT:
                    # --- reference act_quant on this row's slabs: s = 2**ceil(log2(max(|x|, 1e-4) / 448)) ---
                    amax = tl.maximum(tl.max(tl.abs(xs), axis=1), 1e-4)
                    sa = tl.exp2(_log2_ceil(amax * (1.0 / 448.0)).to(tl.float32))  # [NSLAB]
                    q = tl.clamp(xs / sa[:, None], -448.0, 448.0)
                    if e4m3_native_cx():
                        q = q.to(tl.float8e4nv).to(tl.float32)
                    else:
                        q = round_e4m3(q).to(tl.float32)
                else:
                    sa = tl.full((NSLAB,), 1.0, tl.float32)
                    q = xs
                a = tl.reshape(q, (BLOCK_K,))
                slab = tl.sum(tl.reshape(w * a[None, :], (BLOCK_N, NSLAB, SLAB)), axis=2)  # [BLOCK_N, NSLAB]
                contrib = tl.sum(slab * scb * sa[None, :], axis=1)  # [BLOCK_N]
                acc = tl.where(rows[:, None] == m, acc + contrib[None, :], acc)
    row_mask = rows < M
    if SPLIT_K == 1:
        tl.store(out_ptr + rows[:, None] * stride_om + offs_n[None, :], acc.to(OUT), mask=row_mask[:, None] & n_mask[None, :])
    else:
        tl.store(part_ptr + pid_k * stride_pk + rows[:, None] * stride_pm + offs_n[None, :], acc, mask=row_mask[:, None] & n_mask[None, :])


@triton.jit
def _splitk_reduce_kernel(part_ptr, out_ptr, M, N, SPLIT_K: tl.constexpr,
                          stride_pk, stride_pm, stride_om, BLOCK: tl.constexpr, OUT: tl.constexpr):
    m = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + m * stride_pm + offs, mask=mask, other=0.0)
    tl.store(out_ptr + m * stride_om + offs, acc.to(OUT), mask=mask)


_GEMV_BLOCK_N = 32
_GEMV_NARROW_BLOCK_N = 8
def _gemv_cfg(N: int, K: int, slab: int, sm_count: int) -> tuple[int, int, int, int]:
    """Choose whole scale tiles and enough independent programs to cover the device."""
    block_k = next((bk for bk in (512, 256, 128) if bk >= slab and K % bk == 0), slab)
    if triton.cdiv(N, _GEMV_BLOCK_N) >= sm_count:
        return _GEMV_BLOCK_N, block_k, 1, 4
    n_tiles = triton.cdiv(N, _GEMV_NARROW_BLOCK_N)
    split = 1
    while n_tiles * split < sm_count and (K // block_k) % (split * 2) == 0:
        split *= 2
    return _GEMV_NARROW_BLOCK_N, block_k, split, 4


GEMV_MAX_M = 4  # bound the unrolled row loop's register footprint


def _fp8_gemv(
    x: torch.Tensor, weight: torch.Tensor, sb: torch.Tensor, out_dtype: torch.dtype, block: int, *, groups: int = 1, quant_act: bool = True,
) -> torch.Tensor:
    """``x [M, groups, K]`` bf16 x block-diagonal ``weight [groups * N, K]`` fp8 -> ``[M, groups * N]``
    in ``out_dtype`` (``M <= GEMV_MAX_M``). ``quant_act`` applies the reference fp8 activation quant
    (W8A8); off, the activation stays bf16 (W8A16). ``groups == 1`` is the plain GEMV."""
    GN, K = weight.shape
    N = GN // groups
    x = x.reshape(-1, groups, K)
    M = x.shape[0]
    assert 1 <= M <= GEMV_MAX_M, M
    block_m = triton.next_power_of_2(M)
    block_n, block_k, split_k, num_warps = _gemv_cfg(GN, K, block, torch.cuda.get_device_properties(x.device).multi_processor_count)
    out = torch.empty((M, GN), dtype=out_dtype, device=x.device)
    part = out if split_k == 1 else torch.empty((split_k, M, GN), dtype=torch.float32, device=x.device)
    _fp8_gemv_kernel[(triton.cdiv(N, block_n), split_k, groups)](
        x, weight, sb, out, part, M, N, K,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), sb.stride(0), sb.stride(1), out.stride(0),
        part.stride(0) if split_k > 1 else 0, part.stride(1) if split_k > 1 else 0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SLAB=block, SPLIT_K=split_k, QUANT_ACT=quant_act, OUT=_TL_DTYPE[out_dtype],
        num_warps=num_warps,
    )
    if split_k > 1:
        _splitk_reduce_kernel[(triton.cdiv(GN, 256), M)](
            part, out, M, GN, split_k, part.stride(0), part.stride(1), out.stride(0),
            BLOCK=256, OUT=_TL_DTYPE[out_dtype], num_warps=2,
        )
    return out


def grouped_w8a16_gemv(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, *, block: int) -> torch.Tensor:
    """Decode form of the reference's grouped bf16 einsum ``tgd,grd->tgr`` over a block-diagonal fp8
    weight: ``x [t, G, d]`` bf16 (``t <= GEMV_MAX_M``), ``weight [G * r, d]`` fp8 with ``[G * r //
    block, d // block]`` e8m0 ``scale`` -> ``[t, G * r]`` bf16. The weight is dequantized in-kernel
    (exact in bf16: e4m3 x 2^k), the activation is not quantized, so this is the einsum's math with a
    different fp32 summation order."""
    assert x.ndim == 3 and weight.dtype == FP8
    G = x.shape[1]
    sb = (scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale).contiguous()
    return _fp8_gemv(x.contiguous(), e4m3_kernel_view(weight), sb, x.dtype, block, groups=G, quant_act=False)


@triton.jit
def _dequant_block_fp8_kernel(w_ptr, sb_ptr, out_ptr, N, K, stride_wn, stride_sbn, stride_on, SLAB: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n < N)[:, None] & (offs_k < K)[None, :]
    if e4m3_native_cx():
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    else:
        w = e4m3_u8_to_f32(tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=mask, other=0))
    code = tl.load(sb_ptr + (offs_n // SLAB)[:, None] * stride_sbn + (offs_k // SLAB)[None, :], mask=mask, other=0)
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :], (w * tl.exp2(code.to(tl.float32) - 127.0)).to(tl.bfloat16), mask=mask)


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor, *, block: int) -> torch.Tensor:
    """``[N, K]`` block-scaled e4m3 -> bf16 (exact), one pass; the prefill-side companion of
    :func:`grouped_w8a16_gemv` for weights the reference consumes through a bf16 einsum."""
    N, K = weight.shape
    sb = (scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale).contiguous()
    out = torch.empty((N, K), dtype=torch.bfloat16, device=weight.device)
    _dequant_block_fp8_kernel[(triton.cdiv(N, 64), triton.cdiv(K, 256))](
        e4m3_kernel_view(weight), sb, out, N, K, weight.stride(0), sb.stride(0), out.stride(0),
        SLAB=block, BLOCK_N=64, BLOCK_K=256, num_warps=4,
    )
    return out


def block_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block: int = 128,
) -> torch.Tensor:
    """``y = act_quant(x) @ weight^T`` (reference FP8 path).

    ``x``: ``[..., K]`` bf16; ``weight``: ``[N, K]`` float8_e4m3fn; ``scale``:
    ``[N//block, K//block]`` float8_e8m0fnu (weight block scale). Activation is quantized
    to FP8 with a per-``block`` ue8m0 scale; the GEMM applies both scales per ``block``-K slab.
    """
    assert weight.dtype == FP8
    *lead, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert K % block == 0 and N % block == 0, (N, K, block)
    assert tuple(scale.shape) == (N // block, K // block), (tuple(scale.shape), N, K, block)
    compute_dtype = x.dtype if x.dtype in _TL_DTYPE else torch.bfloat16
    sb = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    sb = sb.contiguous()
    w = e4m3_kernel_view(weight)

    x2d = x.reshape(-1, K)
    if x2d.shape[0] == 1:
        # decode: one kernel quantizes the activation rows and streams the weight once
        out = _fp8_gemv(x2d.contiguous(), w, sb, compute_dtype, block).reshape(*lead, N)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    a_fp8, sa = act_quant_fp8(x, block)  # [M,K] fp8, [M,K//block] e8m0 codes
    M = a_fp8.shape[0]

    out = torch.empty((M, N), dtype=compute_dtype, device=x.device)
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = block
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


__all__ = ["block_fp8_linear", "act_quant_fp8", "act_quant_fp8_inplace", "fp4_act_quant_inplace", "grouped_w8a16_gemv", "dequant_block_fp8", "GEMV_MAX_M"]
