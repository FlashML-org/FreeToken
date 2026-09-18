"""Native-NVFP4 W4A16 dense linears, shared across models (Qwen3.5/3.6, GLM-4, ...).

NVFP4 checkpoints store *dense* projections (e.g. ``shared_expert.{gate,up,down}_proj``, dense
MLP ``{gate,up,down}_proj``, ``lm_head``) as NVFP4 (4-bit ``weight`` + fp8-e4m3 per-16 block
``weight_scale`` + a per-tensor ``weight_scale_2`` global scale), exactly like the routed
experts. FreeToken used to *dequantize these to bf16 at load* and run cuBLAS bf16 GEMV -- which,
at decode, reads 4x the bytes of the resident FP4 weight and was the single largest decode
kernel (a ~1 GB bf16 ``lm_head`` GEMV alone was ~628 us/token). This module keeps the weight
NVFP4 and reads it directly in a W4A16 kernel (bf16 activation, FP4 weight), quartering that
traffic. Model packages import these layers instead of rolling their own.

Dequant is arithmetic, not a LUT gather: an E2M1 code ``c = s|e1|e0|m`` placed into fp16 bits
as ``((c & 7) << 9) | ((c & 8) << 12)`` reads back as exactly ``e2m1_value(c) * 2^-14`` --
the fp16 subnormal encoding makes the ``e == 0`` codes (0, 0.5) land on the same 2^-14 scale
as the normals, so a single power-of-two factor folded into the (fp32) block-scale product
undoes it exactly. That is ~5 integer ops per pair of codes instead of a 16-entry L1 gather
per code; the gather serialized the LSU and capped the old decode GEMV at ~36% of the weight-
bandwidth roofline (H100 lm_head), the ALU version reaches ~64%. Numerically the trick is
*bit-identical* to the old fp32-LUT path (all intermediates stay in normal fp32 range).

Dispatch (``nvfp4_dense_linear``):
  * M == 1 (decode): int32 wide-load GEMV, deferred K reduction. Split-K when ``N`` is small
    (shared expert) so the SMs stay fed; single-pass full-K for the huge ``lm_head``.
  * 1 < M <= 64 (batched decode, lm_head prefill last-token batch): tensor-core dot GEMM that
    dequants in the K loop, with the same split-K trick when ``M x N`` alone cannot fill the
    SMs. Reads only the packed weight -- the bf16 scratch path would cost ~4.5x the traffic,
    which at these M is the whole latency.
  * M > 64 (prefill): dequantize the whole weight to a bf16 scratch (memory-roof Triton
    kernel, N-chunked to bound the allocation) and run cuBLAS. In-kernel dequant GEMMs
    re-dequantize every weight tile ``M / BLOCK_M`` times and top out ~3x slower than cuBLAS
    on H100; dequant-once + cuBLAS is within ~7% of the bf16 GEMM at prefill M.

CUDA-graph safe on the decode paths: fixed shapes, no host sync.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from freetoken.layers import BaseOP
from freetoken.layers.base import _concat_prefix

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_kernel_view,
    e4m3_native_cx,
    e4m3_u8_to_f16_x128,
    e4m3_u8_to_f32,
)

FP8 = torch.float8_e4m3fn
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Escape hatch: FREETOKEN_DEBUG_DENSE_NVFP4_REF=1 swaps the triton kernels for a dequant_nvfp4 +
# torch matmul reference (numeric A-B debugging). Evaluated once; the kernels are the default.
_USE_REF = os.environ.get("FREETOKEN_DEBUG_DENSE_NVFP4_REF") == "1"

# Decode GEMV tiles (H100-tuned; the fp16-trick dequant is ALU-cheap so wider K tiles win).
# Two sets: the transposed K-major resident layout (coalesced along N; narrower K tiles win)
# vs the legacy row-major layout (coalesced along K).
_GEMV_BLOCK_N = 32
_GEMV_BLOCK_KW_T = 16
_GEMV_BLOCK_KW_ROW = 32
_GEMV_WARPS = 4
_GEMV_SPLITK_BLOCK_N_T = 32
_GEMV_SPLITK_BLOCK_N_ROW = 16
_GEMV_SPLITK_BLOCK_KW = 16
_GEMV_SPLITK_TARGET = 1024  # target total programs for the split-K grid

# Small-M dot GEMM tiles (batched decode M=2..64 + lm_head last-token prefill batch).
_GEMM_BLOCK_KW = 16
_GEMM_SPLITK_TARGET = 1024  # target total programs for the split-K grid (M > 16)
# M <= 16 (decode bsz 1-16) is pure weight streaming: the split-K grid must span the SMs
# or bandwidth sits idle, so grid sizing works in units of one *wave* = SM count x this
# kernel's resident blocks/SM. Residency is derived from the device against the kernel's
# measured footprint (118 regs/thread x 128 threads, ~36 KB smem at 3 stages): H100
# (132 SMs, 228 KB smem/SM) is register-bound at 4 blocks/SM -> wave 528; consumer parts
# with ~100 KB smem/SM (e.g. RTX 5090 / GB202, 170 SMs) are smem-bound at 2 -> wave 340.
_GEMM_BLOCK_REGS = 118 * 128
_GEMM_BLOCK_SMEM = 36 << 10
_GEMM_WAVE: dict = {}


def _gemm_wave(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    wave = _GEMM_WAVE.get(idx)
    if wave is None:
        props = torch.cuda.get_device_properties(idx)
        blocks = max(1, min(
            props.regs_per_multiprocessor // _GEMM_BLOCK_REGS,
            props.shared_memory_per_multiprocessor // _GEMM_BLOCK_SMEM,
            4,
        ))
        wave = props.multi_processor_count * blocks
        _GEMM_WAVE[idx] = wave
    return wave

# Above this M the dequant-to-scratch + cuBLAS path wins (the in-K dot GEMM re-dequants
# each weight tile M/BLOCK_M times; cuBLAS is ~4x its per-tile FLOP efficiency).
_GEMM_MAX_INKERNEL_M = 64

# bf16 scratch chunk for the dequant + cuBLAS path (bounds the transient allocation).
_SCRATCH_CHUNK_BYTES = 128 << 20

# Arrival counters for the GEMV's fused split-K reduction (one int32 per N-tile). The last
# program to finish a tile's K-slices reduces the partials in-kernel, which removes the
# separate reduce-kernel launch (a ~2us tail on every M==1 decode linear). Counters are
# allocated zeroed and every kernel resets its slot after reducing, so the buffer is
# reusable across launches and CUDA-graph replays with no per-call memset. Shared across
# layers; safe because decode linears are serialized on one stream. Cached per exact size
# so graph-captured launches keep a stable address.
_SPLITK_COUNTERS: dict = {}


def _splitk_counters(n: int, device: torch.device) -> torch.Tensor:
    key = (device, n)
    c = _SPLITK_COUNTERS.get(key)
    if c is None:
        c = torch.zeros(n, dtype=torch.int32, device=device)
        _SPLITK_COUNTERS[key] = c
    return c


@triton.jit
def _nvfp4_pair_f32(v):
    """Dequant two e2m1 codes held in an int32 at bits [3:0] and [19:16].

    Returns ``(lo, hi)`` fp32 = ``e2m1_value * 2^-14``; the caller multiplies by
    ``block_scale * 2^14`` (exact, power of two). Bit trick: magnitude bits go to
    fp16 [11:9] (exponent low bits + mantissa top bit), sign to [15]; fp16's
    subnormal encoding lines the e==0 codes up on the same 2^-14 factor.
    """
    r = ((v << 9) & 0x0E000E00) | ((v & 0x00080008) << 12)
    lo = (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    hi = (r >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return lo, hi


@triton.jit
def _split8(a_vec, BLOCK_KW: tl.constexpr):
    """Contiguous ``[BLOCK_KW * 8]`` activations -> 8 ``[BLOCK_KW]`` vectors, a_j[w] = a[8w+j].

    Register-only shuffles (reshape + split tree); avoids 8 strided gathers per word tile.
    """
    a3 = tl.reshape(a_vec, (BLOCK_KW, 4, 2))
    ev, od = tl.split(a3)  # j in {0,2,4,6} / {1,3,5,7}
    e = tl.reshape(ev, (BLOCK_KW, 2, 2))
    o = tl.reshape(od, (BLOCK_KW, 2, 2))
    e0, e1 = tl.split(e)  # {0,4} / {2,6}
    o0, o1 = tl.split(o)  # {1,5} / {3,7}
    a0, a4 = tl.split(e0)
    a2, a6 = tl.split(e1)
    a1, a5 = tl.split(o0)
    a3_, a7 = tl.split(o1)
    return a0, a1, a2, a3_, a4, a5, a6, a7


# ======================================================================================
# Decode (M==1) W4A16 GEMV: int32 wide loads (8 codes/word) + deferred reduction, fp32 accum.
# ======================================================================================
@triton.jit
def _nvfp4_gemv_kernel(
    a_ptr,        # [K] activation (compute dtype), contiguous
    packed_ptr,   # [N, K // 8] int32 (8 fp4 codes per word, nibble j -> k = 8*w + j)
    scale_ptr,    # [N, K // 16] fp8-e4m3 per-16 block scale
    gscale_ptr,   # [N] fp16 per-output-row global scale (weight_scale_2)
    out_ptr,      # [N] out dtype
    N, K,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr, OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    offs_kw = tl.arange(0, BLOCK_KW)
    K_WORDS = K // 8
    p_base = packed_ptr + offs_n[None, :] * stride_pn
    s_base = scale_ptr + offs_n[None, :] * stride_sn
    partial = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_KW)):
        widx = kw_start * BLOCK_KW + offs_kw
        offs_a = kw_start * BLOCK_KW * 8 + tl.arange(0, BLOCK_KW * 8)
        if EVEN_K:
            word = tl.load(p_base + widx[:, None] * stride_pkw, mask=n_mask[None, :], other=0)
            # 8 codes/word fall in the same or adjacent 16-wide block -> one scale per word.
            s_ptrs = s_base + (widx[:, None] // 2) * stride_sblk
            if e4m3_native_cx():
                scale = tl.load(s_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32) * 16384.0
            else:
                scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=n_mask[None, :], other=0)) * 16384.0
            a_vec = tl.load(a_ptr + offs_a).to(tl.float32)
        else:
            w_mask = widx < K_WORDS
            word = tl.load(
                p_base + widx[:, None] * stride_pkw,
                mask=w_mask[:, None] & n_mask[None, :], other=0,
            )
            s_ptrs = s_base + (widx[:, None] // 2) * stride_sblk
            s_mask = w_mask[:, None] & n_mask[None, :]
            if e4m3_native_cx():
                scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32) * 16384.0
            else:
                scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0)) * 16384.0
            a_vec = tl.load(a_ptr + offs_a, mask=offs_a < K, other=0.0).to(tl.float32)

        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(a_vec, BLOCK_KW)
        acc_w = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word)
        acc_w += a0[:, None] * b_lo
        acc_w += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 4)
        acc_w += a1[:, None] * b_lo
        acc_w += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 8)
        acc_w += a2[:, None] * b_lo
        acc_w += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 12)
        acc_w += a3[:, None] * b_lo
        acc_w += a7[:, None] * b_hi
        partial += acc_w * scale

    acc = tl.sum(partial, axis=0)
    g = tl.load(gscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n, (acc * g).to(OUT), mask=n_mask)


@triton.jit
def _nvfp4_gemv_splitk_kernel(
    a_ptr,        # [K] activation (compute dtype), contiguous
    packed_ptr,   # [N, K // 8] int32
    scale_ptr,    # [N, K // 16] fp8-e4m3
    gscale_ptr,   # [N] fp16 per-output-row global scale
    part_ptr,     # [SPLIT_K, N] fp32 partial sums (pre-global-scale)
    counter_ptr,  # [cdiv(N, BLOCK_N)] int32 arrival counters, zeroed (self-resetting)
    out_ptr,      # [N] out dtype
    N, K, K_WORDS, tiles_per,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_partk, stride_partn,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr, SPLIT_K: tl.constexpr,
    OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    """Split-K decode GEMV for small N: each ``(pid_n, pid_k)`` reduces ``tiles_per`` word-
    tiles of K into a partial sum; many more programs than the single-pass kernel -> fills
    the SMs when ``N`` is small (shared expert). The last program to land a partial for an
    N-tile reduces all SPLIT_K partials and writes the global-scaled output in-kernel (the
    acq_rel arrival atomic orders the partial stores), so no separate reduce launch."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_kw = tl.arange(0, BLOCK_KW)

    tile_start = pid_k * tiles_per
    p_base = packed_ptr + offs_n[None, :] * stride_pn
    s_base = scale_ptr + offs_n[None, :] * stride_sn
    partial = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
    for t in range(tiles_per):
        widx = (tile_start + t) * BLOCK_KW + offs_kw
        offs_a = (tile_start + t) * BLOCK_KW * 8 + tl.arange(0, BLOCK_KW * 8)
        if EVEN_K:
            word = tl.load(p_base + widx[:, None] * stride_pkw, mask=n_mask[None, :], other=0)
            s_ptrs = s_base + (widx[:, None] // 2) * stride_sblk
            if e4m3_native_cx():
                scale = tl.load(s_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32) * 16384.0
            else:
                scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=n_mask[None, :], other=0)) * 16384.0
            a_vec = tl.load(a_ptr + offs_a).to(tl.float32)
        else:
            w_mask = widx < K_WORDS  # out-of-range words read 0 (safe; add nothing)
            word = tl.load(
                p_base + widx[:, None] * stride_pkw,
                mask=w_mask[:, None] & n_mask[None, :], other=0,
            )
            s_ptrs = s_base + (widx[:, None] // 2) * stride_sblk
            s_mask = w_mask[:, None] & n_mask[None, :]
            if e4m3_native_cx():
                scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32) * 16384.0
            else:
                scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0)) * 16384.0
            a_vec = tl.load(a_ptr + offs_a, mask=offs_a < K, other=0.0).to(tl.float32)

        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(a_vec, BLOCK_KW)
        acc_w = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word)
        acc_w += a0[:, None] * b_lo
        acc_w += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 4)
        acc_w += a1[:, None] * b_lo
        acc_w += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 8)
        acc_w += a2[:, None] * b_lo
        acc_w += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 12)
        acc_w += a3[:, None] * b_lo
        acc_w += a7[:, None] * b_hi
        partial += acc_w * scale
    acc = tl.sum(partial, axis=0)
    tl.store(part_ptr + pid_k * stride_partk + offs_n * stride_partn, acc, mask=n_mask)

    cnt = tl.atomic_add(counter_ptr + pid_n, 1)  # default acq_rel/gpu: publishes the store
    if cnt == SPLIT_K - 1:
        total = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):  # fixed order -> replay-stable reduction
            total += tl.load(
                part_ptr + k * stride_partk + offs_n * stride_partn, mask=n_mask, other=0.0
            )
        g = tl.load(gscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs_n, (total * g).to(OUT), mask=n_mask)
        tl.store(counter_ptr + pid_n, 0)  # leave zeroed for the next launch/replay


def _gemv(a: torch.Tensor, packed_i32: torch.Tensor, scale: torch.Tensor,
          gscale: torch.Tensor, out_dtype: torch.dtype, transposed: bool) -> torch.Tensor:
    """M==1 W4A16 GEMV. ``a`` [K] compute-dtype; ``packed_i32`` logical [N, K//8] int32
    (row-major or a K-major transposed view); ``scale`` logical [N, K//16] fp8; ``gscale``
    [N] fp16. Picks split-K so small ``N`` (shared expert) still fills the SMs; large ``N``
    (lm_head) stays single-pass (split_k=1)."""
    out_tl = _TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16]
    N = packed_i32.shape[0]
    K = packed_i32.shape[1] * 8
    scale = e4m3_kernel_view(scale)
    out = torch.empty(N, dtype=out_dtype, device=a.device)
    block_kw = _GEMV_BLOCK_KW_T if transposed else _GEMV_BLOCK_KW_ROW
    sk_block_n = _GEMV_SPLITK_BLOCK_N_T if transposed else _GEMV_SPLITK_BLOCK_N_ROW

    K_WORDS = K // 8
    n_blocks_n = triton.cdiv(N, sk_block_n)
    num_tiles = triton.cdiv(K_WORDS, _GEMV_SPLITK_BLOCK_KW)
    # Target ~a thousand blocks total; pow2 split_k -> stable reduction order.
    split_k = max(1, min(_GEMV_SPLITK_TARGET // max(n_blocks_n, 1), num_tiles))
    split_k = 1 << (split_k.bit_length() - 1)

    if split_k == 1:  # large N (lm_head): single-pass full-K, direct global-scaled write
        even = K % (block_kw * 8) == 0
        _nvfp4_gemv_kernel[(triton.cdiv(N, _GEMV_BLOCK_N),)](
            a, packed_i32, scale, gscale, out, N, K,
            packed_i32.stride(0), packed_i32.stride(1), scale.stride(0), scale.stride(1),
            BLOCK_N=_GEMV_BLOCK_N, BLOCK_KW=block_kw, OUT=out_tl, EVEN_K=even,
            num_warps=_GEMV_WARPS,
        )
        return out

    tiles_per = triton.cdiv(num_tiles, split_k)
    # Even iff every (tile_start + t) word tile is fully in range for every pid_k.
    even = (K % (_GEMV_SPLITK_BLOCK_KW * 8) == 0) and (num_tiles == tiles_per * split_k)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a.device)
    counters = _splitk_counters(n_blocks_n, a.device)
    _nvfp4_gemv_splitk_kernel[(n_blocks_n, split_k)](
        a, packed_i32, scale, gscale, part, counters, out, N, K, K_WORDS, tiles_per,
        packed_i32.stride(0), packed_i32.stride(1), scale.stride(0), scale.stride(1),
        part.stride(0), part.stride(1),
        BLOCK_N=sk_block_n, BLOCK_KW=_GEMV_SPLITK_BLOCK_KW, SPLIT_K=split_k,
        OUT=out_tl, EVEN_K=even,
        num_warps=_GEMV_WARPS,
    )
    return out


# ======================================================================================
# Small-M (2..64) W4A16 dot GEMM: dequant in the K loop, tensor cores, optional split-K.
#
# Swapped-operand ("wdot") formulation: the *weight* tile is the lhs -- loaded as
# [BLOCK_N, BLOCK_KW] words (coalesced along N in the K-major layout), dequanted in
# place, and multiplied against the transposed activation slices: acc^T = W @ a^T.
# Feeding the dequanted weights to the MMA as lhs (A operand, registers) instead of
# rhs (B operand, which wgmma reads from shared memory) removes a shared-memory
# round trip per tile that capped the previous kernel at ~50% "Memory" (L1) SOL.
#
# Dequant + dot run in fp16: the pair bit trick yields exact fp16 values (x 2^-14);
# the fp8-e4m3 block scale converts to fp16 exactly and is pre-multiplied by 128
# (max 448 * 128 < 65504). (e2m1 * 2^-14) * (scale * 128) has <= 6 significand bits
# at magnitude in [2^-17, 32) -> exact even in the fp16-subnormal corner (>= 7 bits
# of subnormal precision at 2^-17), so b is exact and only the fp32 accumulate
# rounds (same as cuBLAS bf16); the epilogue multiplies the fp32 accumulator by
# g * 128 to restore the deferred 2^7 (a power of two: bit-identical to scaling in
# the loop, one fp16 multiply per dot operand cheaper). Activations convert
# bf16 -> fp16 (exact up to 65504; standard fp16-inference range assumption, same
# as Marlin's fp16 mode).
# ======================================================================================
@triton.jit
def _nvfp4_pair_f16(v):
    """Dequant two e2m1 codes at int32 bits [3:0], [19:16] -> (lo, hi) fp16 = value * 2^-14."""
    r = ((v << 9) & 0x0E000E00) | ((v & 0x00080008) << 12)
    lo = (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (r >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return lo, hi


@triton.jit
def _nvfp4_gemm_kernel(
    a_ptr,        # [M, K] activations
    packed_ptr,   # logical [N, K // 8] int32 (row- or K-major via strides)
    scale_ptr,    # logical [N, K // 16] fp8-e4m3
    gscale_ptr,   # [N] fp16
    c_ptr,        # [M, N] output (written when SPLIT_K == 1)
    part_ptr,     # [SPLIT_K, M, N] fp32 partials (written when SPLIT_K > 1)
    M, N, K, K_WORDS, tiles_per,
    stride_am, stride_ak,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_cm, stride_cn,
    stride_qk, stride_qm, stride_qn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr,
    SPLIT_K: tl.constexpr, OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    offs_kw = tl.arange(0, BLOCK_KW)
    offs_ak = tl.arange(0, BLOCK_KW * 8)
    tile0 = pid_k * tiles_per
    a_ptrs = a_ptr + offs_m[:, None] * stride_am \
        + (tile0 * BLOCK_KW * 8 + offs_ak)[None, :] * stride_ak
    b_ptrs = packed_ptr + offs_n[:, None] * stride_pn \
        + (tile0 * BLOCK_KW + offs_kw)[None, :] * stride_pkw
    s_ptrs = scale_ptr + offs_n[:, None] * stride_sn \
        + ((tile0 * BLOCK_KW + offs_kw)[None, :] // 2) * stride_sblk

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for t in range(tiles_per):
        if EVEN_K:
            word = tl.load(b_ptrs, mask=n_mask[:, None], other=0)
            if e4m3_native_cx():
                s128 = tl.load(s_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float16) * 128.0
            else:
                s128 = e4m3_u8_to_f16_x128(tl.load(s_ptrs, mask=n_mask[:, None], other=0))
            a_tile = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        else:
            kw_ok = (tile0 + t) * BLOCK_KW + offs_kw < K_WORDS
            word = tl.load(b_ptrs, mask=n_mask[:, None] & kw_ok[None, :], other=0)
            s_mask = n_mask[:, None] & kw_ok[None, :]
            if e4m3_native_cx():
                s128 = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float16) * 128.0
            else:
                s128 = e4m3_u8_to_f16_x128(tl.load(s_ptrs, mask=s_mask, other=0))
            ak_ok = (tile0 + t) * BLOCK_KW * 8 + offs_ak < K
            a_tile = tl.load(a_ptrs, mask=m_mask[:, None] & ak_ok[None, :], other=0.0)

        # 8 nibble-position dots: b_j [BLOCK_N, BLOCK_KW] (k = 8*kw + j) x a_j^T [KW, BM].
        # b carries 2^-14 from the pair trick and s128 = scale * 2^7; both fp16 products
        # stay exact (<= 6 significand bits, magnitudes in [2^-17, 2^15)), and the missing
        # 2^7 is restored by the fp32 epilogue's g * 128 -- bit-identical to scaling b by
        # 2^7 here, one fewer fp16 multiply per dot operand.
        a0, a1, a2, a3, a4, a5, a6, a7 = _split8_rows(
            a_tile.to(tl.float16), BLOCK_M, BLOCK_KW
        )
        b_lo, b_hi = _nvfp4_pair_f16(word)
        acc = tl.dot(b_lo * s128, tl.trans(a0), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a4), acc)
        b_lo, b_hi = _nvfp4_pair_f16(word >> 4)
        acc = tl.dot(b_lo * s128, tl.trans(a1), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a5), acc)
        b_lo, b_hi = _nvfp4_pair_f16(word >> 8)
        acc = tl.dot(b_lo * s128, tl.trans(a2), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a6), acc)
        b_lo, b_hi = _nvfp4_pair_f16(word >> 12)
        acc = tl.dot(b_lo * s128, tl.trans(a3), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a7), acc)

        a_ptrs += BLOCK_KW * 8 * stride_ak
        b_ptrs += BLOCK_KW * stride_pkw
        s_ptrs += (BLOCK_KW // 2) * stride_sblk

    io_mask = m_mask[None, :] & n_mask[:, None]
    if SPLIT_K == 1:
        g = tl.load(gscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32) * 128.0
        acc = acc * g[:, None]
        c_ptrs = c_ptr + offs_m[None, :] * stride_cm + offs_n[:, None] * stride_cn
        tl.store(c_ptrs, acc.to(OUT), mask=io_mask)
    else:
        q_ptrs = (part_ptr + pid_k * stride_qk + offs_m[None, :] * stride_qm
                  + offs_n[:, None] * stride_qn)
        tl.store(q_ptrs, acc, mask=io_mask)


@triton.jit
def _split8_rows(a_tile, ROWS: tl.constexpr, BLOCK_KW: tl.constexpr):
    """Contiguous ``[ROWS, BLOCK_KW * 8]`` -> 8 ``[ROWS, BLOCK_KW]`` slices, a_j[r, w] = a[r, 8w+j].

    Register-only (reshape + split tree), mirrors :func:`_split8` with a leading rows dim."""
    a4 = tl.reshape(a_tile, (ROWS, BLOCK_KW, 4, 2))
    ev, od = tl.split(a4)  # j in {0,2,4,6} / {1,3,5,7}
    e = tl.reshape(ev, (ROWS, BLOCK_KW, 2, 2))
    o = tl.reshape(od, (ROWS, BLOCK_KW, 2, 2))
    e0, e1 = tl.split(e)  # {0,4} / {2,6}
    o0, o1 = tl.split(o)  # {1,5} / {3,7}
    a0, a4_ = tl.split(e0)
    a2, a6 = tl.split(e1)
    a1, a5 = tl.split(o0)
    a3, a7 = tl.split(o1)
    return a0, a1, a2, a3, a4_, a5, a6, a7


@triton.jit
def _nvfp4_gemm_splitk_reduce_kernel(
    part_ptr, gscale_ptr, out_ptr, M, N, SPLIT_K: tl.constexpr,
    stride_qk, stride_qm, stride_qn,
    stride_om, stride_on,
    BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    """Split-K reduce for the dot GEMM. Kept as a separate launch on purpose: fusing it
    into the GEMM (last-arriver pattern, as the GEMV does) costs more than it saves there
    -- the [BLOCK_N, BLOCK_M] fp32 reduce tile inflates the main loop's register budget
    (118 -> ~150 regs, occupancy 25% -> 19%) and the reduction runs serially at the very
    end of the kernel, where the GEMV's [BLOCK_N] reduce is small enough to be free."""
    pid_m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_qk + pid_m * stride_qm + offs * stride_qn,
                       mask=mask, other=0.0)
    g = tl.load(gscale_ptr + offs, mask=mask, other=0.0).to(tl.float32) * 128.0
    tl.store(out_ptr + pid_m * stride_om + offs * stride_on, (acc * g).to(OUT), mask=mask)


def _pick_split_k_bm16(num_mn: int, num_tiles: int, wave: int) -> int:
    """Split-K for the M <= 16 (decode-batch) grid. At these M the op is pure weight
    streaming, so the grid must reach ~one wave of blocks (``wave`` = SM count x resident
    blocks/SM, see :func:`_gemm_wave`) before bandwidth saturates -- but each program
    still needs >= ~3 K-tiles or the load pipeline never leaves its ramp. Thresholds
    fitted on H100 against Marlin (qkv/o_proj/gate_up/down + shared-expert shapes); the
    wave scaling is what carries them to other SM counts/occupancies."""
    max_sk = max(num_tiles // 3, 1)
    if num_tiles <= 16:  # tiny K (shared experts): launch-bound, just spread what's there
        max_sk = num_tiles
    min_grid = wave - wave // 32  # just-under-a-wave grids (e.g. 512 vs 528) are fine
    split_k = 1
    while split_k * 2 <= max_sk and num_mn * split_k < min_grid:
        split_k *= 2
    # A grid within ~15% of exactly one wave quantizes badly (one full wave + a tiny
    # straggler wave of full-length programs); take 2 waves when K can afford it.
    blocks = num_mn * split_k
    if (0.85 * wave <= blocks <= 1.15 * wave
            and split_k * 2 <= max_sk and num_tiles >= 10 * split_k):
        split_k *= 2
    return split_k


def _gemm_inkernel(a: torch.Tensor, packed_i32: torch.Tensor, scale: torch.Tensor,
                   gscale: torch.Tensor, out_dtype: torch.dtype,
                   transposed: bool) -> torch.Tensor:
    """Batched-decode dot GEMM (M <= 64): weight read once, split-K keeps the SMs fed when
    ``M x N`` alone yields too few programs (shared-expert N at small M)."""
    M, K = a.shape
    N = packed_i32.shape[0]
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    scale = e4m3_kernel_view(scale)
    out = torch.empty((M, N), dtype=compute, device=a.device)
    BLOCK_M = 16 if M <= 16 else (32 if M <= 32 else 64)
    BLOCK_N = 128
    num_warps = 8 if BLOCK_M == 64 else 4
    num_stages = 3

    K_WORDS = K // 8
    num_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    num_tiles = triton.cdiv(K_WORDS, _GEMM_BLOCK_KW)
    if BLOCK_M == 16:
        split_k = _pick_split_k_bm16(num_mn, num_tiles, _gemm_wave(a.device))
    else:
        split_k = max(1, min(_GEMM_SPLITK_TARGET // max(num_mn, 1), num_tiles))
        split_k = 1 << (split_k.bit_length() - 1)
        # Prefer a split that divides the tile count evenly: every program then runs the
        # mask-free EVEN_K loop (~15-30% faster). Only fall back to the uneven split when
        # evenness would collapse the grid (and with it SM occupancy).
        even_k = split_k
        while even_k > 1 and num_tiles % even_k != 0:
            even_k //= 2
        if even_k * 4 >= split_k:
            split_k = even_k
    tiles_per = triton.cdiv(num_tiles, split_k)
    even = (K % (_GEMM_BLOCK_KW * 8) == 0) and (num_tiles == tiles_per * split_k)
    part = (torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
            if split_k > 1 else out)  # unused dummy when split_k == 1
    _nvfp4_gemm_kernel[(num_mn, split_k)](
        a, packed_i32, scale, gscale, out, part,
        M, N, K, K_WORDS, tiles_per,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        part.stride(0) if split_k > 1 else 0,
        part.stride(1) if split_k > 1 else 0,
        part.stride(2) if split_k > 1 else 0,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_KW=_GEMM_BLOCK_KW,
        SPLIT_K=split_k, OUT=_TL_DTYPE[compute], EVEN_K=even,
        num_warps=num_warps, num_stages=num_stages,
    )
    if split_k > 1:
        # Small blocks spread the reduce over more SMs (it is latency-bound, ~2.5us);
        # very wide N amortizes better with fewer, fatter blocks.
        r_block, r_warps = (128, 2) if N <= 8192 else (512, 8)
        _nvfp4_gemm_splitk_reduce_kernel[(M, triton.cdiv(N, r_block))](
            part, gscale, out, M, N, split_k,
            part.stride(0), part.stride(1), part.stride(2),
            out.stride(0), out.stride(1),
            BLOCK=r_block, OUT=_TL_DTYPE[compute], num_warps=r_warps,
        )
    return out


# ======================================================================================
# Dequant-to-scratch + cuBLAS (prefill / batched decode): dequant once at memory roof.
# ======================================================================================
@triton.jit
def _nvfp4_dequant_rows_kernel(
    packed_ptr,   # [N, K // 8] int32
    scale_ptr,    # [N, K // 16] fp8-e4m3
    gscale_ptr,   # [N] fp16
    out_ptr,      # [N, K] compute dtype
    K_WORDS,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_on, stride_ok,
    BLOCK_KW: tl.constexpr, OUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_kw = pid_k * BLOCK_KW + tl.arange(0, BLOCK_KW)
    kw_mask = offs_kw < K_WORDS

    word = tl.load(packed_ptr + pid_n * stride_pn + offs_kw * stride_pkw, mask=kw_mask, other=0)
    s_ptrs = scale_ptr + pid_n * stride_sn + (offs_kw // 2) * stride_sblk
    if e4m3_native_cx():
        scale = tl.load(s_ptrs, mask=kw_mask, other=0.0).to(tl.float32)
    else:
        scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=kw_mask, other=0))
    g = tl.load(gscale_ptr + pid_n).to(tl.float32)
    scale = scale * (16384.0 * g)

    shifts = (tl.arange(0, 8) * 4).to(tl.int32)
    codes = (word[:, None] >> shifts[None, :]) & 0xF  # [KW, 8], natural k order
    r = ((codes << 9) & 0x0E00) | ((codes & 0x8) << 12)
    vals = r.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32) * scale[:, None]
    offs_k = offs_kw[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(out_ptr + pid_n * stride_on + offs_k * stride_ok, vals.to(OUT), mask=kw_mask[:, None])


@triton.jit
def _nvfp4_dequant_cols_kernel(
    packed_ptr,   # K-major: element (n, kw) at kw * stride_pkw + n (stride_pn == 1)
    scale_ptr,
    gscale_ptr,   # [N] fp16
    out_ptr,      # [n_rows, K] compute dtype, row-major
    N, K_WORDS,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_on, stride_ok,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr, OUT: tl.constexpr,
):
    """Dequant for the K-major resident layout: loads coalesce along N, the [K-tile, N-tile]
    value block is transposed in-register, stores coalesce along K of the row-major scratch."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_kw = pid_k * BLOCK_KW + tl.arange(0, BLOCK_KW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kw_mask = offs_kw < K_WORDS
    n_mask = offs_n < N

    word = tl.load(
        packed_ptr + offs_kw[:, None] * stride_pkw + offs_n[None, :] * stride_pn,
        mask=kw_mask[:, None] & n_mask[None, :], other=0,
    )
    s_ptrs = scale_ptr + (offs_kw[:, None] // 2) * stride_sblk + offs_n[None, :] * stride_sn
    s_mask = kw_mask[:, None] & n_mask[None, :]
    if e4m3_native_cx():
        scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    else:
        scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))
    g = tl.load(gscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    scale = scale * (16384.0 * g[None, :])

    shifts = (tl.arange(0, 8) * 4).to(tl.int32)
    codes = (word[:, None, :] >> shifts[None, :, None]) & 0xF  # [KW, 8, BN], k order
    r = ((codes << 9) & 0x0E00) | ((codes & 0x8) << 12)
    vals = r.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32) * scale[:, None, :]
    vals = tl.reshape(vals, (BLOCK_KW * 8, BLOCK_N))
    vals_t = tl.trans(vals.to(OUT))  # [BN, KW*8]
    offs_k = pid_k * BLOCK_KW * 8 + tl.arange(0, BLOCK_KW * 8)
    tl.store(
        out_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok,
        vals_t, mask=n_mask[:, None] & (offs_k[None, :] < K_WORDS * 8),
    )


def _dequant_rows(packed_i32: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor,
                  out: torch.Tensor, transposed: bool) -> None:
    """Dequant logical ``packed_i32 [n, K//8]`` (+ scales) into preallocated ``out [n, K]``."""
    n = packed_i32.shape[0]
    K_WORDS = packed_i32.shape[1]
    scale = e4m3_kernel_view(scale)
    if transposed:
        BLOCK_N, BLOCK_KW = 64, 16
        _nvfp4_dequant_cols_kernel[(triton.cdiv(K_WORDS, BLOCK_KW), triton.cdiv(n, BLOCK_N))](
            packed_i32, scale, gscale, out, n, K_WORDS,
            packed_i32.stride(0), packed_i32.stride(1), scale.stride(0), scale.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_KW=BLOCK_KW, OUT=_TL_DTYPE[out.dtype], num_warps=4,
        )
        return
    BLOCK_KW = 128
    _nvfp4_dequant_rows_kernel[(n, triton.cdiv(K_WORDS, BLOCK_KW))](
        packed_i32, scale, gscale, out, K_WORDS,
        packed_i32.stride(0), packed_i32.stride(1), scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_KW=BLOCK_KW, OUT=_TL_DTYPE[out.dtype], num_warps=4,
    )


def _gemm_scratch(a: torch.Tensor, packed_i32: torch.Tensor, scale: torch.Tensor,
                  gscale: torch.Tensor, out_dtype: torch.dtype,
                  transposed: bool) -> torch.Tensor:
    """Dequant the weight to a bf16 scratch (N-chunked) and run cuBLAS. The dequant runs
    once at the memory roof, so at prefill M this sits within ~7% of a resident-bf16 GEMM
    while the weight stays FP4-resident."""
    M, K = a.shape
    N = packed_i32.shape[0]
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    a = a.to(compute)
    chunk = min(N, _SCRATCH_CHUNK_BYTES // (K * compute.itemsize))
    # Keep the chunked GEMM's N dimension 256-aligned: an odd N (e.g. 13107) drops
    # cuBLAS onto a ~4x slower misaligned kernel.
    chunk = max(256, chunk - chunk % 256)
    if chunk >= N:
        w = torch.empty((N, K), dtype=compute, device=a.device)
        _dequant_rows(packed_i32, scale, gscale, w, transposed)
        return torch.nn.functional.linear(a, w)
    out = torch.empty((M, N), dtype=compute, device=a.device)
    w = torch.empty((chunk, K), dtype=compute, device=a.device)
    tmp = torch.empty((M, chunk), dtype=compute, device=a.device)
    for n0 in range(0, N, chunk):
        n1 = min(n0 + chunk, N)
        wc = w[: n1 - n0]
        _dequant_rows(packed_i32[n0:n1], scale[n0:n1], gscale[n0:n1], wc, transposed)
        # matmul into a contiguous temp: cuBLAS refuses a strided-out epilogue and torch
        # would fall back to a slow path if given the non-contiguous out[:, n0:n1] directly.
        tc = tmp[:, : n1 - n0]
        torch.matmul(a, wc.t(), out=tc)
        out[:, n0:n1].copy_(tc)
    return out


def _gemm(a: torch.Tensor, packed_i32: torch.Tensor, scale: torch.Tensor,
          gscale: torch.Tensor, out_dtype: torch.dtype, transposed: bool) -> torch.Tensor:
    """M>1 W4A16 GEMM dispatch. Small M (batched decode, lm_head last-token batch) reads
    the packed weight directly in the dot GEMM (split-K when needed); larger M (prefill)
    goes dequant-to-scratch + cuBLAS."""
    M = a.shape[0]
    if M <= _GEMM_MAX_INKERNEL_M:
        return _gemm_inkernel(a, packed_i32, scale, gscale, out_dtype, transposed)
    return _gemm_scratch(a, packed_i32, scale, gscale, out_dtype, transposed)


def _ref(a: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor,
         gscale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Numeric reference: dequant the FP4 weight to bf16 (well-tested ``dequant_nvfp4``)
    then a plain matmul. Used by FREETOKEN_DEBUG_DENSE_NVFP4_REF=1 for A-B validation."""
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    slots = torch.zeros(1, dtype=torch.int32, device=a.device)
    w = dequant_nvfp4(
        packed.unsqueeze(0), scale.unsqueeze(0), gscale.unsqueeze(0), slots, dtype=torch.bfloat16
    )[0]  # [N, K] bf16
    return (a.reshape(-1, a.shape[-1]) @ w.t()).to(out_dtype)


def _linear_impl(
    x: torch.Tensor, packed_i32: torch.Tensor, scale: torch.Tensor,
    gscale: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype,
    transposed: bool,
) -> torch.Tensor:
    """Shared dispatch on a logical ``[N, K//8]`` int32 weight view (either storage order)."""
    *lead, K = x.shape
    N = packed_i32.shape[0]
    if x.numel() // K == 1:
        out = _gemv(x.reshape(K), packed_i32, scale, gscale, out_dtype,
                    transposed).reshape(*lead, N)
    else:
        out = _gemm(x.reshape(-1, K).contiguous(), packed_i32, scale, gscale, out_dtype,
                    transposed).reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def nvfp4_dense_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    weight_global: torch.Tensor, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ dequant(weight)^T`` on the checkpoint-native row-major layout.
    ``weight`` [N, K//2] uint8, ``weight_scale`` [N, K//16] fp8-e4m3, ``weight_global`` [N] fp16."""
    if _USE_REF:
        *lead, K = x.shape
        out = _ref(x, weight, weight_scale, weight_global, x.dtype).reshape(*lead, weight.shape[0])
        return out + bias.to(out.dtype) if bias is not None else out
    return _linear_impl(
        x, weight.view(torch.int32), weight_scale, weight_global, bias, x.dtype, transposed=False
    )


def nvfp4_transpose_resident(weight: torch.Tensor, weight_scale: torch.Tensor):
    """Repack a row-major NVFP4 weight (``[N, K//2]`` uint8 + ``[N, K//16]`` fp8) into the
    K-major resident layout (``[K//8, N]`` int32 + ``[K//16, N]`` fp8). K-major storage makes
    the GEMV/dot-GEMM weight loads coalesce along N (the tile's wide axis), lifting batched
    decode from ~18% to ~35% and M=1 from ~50% to ~60-70% of the weight-bandwidth roof on
    H100. Done once at load; the transform is an exact permutation."""
    return (
        weight.view(torch.int32).t().contiguous(),
        weight_scale.t().contiguous(),
    )


def nvfp4_dense_linear_t(
    x: torch.Tensor, weight_t: torch.Tensor, scale_t: torch.Tensor,
    weight_global: torch.Tensor, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ dequant(W)^T`` on the K-major resident layout from
    :func:`nvfp4_transpose_resident` (``weight_t`` [K//8, N] int32, ``scale_t`` [K//16, N])."""
    if _USE_REF:
        out = _gemm_scratch(
            x.reshape(-1, x.shape[-1]).contiguous(), weight_t.t(), scale_t.t(),
            weight_global, x.dtype, transposed=True,
        ).reshape(*x.shape[:-1], weight_t.shape[1])
        return out + bias.to(out.dtype) if bias is not None else out
    return _linear_impl(
        x, weight_t.t(), scale_t.t(), weight_global, bias, x.dtype, transposed=True
    )


# ======================================================================================
# BaseOP linear layers (TP=1, replicated). Buffers: uint8 packed ``weight`` + fp8 block
# ``weight_scale`` + fp16 per-row ``weight_global``.
# ======================================================================================
class Nvfp4DenseLinear(BaseOP):
    """Replicated NVFP4 dense linear (W4A16). Drop-in for ``LinearReplicated`` /
    ``LinearRowParallel`` at TP=1 on the mixed-precision checkpoint's NVFP4 dense weights.

    Buffers are declared (and loaded) in the checkpoint's row-major layout; at load the
    packed weight + block scales are repacked to K-major (:func:`nvfp4_transpose_resident`)
    so the decode kernels' weight loads coalesce along N (~2x batched-decode throughput)."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        assert in_features % 16 == 0, f"NVFP4 in_features must be %16, got {in_features}"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features // 2, dtype=torch.uint8)
        self.weight_scale = torch.empty(out_features, in_features // 16, dtype=FP8)
        self.weight_global = torch.empty(out_features, dtype=torch.float16)
        self.bias = torch.empty(out_features) if has_bias else None
        self._transposed = False

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        w = state_dict.pop(_concat_prefix(prefix, "weight"))
        s = state_dict.pop(_concat_prefix(prefix, "weight_scale"))
        assert w.shape == self.weight.shape and w.dtype == torch.uint8
        assert s.shape == self.weight_scale.shape
        self.weight, self.weight_scale = nvfp4_transpose_resident(w, s)
        self.weight_global = state_dict.pop(_concat_prefix(prefix, "weight_global"))
        if self.bias is not None:
            self.bias = state_dict.pop(_concat_prefix(prefix, "bias"))
        self._transposed = True
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._transposed:
            return nvfp4_dense_linear_t(
                x, self.weight, self.weight_scale, self.weight_global, self.bias
            )
        return nvfp4_dense_linear(x, self.weight, self.weight_scale, self.weight_global, self.bias)


class Nvfp4DenseColMerged(Nvfp4DenseLinear):
    """Column-merged NVFP4 dense linear (drop-in for ``LinearColParallelMerged``):
    one packed weight concatenating several projections on the output dim; each part keeps its
    own per-row ``weight_global`` (and block scales), so the fused weight is exact. The caller
    splits the output by the (TP-local) part sizes (e.g. shared-expert gate|up).

    With ``local_output_sizes`` (TP>1) each part is sharded along the output dim
    independently and the block scales / per-row globals are sharded identically."""

    def __init__(self, in_features: int, output_sizes: list[int], has_bias: bool = False,
                 local_output_sizes: list[int] | None = None):
        self.output_sizes = list(output_sizes)
        if local_output_sizes is not None:
            assert len(local_output_sizes) == len(output_sizes), (
                local_output_sizes, output_sizes
            )
            out_features = sum(local_output_sizes)
        else:
            out_features = sum(output_sizes)
        super().__init__(in_features, out_features, has_bias)


class Nvfp4DenseRowParallel(Nvfp4DenseLinear):
    """Row-parallel NVFP4 dense linear (drop-in for ``LinearRowParallel``): packed ``weight``
    ``[out, in/tp//2]`` + block ``weight_scale`` ``[out, in/tp//16]`` + per-row
    ``weight_global`` ``[out]`` (the output dim is not sharded). Each rank reduces over its
    input shard only; the all-reduce completes the sum and the bias (if any) is added once,
    after it. The K-major repack at load is an exact permutation, so it works on the shard."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        from freetoken.distributed import DistributedCommunicator, get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        self._tp_size = tp_info.size
        self._comm = DistributedCommunicator() if tp_info.size > 1 else None
        super().__init__(div_even(in_features, tp_info.size), out_features, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._transposed:
            y = nvfp4_dense_linear_t(
                x, self.weight, self.weight_scale, self.weight_global, None
            )
        else:
            y = nvfp4_dense_linear(x, self.weight, self.weight_scale, self.weight_global, None)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


class Nvfp4LMHead(BaseOP):
    """NVFP4 (W4A16) LM head for the mixed checkpoint (TP=1, untied). Mirrors
    ``ParallelLMHead.forward`` at TP=1: slice to the last token per sequence at prefill, then
    the W4A16 GEMV/GEMM instead of a bf16 ``F.linear`` over the (here ~1 GB) bf16 weight."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        assert embedding_dim % 16 == 0
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(num_embeddings, embedding_dim // 2, dtype=torch.uint8)
        self.weight_scale = torch.empty(num_embeddings, embedding_dim // 16, dtype=FP8)
        self.weight_global = torch.empty(num_embeddings, dtype=torch.float16)
        self._transposed = False

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        w = state_dict.pop(_concat_prefix(prefix, "weight"))
        s = state_dict.pop(_concat_prefix(prefix, "weight_scale"))
        assert w.shape == self.weight.shape and w.dtype == torch.uint8
        self.weight, self.weight_scale = nvfp4_transpose_resident(w, s)
        self.weight_global = state_dict.pop(_concat_prefix(prefix, "weight_global"))
        self._transposed = True
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        if self._transposed:
            return nvfp4_dense_linear_t(x, self.weight, self.weight_scale, self.weight_global)
        return nvfp4_dense_linear(x, self.weight, self.weight_scale, self.weight_global)


__all__ = [
    "FP8",
    "nvfp4_dense_linear",
    "nvfp4_dense_linear_t",
    "nvfp4_transpose_resident",
    "Nvfp4DenseLinear",
    "Nvfp4DenseColMerged",
    "Nvfp4DenseRowParallel",
    "Nvfp4LMHead",
]
