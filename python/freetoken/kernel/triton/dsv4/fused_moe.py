"""Inline-dequant DeepSeek-FP4 fused-MoE Triton kernels.

DeepSeek-V4 routed experts are stored as ``float4_e2m1fn_x2`` (two E2M1 codes per
byte) with a per-32 block scale in ``float8_e8m0fnu`` (power-of-two). This differs
from NVFP4 (``nvfp4_fused_moe.py``) in two ways:
  - block stride is ``K // 32`` (32 values per scale) -> ``sblk = byte_idx // 16``;
  - the scale is decoded as ``2^(code - 127)`` and there is NO separate per-row
    global scale.

Layout for an expert weight ``W[N, K]``:
  - ``packed[slot, n, k//2]`` uint8: two FP4 codes per byte, low nibble = even k.
  - ``scale[slot, n, k//32]`` float8_e8m0fnu (passed as uint8 exponent codes).
  - ``W[n, k] = E2M1_LUT[code] * 2^(scale[n, k//32] - 127)``.

As with NVFP4 the activations stay bf16 (fp32 accumulation), which is strictly
more precise than the reference's FP8-activation path.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import _log2_ceil
from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, round_e4m3

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
def _decode_dsfp4_moe_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 32] uint8 (e8m0 codes)
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
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    if slot < 0:
        # an inactive route (hybrid decode computes it on the CPU): zero contribution, and no read of
        # any slot -- a slot's storage may be uninitialized, and 0 * NaN is not 0
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_N,), dtype=compute_type), mask=(route_id < total_routes) & n_mask)
        return
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    NB: tl.constexpr = BLOCK_SIZE_KB // 16   # 16 bytes (==32 fp4 values) per e8m0 scale block
    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    offs_nb = tl.arange(0, NB)
    K_BYTES = K // 2
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    # Tile is [BLOCK_N, BLOCK_KB] with the contiguous K (byte) axis last so the
    # weight reads coalesce; reduce over K. (The transposed layout starved HBM.)
    # K_BYTES is a multiple of BLOCK_SIZE_KB for every DSV4 decode shape, so no
    # byte-tail mask is needed -- this keeps the scale exp2 off the critical path.
    for kb_start in range(0, K_BYTES // BLOCK_SIZE_KB):
        byte_idx = kb_start * BLOCK_SIZE_KB + offs_kb
        p_ptrs = packed_slot + offs_n[:, None] * stride_pn + byte_idx[None, :] * stride_pkb
        bytes_ = tl.load(p_ptrs, mask=n_mask[:, None], other=0).to(tl.int32)
        b_lo = tl.load(lut_ptr + (bytes_ & 0xF))
        b_hi = tl.load(lut_ptr + ((bytes_ >> 4) & 0xF))

        # One exp2 per (n, 16-byte scale block) instead of per element (the scale is
        # constant over each 16-byte run): ~16x fewer SFU ops -> ~1.6x throughput.
        sblk = kb_start * NB + offs_nb
        codes = tl.load(scale_slot + offs_n[:, None] * stride_sn + sblk[None, :] * stride_sblk,
                        mask=n_mask[:, None], other=0).to(tl.float32)
        sc = tl.exp2(codes - 127.0)                                          # [BLOCK_N, NB]
        b_lo = tl.reshape(b_lo, (BLOCK_SIZE_N, NB, 16)) * sc[:, :, None]
        b_hi = tl.reshape(b_hi, (BLOCK_SIZE_N, NB, 16)) * sc[:, :, None]
        b_lo = tl.reshape(b_lo, (BLOCK_SIZE_N, BLOCK_SIZE_KB))
        b_hi = tl.reshape(b_hi, (BLOCK_SIZE_N, BLOCK_SIZE_KB))

        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak).to(tl.float32)
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak).to(tl.float32)
        accumulator += tl.sum(b_lo * a_lo[None, :], axis=1)
        accumulator += tl.sum(b_hi * a_hi[None, :], axis=1)

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _swiglu_kernel(
    gu_ptr,            # [R, 2I] gate_up (compute dtype)
    out_ptr,           # [R, I] activated (compute dtype)
    R, I, limit,
    stride_gr, stride_gi, stride_or, stride_oi,
    BLOCK: tl.constexpr, HAS_LIMIT: tl.constexpr, compute_type: tl.constexpr,
    ACT_BLOCK: tl.constexpr,  # > 0: fp8 (e4m3, ue8m0 per ACT_BLOCK) round-trip of the stored value, fused
):
    """Fused SwiGLU: ``out = silu(min(gate, limit)) * clamp(up, -limit, limit)``.

    Collapses the gate/up split + two clamps + silu + mul + the fp32 round-trip
    (6 elementwise launches over [R, I]) between the two FP4 GEMVs into one pass. With
    ``ACT_BLOCK`` the down projection's activation quant (``act_quant_fp8_inplace`` on the bf16
    value: ``s = 2**ceil(log2(max(|x|, 1e-4) / 448))``, e4m3 rounding, dequant) is applied to the
    same tile before the store -- one launch and one memory pass fewer, bit-identical."""
    row = tl.program_id(0)
    cb = tl.program_id(1)
    offs = cb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    g = tl.load(gu_ptr + row * stride_gr + offs * stride_gi, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * stride_gr + (I + offs) * stride_gi, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        g = tl.minimum(g, limit)
        u = tl.minimum(tl.maximum(u, -limit), limit)
    act = (g * tl.sigmoid(g)) * u
    if ACT_BLOCK > 0:
        tl.static_assert(BLOCK % ACT_BLOCK == 0, "the tile holds whole quant blocks")
        xs = tl.reshape(act.to(compute_type).to(tl.float32), (BLOCK // ACT_BLOCK, ACT_BLOCK))  # the value the store would hold
        amax = tl.maximum(tl.max(tl.abs(xs), axis=1), 1e-4)
        sc = tl.exp2(_log2_ceil(amax * (1.0 / 448.0)).to(tl.float32))
        q = tl.clamp(xs / sc[:, None], -448.0, 448.0)
        if e4m3_native_cx():
            q = q.to(tl.float8e4nv).to(tl.float32)
        else:
            q = round_e4m3(q)
        act = tl.reshape(q * sc[:, None], (BLOCK,))
    tl.store(out_ptr + row * stride_or + offs * stride_oi, act.to(compute_type), mask=mask)


# REUSE-NOTE: trimmed local copy of the grouped/sorted fused-MoE GEMM from
# kernel/triton/mxfp4_moe.py (mxfp4_fused_moe_kernel) without the expert bias.
# ds_fp4 shares MXFP4's numerics (E2M1 + e8m0 per-32); the banks stay in the
# decode GEMV's [S, N, K//2] layout, so the B tiles coalesce along K instead
# of N. The dequant is neither that file's float arithmetic nor the decode
# GEMV's LUT gather (a dependent gather serializes in the tl.dot regime):
# the bf16 bit pattern is assembled with integer ops and the pow-of-two e8m0
# scale is applied as an exponent-field add -- zero SFU/fp32 work, bit-equal
# to LUT[code] * 2^(scale-127).
@triton.jit
def _fp4_bf16_bits(nib, shift):
    """E2M1 nibble [tile] + e8m0 exponent-field shift -> bf16 bit pattern
    (int32). 0.5*m for the subnormal codes, (1 + m/2) * 2^(e-1) -> exponent
    field 126+e, mantissa bit m<<6. ``shift = (e8m0 - 127) << 7`` folds into
    both branches: ``m * (bits(0.5) + shift)`` still yields +/-0 for the zero
    codes, so no separate zero guard is needed."""
    sign = (nib & 0x8) << 12
    e = (nib >> 1) & 0x3
    m = nib & 0x1
    mag = tl.where(e == 0, m * (0x3F00 + shift), ((126 + e) << 7) + (m << 6) + shift)
    return sign + mag


@triton.jit
def _prefill_dsfp4_moe_kernel(
    a_ptr,             # [M, K] activations (compute dtype, FP8 round-tripped)
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 32] uint8 (e8m0 codes)
    c_ptr,             # [M, TOP_K, N] output, indexed flat over M*TOP_K routes
    topk_weights_ptr,  # [M * TOP_K] fp32
    sorted_token_ids_ptr,
    expert_ids_ptr,    # bank row per M-block
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    offs_route_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_route = tl.load(sorted_token_ids_ptr + offs_route_id).to(tl.int64)
    route_mask = offs_route < num_valid_tokens

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    route_rows = offs_route // top_k

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    a_ptrs = a_ptr + route_rows[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B tile is built transposed [BN, ...]: the packed bytes then read
    # contiguously along their K-major axis (one byte per two k-values, loaded
    # once), lo/hi nibbles interleave back into k order, and tl.trans feeds the
    # dot. No per-element parity select in the hot loop.
    KB: tl.constexpr = BLOCK_SIZE_K // 2
    offs_kb = tl.arange(0, KB)
    packed_base = packed_ptr + slot * stride_pe + offs_n[:, None] * stride_pn
    scale_base = scale_ptr + slot * stride_se + offs_n[:, None] * stride_sn

    # One e8m0 row covers 32 k-values: load [BN, BLOCK_K//32] once per
    # iteration and broadcast, instead of a per-k gather re-reading it 32x.
    NSB: tl.constexpr = BLOCK_SIZE_K // 32
    tl.static_assert(BLOCK_SIZE_K % 32 == 0)
    offs_sb = tl.arange(0, NSB)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + offs_k
        a = tl.load(
            a_ptrs,
            mask=route_mask[:, None] & (k_offsets[None, :] < K),
            other=0.0,
        )
        byte_idx = k_start // 2 + offs_kb
        packed = tl.load(
            packed_base + byte_idx[None, :] * stride_pkb,
            mask=(offs_n[:, None] < N) & (byte_idx[None, :] * 2 < K),
            other=0,
        ).to(tl.int32)
        sblk = k_start // 32 + offs_sb
        codes = tl.load(
            scale_base + sblk[None, :] * stride_sblk,
            mask=(offs_n[:, None] < N) & (sblk[None, :] * 32 < K),
            other=127,
        )
        shift = (codes.to(tl.int32) - 127) << 7  # [BN, NSB] exponent-field add
        shift = tl.reshape(
            tl.broadcast_to(shift[:, :, None], (BLOCK_SIZE_N, NSB, 16)), (BLOCK_SIZE_N, KB)
        )
        bits = tl.interleave(
            _fp4_bf16_bits(packed & 0x0F, shift), _fp4_bf16_bits((packed >> 4) & 0x0F, shift)
        )
        b = tl.reshape(bits, (BLOCK_SIZE_N, BLOCK_SIZE_K)).to(tl.uint16).to(compute_type, bitcast=True)
        accumulator += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_SIZE_K * stride_ak

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + offs_route, mask=route_mask, other=0.0)
        accumulator = accumulator * weight[:, None]

    c_ptrs = c_ptr + stride_cm * offs_route[:, None] + stride_cn * offs_n[None, :]
    c_mask = route_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)


def _compute_type(dtype: torch.dtype):
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}[dtype]


def fused_swiglu(gate_up: torch.Tensor, limit: float, *, act_block: int = 0) -> torch.Tensor:
    """``[..., 2I] -> [..., I]`` SwiGLU in a single kernel (see ``_swiglu_kernel``). ``act_block > 0``
    also applies the down projection's fp8 activation round-trip (block ``act_block``) in the same pass."""
    *lead, two_I = gate_up.shape
    I = two_I // 2
    gu = gate_up.reshape(-1, two_I)
    R = gu.shape[0]
    out = torch.empty((R, I), dtype=gate_up.dtype, device=gate_up.device)
    BLOCK = 1024
    assert act_block == 0 or (I % act_block == 0 and BLOCK % act_block == 0), (I, act_block)
    grid = (R, triton.cdiv(I, BLOCK))
    _swiglu_kernel[grid](
        gu, out, R, I, float(limit),
        gu.stride(0), gu.stride(1), out.stride(0), out.stride(1),
        BLOCK=BLOCK, HAS_LIMIT=limit > 0,
        compute_type=_compute_type(gate_up.dtype), ACT_BLOCK=act_block, num_warps=4,
    )
    return out.reshape(*lead, I)


__all__ = [
    "_decode_dsfp4_moe_kernel",
    "_prefill_dsfp4_moe_kernel",
    "_swiglu_kernel",
    "fused_swiglu",
    "_e2m1_lut",
]
