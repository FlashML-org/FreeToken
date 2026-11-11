"""E4M3 KV storage with one FP32 scale per token, layer, head and K/V row.

Quantize and scatter in one launch. Codes use uint8 on every GPU so allocation
and Triton signatures agree even on devices without native E4M3 support.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_f32_to_u8,
    e4m3_u8_to_f32,
    round_e4m3,
)

FP8 = torch.float8_e4m3fn
KV_SCALE_DTYPE = torch.float32
KV_QUANT_FP8 = "fp8"


@triton.jit
def _reconstruct_kv_prefix_kernel(
    kc, vc, ks, vs, indptr, indices, prefixes, ko, vo,
    KCS: tl.constexpr, VCS: tl.constexpr,
    KSS: tl.constexpr, VSS: tl.constexpr,
    KOS: tl.constexpr, VOS: tl.constexpr,
    CAPACITY: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(1)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // (HEADS * D)
    head = offset // D % HEADS
    dim = offset % D
    start = tl.load(indptr + request)
    end = tl.load(indptr + request + 1)
    length = tl.minimum(tl.load(prefixes + request), end - start)
    if tl.program_id(0) * BLOCK < length * HEADS * D:
        slot = tl.load(indices + start + row, row < length, other=-1).to(tl.int64)
        valid = (row < length) & (slot >= 0) & (slot < CAPACITY)
        ck = tl.load(kc + slot * KCS[0] + head * KCS[1] + dim * KCS[2], valid, other=0)
        cv = tl.load(vc + slot * VCS[0] + head * VCS[1] + dim * VCS[2], valid, other=0)
        sk = tl.load(ks + slot * KSS[0] + head * KSS[1], valid, other=0.)
        sv = tl.load(vs + slot * VSS[0] + head * VSS[1], valid, other=0.)
        tl.store(ko + slot * KOS[0] + head * KOS[1] + dim * KOS[2], e4m3_u8_to_f32(ck) * sk, valid)
        tl.store(vo + slot * VOS[0] + head * VOS[1] + dim * VOS[2], e4m3_u8_to_f32(cv) * sv, valid)


def reconstruct_kv_prefix(
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_scale: torch.Tensor, v_scale: torch.Tensor,
    indptr: torch.Tensor, indices: torch.Tensor, prefix_lens: torch.Tensor,
    k_out: torch.Tensor, v_out: torch.Tensor,
) -> None:
    """Reconstruct historical rows once, retaining their physical slot addresses."""
    assert k_cache.ndim == 3
    assert k_cache.shape == v_cache.shape == k_out.shape == v_out.shape
    assert k_cache.dtype == v_cache.dtype == torch.uint8
    assert k_out.dtype == v_out.dtype == torch.bfloat16
    assert k_scale.shape == v_scale.shape == k_cache.shape[:2]
    assert k_scale.dtype == v_scale.dtype == torch.float32
    assert indptr.ndim == indices.ndim == prefix_lens.ndim == 1
    assert indptr.numel() == prefix_lens.numel() + 1
    assert all(x.dtype in (torch.int32, torch.int64) and x.stride(0) == 1
               for x in (indptr, indices, prefix_lens))
    assert k_cache.is_cuda and all(x.device == k_cache.device for x in
        (v_cache, k_scale, v_scale, indptr, indices, prefix_lens, k_out, v_out))
    if not indices.numel() or not prefix_lens.numel():
        return
    capacity, heads, dim = k_cache.shape
    _reconstruct_kv_prefix_kernel[(triton.cdiv(indices.numel() * heads * dim, 1024), prefix_lens.numel())](
        k_cache, v_cache, k_scale, v_scale, indptr, indices, prefix_lens, k_out, v_out,
        k_cache.stride(), v_cache.stride(), k_scale.stride(), v_scale.stride(),
        k_out.stride(), v_out.stride(), capacity, heads, dim, 1024,
        num_warps=4,
    )


def kv_codes_dtype() -> torch.dtype:
    """Raw E4M3 code dtype, independent of the target GPU."""
    return torch.uint8


def alloc_codes(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Initialize dummy and unwritten slots to finite E4M3 zero."""
    return torch.zeros(shape, dtype=kv_codes_dtype(), device=device)


def codes_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """Decode through PyTorch for diagnostics and independent kernel checks."""
    if codes.dtype is not FP8:
        codes = codes.view(FP8)
    return codes.to(torch.float32)


@triton.jit
def _kv_quant_scatter_kernel(
    k_src,
    v_src,
    k_dst,
    v_dst,
    k_scale,
    v_scale,
    idx_ptr,
    stride_xs,  # K source row pitch, in elements (the qkv slice is wider than one row)
    stride_vx,  # V source row pitch. K and V need not share one: a .clamp() on one side
                # leaves it densely packed, so reusing K's pitch reads V off its rows.
    stride_kd,  # K cache row pitch, in elements (== HEADS * D)
    stride_vd,
    stride_ks,  # scale row pitch, in elements (== HEADS)
    stride_vs,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (token, kv_head): quantize the row and write it to slot
    ``idx_ptr[token]`` (the same ``out_loc`` the bf16 store scatters through)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(idx_ptr + t).to(tl.int64)  # int64: slots * row can pass 2**31
    d = tl.arange(0, BLOCK_D)
    mask = d < D

    off = h * D + d
    xk = tl.load(k_src + t * stride_xs + off, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(v_src + t * stride_vx + off, mask=mask, other=0.0).to(tl.float32)

    # 448 == e4m3 finite max; 1e-10 is the amax floor of the activation quant in
    # kernel/triton/fp8_block_linear.py (literals keep the kernel self-contained).
    sk = tl.maximum(tl.max(tl.abs(xk), axis=0), 1e-10) / 448.0
    sv = tl.maximum(tl.max(tl.abs(xv), axis=0), 1e-10) / 448.0
    qk = tl.clamp(xk / sk, -448.0, 448.0)
    qv = tl.clamp(xv / sv, -448.0, 448.0)

    # Round to the E4M3 grid before packing the code bits (round-to-nearest-even).
    out_k = e4m3_f32_to_u8(round_e4m3(qk))
    out_v = e4m3_f32_to_u8(round_e4m3(qv))

    tl.store(k_dst + pos * stride_kd + h * D + d, out_k, mask=mask)
    tl.store(v_dst + pos * stride_vd + h * D + d, out_v, mask=mask)
    tl.store(k_scale + pos * stride_ks + h, sk)
    tl.store(v_scale + pos * stride_vs + h, sv)


def quantize_kv_to_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize fresh K/V rows into an fp8 KV pool.

    ``k``/``v``  : ``[T, num_kv_heads * head_dim]`` compute-dtype rows -- exactly what
        the attention backends hand to ``store_kv`` (a slice of the qkv projection,
        so the row pitch may be wider than the row itself).
    ``out_loc``  : ``[T]`` device slot index per row (int32 or int64).
    ``*_cache``  : ``[num_slots, num_kv_heads, head_dim]`` of :func:`kv_codes_dtype`.
    ``*_scale``  : ``[num_slots, num_kv_heads]`` fp32, indexed by the SAME slot.
    """
    tokens = k.shape[0]
    if tokens == 0:
        return
    assert k.dim() == 2 and v.shape == k.shape, (k.shape, v.shape)
    assert k.stride(1) == 1 and v.stride(1) == 1, "K/V rows must be contiguous"
    heads, dim = k_cache.shape[1], k_cache.shape[2]
    assert k.shape[1] == heads * dim, (tuple(k.shape), tuple(k_cache.shape))
    assert k_cache.shape == v_cache.shape, (tuple(k_cache.shape), tuple(v_cache.shape))
    assert k_scale.shape == v_scale.shape == (k_cache.shape[0], heads), (
        tuple(k_scale.shape),
        tuple(k_cache.shape),
    )
    assert k_cache.dtype == v_cache.dtype == kv_codes_dtype(), (
        k_cache.dtype,
        kv_codes_dtype(),
    )
    assert k_scale.dtype == v_scale.dtype == KV_SCALE_DTYPE
    assert k_cache.stride(1) == v_cache.stride(1) == dim
    assert k_cache.stride(2) == v_cache.stride(2) == 1
    assert k_scale.stride(1) == v_scale.stride(1) == 1
    assert out_loc.shape == (tokens,) and out_loc.dtype in (torch.int32, torch.int64)
    assert out_loc.stride(0) == 1
    assert all(t.device == k.device for t in (v, out_loc, k_cache, v_cache, k_scale, v_scale))
    _kv_quant_scatter_kernel[(tokens, heads)](
        k,
        v,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        out_loc,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        k_scale.stride(0),
        v_scale.stride(0),
        D=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=1,
    )
