"""Row-scaled NVFP4 KV storage, with low nibble first and 16-wide blocks.

Each (token, head) row has an FP32 scale; each block has an E4M3 scale
stored as uint8. Reconstruction is E2M1 * block_scale * row_scale.
Row-local second-level scales keep appends independent of the cached prefix.
"""

import torch
import triton
import triton.language as tl

from .e4m3_compat import e4m3_f32_to_u8, e4m3_u8_to_f32, round_e4m3


@triton.jit
def _encode_e2m1(x):
    a = tl.abs(x)
    code = tl.full(x.shape, 0, tl.int32)
    code = tl.where(a > 0.25, 1, code)
    code = tl.where(a >= 0.75, 2, code)
    code = tl.where(a > 1.25, 3, code)
    code = tl.where(a >= 1.75, 4, code)
    code = tl.where(a > 2.5, 5, code)
    code = tl.where(a >= 3.5, 6, code)
    code = tl.where(a > 5.0, 7, code)
    return code | tl.where(x < 0, 8, 0)


@triton.jit
def _decode_e2m1(code):
    # Place E2M1 bits in FP16, then compensate the exponent-bias difference (15 - 1).
    bits = ((code & 8).to(tl.uint16) << 12) | ((code & 7).to(tl.uint16) << 9)
    return bits.to(tl.float16, bitcast=True).to(tl.float32) * 16384.0


@triton.jit
def load_nvfp4(ptr, block_ptr, row_ptr, slots, head, dims, slot_mask,
               stride_slot, stride_head, stride_row, D: tl.constexpr,
               DIM_OFFSET: tl.constexpr = 0):
    TRANSPOSE: tl.constexpr = dims.shape[0] != 1
    WIDTH: tl.constexpr = dims.shape[0] if TRANSPOSE else dims.shape[1]
    TOKENS: tl.constexpr = slots.shape[0] * slots.shape[1]
    slot = slots.reshape(TOKENS).to(tl.int64)
    valid = slot_mask.reshape(TOKENS)
    dim = DIM_OFFSET + tl.arange(0, WIDTH)
    packed = tl.load(
        ptr + slot[:, None] * stride_slot + head * stride_head + (dim[None, :] // 2),
        valid[:, None] & (dim[None, :] < D), other=0,
    ).to(tl.int32)
    codes = tl.where((dim[None, :] & 1) == 0, packed & 15, packed >> 4)
    block_dim = dim // 16
    block = tl.load(
        block_ptr + (slot[:, None] * stride_row + head) * (D // 16) + block_dim[None, :],
        valid[:, None] & (dim[None, :] < D), other=0,
    )
    row = tl.load(row_ptr + slot * stride_row + head, valid, other=0)
    value = _decode_e2m1(codes) * e4m3_u8_to_f32(block) * row[:, None]
    if TRANSPOSE:
        return tl.trans(value)
    else:
        return value


@triton.jit
def _quantize_row(src, dst, block_ptr, row_ptr, t, h, slot, stride_src,
                  HEADS: tl.constexpr, D: tl.constexpr, BLOCKS: tl.constexpr):
    blocks = tl.arange(0, BLOCKS)
    dims = blocks[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(src + t * stride_src + h * D + dims, dims < D, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    row_scale = tl.maximum(tl.max(amax, 0), 1e-10) / (6.0 * 448.0)
    block_scale = round_e4m3(tl.minimum(tl.div_rn(amax, 6.0 * row_scale), 448.0))
    # Quantize against the scale actually stored, including E4M3 rounding/underflow.
    denom = tl.where(block_scale > 0, block_scale * row_scale, 1.0)
    normalized = tl.where(block_scale[:, None] > 0, tl.div_rn(x, denom[:, None]), 0.0)
    codes = _encode_e2m1(normalized).reshape(BLOCKS, 8, 2)
    lo, hi = tl.split(codes)
    packed = lo | (hi << 4)
    byte_dims = blocks[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(dst + (slot * HEADS + h) * (D // 2) + byte_dims, packed, byte_dims < D // 2)
    tl.store(block_ptr + (slot * HEADS + h) * (D // 16) + blocks,
             e4m3_f32_to_u8(block_scale), blocks < D // 16)
    tl.store(row_ptr + slot * HEADS + h, row_scale)


@triton.jit
def _scatter_rows(src, dst, block, row, indices, stride_src,
                  D: tl.constexpr, BLOCKS: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(indices + t).to(tl.int64)
    _quantize_row(src, dst, block, row, t, 0, slot, stride_src, 1, D, BLOCKS)


def quantize_nvfp4_rows_to_cache(rows, out_loc, cache, scales, block_scales) -> None:
    """Quantize a single MLA latent slab, with one second-level scale per token."""
    tokens, dim = rows.shape
    slots = cache.shape[0]
    assert dim % 16 == 0 and rows.stride(1) == 1
    assert rows.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert cache.shape == (slots, dim // 2) and cache.dtype == torch.uint8
    assert scales.shape == (slots,) and scales.dtype == torch.float32
    assert block_scales.shape == (slots, dim // 16) and block_scales.dtype == torch.uint8
    assert out_loc.shape == (tokens,) and out_loc.dtype in (torch.int32, torch.int64)
    for tensor in (cache, scales, block_scales, out_loc):
        assert tensor.is_contiguous() and tensor.device == rows.device
    assert rows.is_cuda
    if tokens:
        _scatter_rows[(tokens,)](
            rows, cache, block_scales, scales, out_loc, rows.stride(0),
            D=dim, BLOCKS=triton.next_power_of_2(dim // 16),
            num_warps=4, enable_fp_fusion=False,
        )


@triton.jit
def _scatter(k, v, kc, vc, kb, vb, kr, vr, indices, stride_k, stride_v,
             HEADS: tl.constexpr, D: tl.constexpr, BLOCKS: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(indices + t).to(tl.int64)
    _quantize_row(k, kc, kb, kr, t, h, slot, stride_k, HEADS, D, BLOCKS)
    _quantize_row(v, vc, vb, vr, t, h, slot, stride_v, HEADS, D, BLOCKS)


def quantize_nvfp4_to_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_block_scale: torch.Tensor,
    v_block_scale: torch.Tensor,
) -> None:
    tokens, width = k.shape
    slots, heads, packed_dim = k_cache.shape
    dim = packed_dim * 2
    assert dim % 16 == 0 and width == heads * dim
    assert v.shape == k.shape and k.stride(1) == v.stride(1) == 1
    assert k.dtype in (torch.float16, torch.bfloat16, torch.float32) and v.dtype == k.dtype
    assert out_loc.shape == (tokens,) and out_loc.dtype in (torch.int32, torch.int64)
    assert out_loc.is_contiguous()
    for codes, row, block in ((k_cache, k_scale, k_block_scale), (v_cache, v_scale, v_block_scale)):
        assert codes.shape == (slots, heads, packed_dim) and codes.dtype == torch.uint8
        assert row.shape == (slots, heads) and row.dtype == torch.float32
        assert block.shape == (slots, heads, dim // 16) and block.dtype == torch.uint8
        assert codes.is_contiguous() and row.is_contiguous() and block.is_contiguous()
        assert codes.device == row.device == block.device == k.device
    assert k.is_cuda and v.device == out_loc.device == k.device
    if tokens:
        _scatter[(tokens, heads)](k, v, k_cache, v_cache, k_block_scale, v_block_scale,
                                 k_scale, v_scale, out_loc, k.stride(0), v.stride(0),
                                 HEADS=heads, D=dim, BLOCKS=triton.next_power_of_2(dim // 16),
                                 num_warps=4, enable_fp_fusion=False)
