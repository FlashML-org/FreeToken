from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language import target_info

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_f32_to_u8,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    f32_to_e4m3,
    round_e4m3,
)
from freetoken.kvcache.format import KVFormat


@triton.jit
def kv_tile(ptrs, mask, FMT: tl.constexpr):
    """One K or V tile in the dtype ``tl.dot`` takes."""
    if FMT == "none":
        tile = tl.load(ptrs, mask=mask, other=0.0)
    elif FMT == "fp8_per_token_head":
        codes = tl.load(ptrs, mask=mask, other=0)
        if e4m3_native_cx():
            tile = codes.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        else:
            tile = e4m3_u8_to_f32(codes).to(tl.bfloat16)
    elif FMT == "int8_per_token_head":
        tile = tl.load(ptrs, mask=mask, other=0).to(tl.bfloat16)
    return tile


@triton.jit
def _stage(x):
    if target_info.is_hip():
        return x
    # Triton moves a dot operand's conversion back to its load past any side-effect-free op, so every warp would decode the whole tile; this asm stops that
    return tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=False, pack=1)


@triton.jit
def kv_tile_staged(ptrs, mask, FMT: tl.constexpr):
    """kv_tile for a tile that every warp of the CTA reads in full: on NVIDIA it decodes once per CTA and hands the bf16 tile to the MMA through shared memory; on HIP it is kv_tile."""
    if FMT == "none":
        tile = tl.load(ptrs, mask=mask, other=0.0)
    else:
        codes = tl.load(ptrs, mask=mask, other=0)
        if FMT == "fp8_per_token_head":
            if e4m3_native_cx():
                x = codes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            else:
                x = e4m3_u8_to_f32(codes)
        else:
            x = codes.to(tl.float32)
        tile = _stage(x).to(tl.bfloat16)
    return tile


@triton.jit
def kv_row_scale(ptrs, mask, FMT: tl.constexpr):
    """Row scales of a tile; all ones without scales, so the caller's multiply folds away."""
    if FMT == "none":
        scale = tl.full(mask.shape, 1.0, tl.float32)
    else:
        scale = tl.load(ptrs, mask=mask, other=0.0)
    return scale


def kv_scale_args(
    kv_format: KVFormat,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    placeholder: torch.Tensor,
    shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Scale pointers and slot strides for a launch; a format without scales never reads them. ``shape`` is ``(slots, heads)`` when the caches are paged."""
    if not kv_format.quantized:
        assert k_cache.is_floating_point() and v_cache.is_floating_point()
        return placeholder, placeholder, 0, 0
    assert k_cache.dtype == v_cache.dtype == kv_format.elem_store_dtype
    assert k_scale.dtype == v_scale.dtype == kv_format.scale_store_dtype
    shape = tuple(k_cache.shape[:2]) if shape is None else shape
    assert tuple(k_scale.shape[:2]) == tuple(v_scale.shape[:2]) == shape
    assert k_scale.stride(1) == v_scale.stride(1) == 1
    return k_scale, v_scale, k_scale.stride(0), v_scale.stride(0)


@triton.jit
def _encode_row(x, FMT: tl.constexpr):
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-10)
    if FMT == "fp8_per_token_head":
        scale = amax / 448.0
        q = tl.clamp(x / scale, -448.0, 448.0)
        if e4m3_native_cx():
            codes = f32_to_e4m3(q).to(tl.uint8, bitcast=True)
        else:
            codes = e4m3_f32_to_u8(round_e4m3(q))
    elif FMT == "int8_per_token_head":
        scale = amax / 127.0
        q = x / scale
        # the int8 cast truncates, so round half away from zero first
        codes = tl.clamp(tl.where(q >= 0, q + 0.5, q - 0.5), -128.0, 127.0).to(tl.int8)
    return codes, scale


@triton.jit
def _kv_store_kernel(
    k_src,
    v_src,
    k_dst,
    v_dst,
    k_scale_dst,
    v_scale_dst,
    slots,
    stride_kt,
    stride_vt,
    stride_kd,
    stride_vd,
    stride_ks,
    stride_vs,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FMT: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots + token).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    row = head * D + offs

    k_row = tl.load(k_src + token * stride_kt + row, mask=mask, other=0.0).to(tl.float32)
    v_row = tl.load(v_src + token * stride_vt + row, mask=mask, other=0.0).to(tl.float32)
    k_codes, k_scale = _encode_row(k_row, FMT)
    v_codes, v_scale = _encode_row(v_row, FMT)

    tl.store(k_dst + slot * stride_kd + row, k_codes, mask=mask)
    tl.store(v_dst + slot * stride_vd + row, v_codes, mask=mask)
    tl.store(k_scale_dst + slot * stride_ks + head, k_scale)
    tl.store(v_scale_dst + slot * stride_vs + head, v_scale)


def store_kv_rows(
    fmt: KVFormat,
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize ``[tokens, heads * head_dim]`` K and V rows and write codes and scales to the ``out_loc`` slots."""
    tokens = k.shape[0]
    if tokens == 0:
        return
    _, heads, head_dim = k_cache.shape
    assert k.shape == v.shape == (tokens, heads * head_dim), (k.shape, v.shape, heads, head_dim)
    assert k.stride(1) == v.stride(1) == 1 and k_cache.stride(1) == v_cache.stride(1) == head_dim
    assert k_cache.dtype == v_cache.dtype == fmt.elem_store_dtype
    assert k_scale.dtype == v_scale.dtype == fmt.scale_store_dtype
    assert k_scale.shape[:2] == k_cache.shape[:2] and v_scale.shape[:2] == v_cache.shape[:2]
    assert k_scale.stride(1) == v_scale.stride(1) == 1
    _kv_store_kernel[(tokens, heads)](
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
        D=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        FMT=fmt.name,
        num_warps=1,
    )



@triton.jit
def _latent_store_kernel(
    c_src,
    r_src,
    dst,
    scale_dst,
    slots,
    stride_ct,
    stride_cd,
    stride_rt,
    stride_rd,
    stride_d,
    stride_s,
    D_V: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    FMT: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < D_V
    latent = tl.load(c_src + token * stride_ct + offs_v * stride_cd, mask=mask_v, other=0.0).to(tl.float32)
    codes, scale = _encode_row(latent, FMT)
    tl.store(dst + slot * stride_d + offs_v, codes, mask=mask_v)
    tl.store(scale_dst + slot * stride_s, scale)
    if D_R > 0:
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < D_R
        rope = tl.load(r_src + token * stride_rt + offs_r * stride_rd, mask=mask_r, other=0.0).to(tl.float32)
        codes, scale = _encode_row(rope, FMT)
        tl.store(dst + slot * stride_d + D_V + offs_r, codes, mask=mask_r)
        tl.store(scale_dst + slot * stride_s + 1, scale)


def store_latent_rows(
    fmt: KVFormat,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    out_loc: torch.Tensor,
    rows: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Quantize MLA latent rows ``c_kv | k_rope`` into the ``[slots, latent_dim]`` pool, each part with its own scale."""
    tokens = c_kv.shape[0]
    if tokens == 0:
        return
    d_v, d_r = c_kv.shape[-1], k_rope.shape[-1]
    assert k_rope.shape[0] == tokens and rows.shape[1] == d_v + d_r and rows.stride(1) == 1
    assert rows.dtype == fmt.elem_store_dtype and scale.dtype == fmt.scale_store_dtype
    assert scale.shape == (rows.shape[0], 2) and scale.stride(1) == 1
    _latent_store_kernel[(tokens,)](
        c_kv,
        k_rope,
        rows,
        scale,
        out_loc,
        c_kv.stride(0),
        c_kv.stride(1),
        k_rope.stride(0),
        k_rope.stride(1),
        rows.stride(0),
        scale.stride(0),
        D_V=d_v,
        D_R=d_r,
        BLOCK_V=triton.next_power_of_2(d_v),
        BLOCK_R=triton.next_power_of_2(max(d_r, 1)),
        FMT=fmt.name,
        num_warps=1,
    )
