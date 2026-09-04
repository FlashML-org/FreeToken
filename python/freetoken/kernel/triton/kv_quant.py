"""FP8 (e4m3) KV-cache storage: per-token, per-head symmetric quantization.

One KV row is one ``(token, kv_head)`` slice of ``head_dim`` elements. It is stored
as ``head_dim`` e4m3 bytes plus ONE fp32 scale shared by the whole row:

    scale = max(amax(row) / 448, eps)          # 448 == e4m3 finite max
    code  = round_e4m3(clamp(row / scale))     # RNE onto the e4m3 grid
    read  = code.to(f32) * scale               # in the attention kernels

Granularity rationale: an fp32 scale per (token, head) costs ``4 / head_dim`` bytes
per element (3% at head_dim 128, 6% at 64) while tracking each key's own magnitude,
which is what keeps a quantized KV from collapsing on outlier heads. A coarser
per-tensor scale needs no storage at all but has no headroom for them; a finer
per-element "scale" is the format itself.

Architectures below sm_89 have no fp8e4nv type in Triton (see
:mod:`freetoken.kernel.triton.e4m3_compat`), so the codes live in a plain ``uint8``
buffer on EVERY architecture and are decoded by :func:`e4m3_u8_to_f32`. That keeps one
set of bytes and one set of numbers across GPUs, and it is why :func:`kv_codes_dtype`
is a constant rather than a question: the fp8 type never appears in a kernel
signature, so nothing here can disagree with the host that allocated the buffer.
Choosing the encode/decode per target -- by an arch probe, or by testing the pointer's
element type -- is what broke this feature twice on real hardware (see
:func:`kv_load_e4m3_tile_f32`).

The write path replaces ``kernel.store_cache`` for a quantized pool: the plain store
kernel is a raw byte copy that requires the source and the cache to share a dtype,
and quantization is exactly the step where the two diverge. Folding the scatter into
the quantization kernel keeps that to a single launch (and a single HBM round trip)
under CUDA-graph capture, where the slot ids arrive as a device tensor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_f32_to_u8,
    round_e4m3,
)

FP8 = torch.float8_e4m3fn
KV_SCALE_DTYPE = torch.float32
KV_QUANT_FP8 = "fp8"


def kv_codes_dtype() -> torch.dtype:
    """Storage dtype of one quantized KV element: e4m3 bytes in a uint8 buffer.

    A constant, deliberately. Every attempt to answer this per target -- triton's
    compile-time probe, then testing the pointer's element type at the load -- ended up
    picking an arm that did not match the buffer the host had just allocated (see the
    module header). torch still reads these bytes as fp8 whenever real numbers are
    wanted: :func:`codes_to_f32`.
    """
    return torch.uint8


def alloc_codes(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """A zero-filled code buffer of :func:`kv_codes_dtype` -- bytes, on every arch.

    Zero-filling matters because the pools read slots that were never written: a stale
    byte decodes to a real number (0x7F/0xFF even to NaN once reinterpreted as fp8),
    while 0x00 is exactly 0.0 (same reasoning as kvcache/bsa_pool.py).
    """
    return torch.zeros(shape, dtype=kv_codes_dtype(), device=device)


def codes_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """Decode a code buffer to fp32 ON THE HOST (torch's own e4m3 cast).

    Works on either storage dtype -- uint8 bytes are reinterpreted as fp8 first -- so
    a test or a debugging tool reads the same numbers on sm_86 as on sm_90, and does
    it through torch rather than through the software decoder it is checking.
    """
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

    # Straight-line, like the reader: round onto the e4m3 grid in ONE step (RNE) and
    # pack the bits into the byte buffer. No fp8 type on either side -- the reason this
    # feature stopped compiling twice is explained in kv_codes_dtype's docstring.
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
    assert k_scale.dtype == KV_SCALE_DTYPE, k_scale.dtype
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


__all__ = [
    "FP8",
    "KV_QUANT_FP8",
    "KV_SCALE_DTYPE",
    "alloc_codes",
    "codes_to_f32",
    "kv_codes_dtype",
    "quantize_kv_to_cache",
]

