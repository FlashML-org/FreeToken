"""Quantizing store into a KV pool (8-bit and sub-byte).

A program handles one ``(token, kv_head)`` pair: it loads that head's ``head_dim``
values as a ``[head_dim // BLOCK, BLOCK]`` tile, reduces max-abs along the block, and
writes the quantized values plus one scale per block. K and V are done in the same
program -- they share the token's slot index and the tile geometry, so doing both
halves the launch count and the index math.

Three layouts live behind ``LAYOUT``:
  * ``q8`` -- one int8 (or fp8) value per element. Bytes = elements along head_dim.
  * ``q4`` -- 16 bytes pack 32 4-bit values (low nibble + high nibble per byte).
  * ``q6`` -- 24 bytes pack 32 6-bit values: 16-byte low plane (low 4 bits, same
    nibble layout as Q4) followed by 8-byte high plane (top 2 bits at bit positions
    0, 2, 4, 6).

Triton 3.7.1 quirks this file walks around (memory: freetoken-kv-subbyte-quant):
  1. ``if <constexpr>`` dead branches still get type-checked: every branch must
     produce a value of compatible shape.
  2. All returns in a jit function merge into one type unconditionally. We work
     entirely in registers and produce a single packed value per element.
  3. Shift/broadcast vectors must be explicitly aligned to the right axis (use
     ``[:, None]`` / ``[None, None, :]``); a bare ``[QBLOCK]`` vector falls onto
     the token axis when D is on rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kvcache.quant import BLOCK
from freetoken.kvcache.quant import LAYOUT_NVFP4 as _LAYOUT_NVFP4
from freetoken.kvcache.quant import LAYOUT_Q4 as _LAYOUT_Q4
from freetoken.kvcache.quant import LAYOUT_Q6 as _LAYOUT_Q6
from freetoken.kvcache.quant import LAYOUT_Q8 as _LAYOUT_Q8

# Bind to tl.constexpr so the @triton.jit kernel can use them in `if LAYOUT == ...`
# (Triton 3.7.1 forbids reading plain Python globals from inside a jit function).
LAYOUT_Q8 = tl.constexpr(_LAYOUT_Q8)
LAYOUT_Q4 = tl.constexpr(_LAYOUT_Q4)
LAYOUT_Q6 = tl.constexpr(_LAYOUT_Q6)
LAYOUT_NVFP4 = tl.constexpr(_LAYOUT_NVFP4)


@triton.jit
def _store_kv_quant_kernel(
    k_ptr,  # [tokens, heads, D] source, compute dtype (bf16)
    v_ptr,
    kc_ptr,  # [slots, heads, D_PHYSICAL] destination, storage dtype
    vc_ptr,
    ks_ptr,  # [slots, heads, D // BLOCK] scales, fp16
    vs_ptr,
    indices_ptr,  # [tokens] destination slot per token
    stride_kt,
    stride_kh,
    stride_ct,
    stride_ch,
    stride_st,
    stride_sh,
    D: tl.constexpr,  # logical head_dim
    D_PHYSICAL: tl.constexpr,  # packed byte count (== D for q8)
    MAX_MAG: tl.constexpr,
    IS_INT: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,  # D // BLOCK
    LAYOUT: tl.constexpr,  # "q8" | "q4" | "q6"
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(indices_ptr + tok).to(tl.int64)

    # [NBLOCK, BLOCK] tile over head_dim: rows are quant blocks, columns the elements
    # sharing one scale. (memory: "BLK_STRIDE = payload_bytes_per_block" for the
    # WRITER; the read side just uses D.)
    offs = tl.arange(0, NBLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    scale_offs = tl.arange(0, NBLOCK)

    for is_v in tl.static_range(2):
        src_ptr = v_ptr if is_v else k_ptr
        dst_ptr = vc_ptr if is_v else kc_ptr
        sc_ptr = vs_ptr if is_v else ks_ptr

        x = tl.load(src_ptr + tok * stride_kt + head * stride_kh + offs).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        # An all-zero block quantizes to zeros under any positive scale; 1.0 keeps the
        # division finite.
        scale = tl.where(amax > 0, amax / MAX_MAG, 1.0)
        # Round to the stored precision before dividing, so the value written here and
        # the value the attention kernels read back are scaled by the identical number.
        scale = scale.to(sc_ptr.dtype.element_ty).to(tl.float32)
        # div_rn, not `/`: the plain operator is free to lower to a reciprocal multiply,
        # which disagrees with the torch reference on values sitting exactly between two
        # quantization steps. IEEE round-to-nearest divide makes the two bit-identical.
        q = tl.math.div_rn(x, scale[:, None])
        if IS_INT:
            # Round half away from zero (what GGUF's Q8_0 / Q4_0 / Q6_0 do), then clamp.
            # The 4-bit signed range is [-8, 7] (16 levels, stored as unsigned
            # 0..15 -- 8 maps to -8 in the XOR-sub sign extension) and 6-bit
            # signed is [-32, 31] (64 levels). For Q4, MAX_MAG=8 and the
            # writer must clamp at MAX_MAG-1=7 so the dequant sees a real
            # signed value; for Q6, MAX_MAG=31 already aligns the storage
            # with the 6-bit signed range so we clamp at MAX_MAG.
            q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
            if LAYOUT == LAYOUT_Q4:
                q = tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG - 1.0)
            else:
                q = tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG)
        else:
            if LAYOUT == LAYOUT_NVFP4:
                # E2M1 4-bit float: q is already in [-6, 6] from the div_rn above
                # (scale = amax / 6 ensures |q| <= 6). The pack step further down
                # calls _round_e2m1, so we keep `q` as fp32 here.
                pass
            else:
                # The native fp32 -> float8e4nv downcast does not round to nearest on
                # every arch (it lowers as a truncating fp32 -> fp16 -> e4m3 double-round
                # on sm_89), so values just above a grid midpoint collapse downward and
                # disagree with the RNE torch reference. Round explicitly first.
                from freetoken.kernel.triton.e4m3_compat import round_e4m3

                q = round_e4m3(tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG))

        # ---- pack into the storage dtype ----
        if LAYOUT == LAYOUT_Q8:
            # Direct: one value per byte.
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + offs,
                q.to(dst_ptr.dtype.element_ty),
            )
        elif LAYOUT == LAYOUT_Q4:
            # Pack two 4-bit values per byte: low nibble = even index, high = odd.
            qi = q.to(tl.int8)  # [NBLOCK, 32] in [-8, 7]
            low4 = qi & 0xF  # [NBLOCK, 32] each in [0, 15]
            # Reshape to [NBLOCK, 16, 2] (last dim = 2 to satisfy tl.split), then
            # split the last dim into the even/odd halves, each [NBLOCK, 16].
            pairs = tl.reshape(low4, (NBLOCK, 16, 2))
            lo, hi = tl.split(pairs)  # each [NBLOCK, 16]
            packed = (lo | (hi << 4)).to(tl.uint8)
            byte_offs = tl.arange(0, NBLOCK)[:, None] * 16 + tl.arange(0, 16)[None, :]
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + byte_offs,
                packed,
            )
        elif LAYOUT == LAYOUT_Q6:
            # 32 values -> 16-byte low plane (nibbles) + 8-byte high plane (2-bit tops
            # at bit positions 0, 2, 4, 6). Layout per block: lo first, then hi.
            qi = q.to(tl.int8)  # [NBLOCK, 32] in [-32, 31]
            lo4 = qi & 0xF  # low 4 bits
            hi2 = (qi >> 4) & 0x3  # top 2 bits
            lo_pairs = tl.reshape(lo4, (NBLOCK, 16, 2))
            lo, hi_lo = tl.split(lo_pairs)  # each [NBLOCK, 16]
            packed_lo = (lo | (hi_lo << 4)).to(tl.uint8)
            hi_groups = tl.reshape(hi2, (NBLOCK, 8, 4))
            # masks for bit positions 0, 2, 4, 6 = 1, 4, 16, 64; broadcasts to
            # [NBLOCK, 8, 4] for elementwise multiply along the value axis.
            idx = tl.arange(0, 4)
            masks = tl.where(
                idx == 0, 1,
                tl.where(idx == 1, 4,
                         tl.where(idx == 2, 16, 64)),
            )
            packed_hi = tl.sum(hi_groups * masks[None, None, :], axis=2).to(tl.uint8)
            # [NBLOCK, 8]
            base = tl.arange(0, NBLOCK)[:, None] * 24
            lo_offs = base + tl.arange(0, 16)[None, :]
            hi_offs = base + 16 + tl.arange(0, 8)[None, :]
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + lo_offs,
                packed_lo,
            )
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + hi_offs,
                packed_hi,
            )
        elif LAYOUT == LAYOUT_NVFP4:
            # NVFP4 is float, not integer: `q` here is the float quotient
            # (value / scale). Round each fp32 to the nearest E2M1 4-bit code
            # in [0, 15] using the same tie-break as the spec's torch argmin
            # (lowest index wins, so magnitude 0 ties go to code 0 = +0).
            # The cutoffs are midpoints between adjacent E2M1 magnitudes
            # [0, 0.5, 1, 1.5, 2, 3, 4, 6]: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0.
            # We use `<=` (round half DOWN to the lower code) so the tie-break
            # matches argmin's "lowest index on ties" behavior bit-for-bit
            # (e.g. q = 1.25, equidistant from 1.0 (code 2) and 1.5 (code 3),
            # both should land on code 2).
            mag = tl.minimum(tl.abs(q), 6.0)
            mag_code = tl.where(
                mag <= 0.25, 0,
                tl.where(
                    mag <= 0.75, 1,
                    tl.where(
                        mag <= 1.25, 2,
                        tl.where(
                            mag <= 1.75, 3,
                            tl.where(
                                mag <= 2.5, 4,
                                tl.where(
                                    mag <= 3.5, 5,
                                    tl.where(
                                        mag <= 5.0, 6, 7,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            # Sign bit only when magnitude > 0, so 0-magnitude ties land on
            # code 0 (= +0) the same way the spec's argmin does.
            sign = tl.where((q < 0) & (mag_code > 0), 8, 0)
            codes = (mag_code | sign).to(tl.int32)  # [NBLOCK, 16] in [0, 15]
            # Pack two codes per byte: low nibble = even index, high nibble = odd.
            pairs = tl.reshape(codes, (NBLOCK, 8, 2))
            lo, hi = tl.split(pairs)  # each [NBLOCK, 8]
            packed = (lo | (hi << 4)).to(tl.uint8)
            byte_offs = tl.arange(0, NBLOCK)[:, None] * 8 + tl.arange(0, 8)[None, :]
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + byte_offs,
                packed,
            )
        else:
            tl.static_assert(False, f"unknown LAYOUT {LAYOUT!r}")

        # Scales: one per block (logical, not packed). Same address math as Q8.
        tl.store(
            sc_ptr + slot * stride_st + head * stride_sh + scale_offs,
            scale.to(sc_ptr.dtype.element_ty),
        )


def store_kv_quant(
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec,
) -> None:
    """Quantize ``k``/``v`` ``[tokens, heads, D]`` into the pool slots ``indices``.

    ``k_cache``/``v_cache`` are ``[slots, heads, D_PHYSICAL]`` in the spec's storage
    dtype, where ``D_PHYSICAL == D`` for 8-bit and ``D * bits // 8`` for sub-byte.
    ``k_scale``/``v_scale`` are ``[slots, heads, D // BLOCK]`` in fp16.
    """
    from freetoken.kvcache.quant import BLOCK

    num_tokens, num_heads, head_dim = k.shape
    if num_tokens == 0:
        return
    # NVFP4 quantizes in 16-element blocks; Q2/Q4/Q6 (and q8 for 32-wide tiles) in 32.
    # We pick the per-kernel block size from the spec so a single kernel can serve both.
    block = spec.block_size if spec.layout == "nvfp4" else BLOCK
    assert head_dim % block == 0, f"head_dim {head_dim} not a multiple of {block}"
    # The cache's last axis is the PACKED byte count, which differs from the source's
    # head_dim for sub-byte schemes. We pass both as constexprs to the kernel.
    d_physical = k_cache.shape[-1]
    expected_physical = spec.physical_head_dim(head_dim) if spec.enabled else head_dim
    assert d_physical == expected_physical, (
        f"cache physical dim {d_physical} != spec {spec.name} expected {expected_physical}"
    )
    _store_kv_quant_kernel[(num_tokens, num_heads)](
        k,
        v,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        indices,
        k.stride(0),
        k.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        D=head_dim,
        D_PHYSICAL=d_physical,
        MAX_MAG=spec.max_magnitude,
        IS_INT=spec.is_integer,
        BLOCK=block,
        NBLOCK=head_dim // block,
        LAYOUT=spec.layout,
        num_warps=4,
    )


__all__ = ["store_kv_quant"]
