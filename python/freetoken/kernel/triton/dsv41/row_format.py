"""Device-side row-format helpers shared by the DSV41 kernels.

The Python-side descriptors live in ``freetoken.kvcache.dsv4.v41_row_format``; the ``FMT`` constexpr a
kernel receives is ``RowFormat.code``:

  0  bf16          2 B / channel, no scale
  1  fp8_e8m0_b32  e4m3 byte / channel, ue8m0 scale byte per 32 channels
  2  fp4_e4m3_b16  e2m1 nibble / channel (even channel low nibble), e4m3 scale byte per 16
  3  fp4_e8m0_b32  e2m1 nibble / channel, ue8m0 scale byte per 32

A row is ``[values | scales]``; ``ROW_BYTES`` is the pool's row stride. Every dequantized value
is rounded through bf16 before use -- the reference bakes its quant round-trip into a bf16 cache,
so this is what its attention / indexer kernels actually read.
"""

from __future__ import annotations

import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_u8_to_f32

FMT_BF16 = 0
FMT_FP8_E8M0_B32 = 1
FMT_FP4_E4M3_B16 = 2
FMT_FP4_E8M0_B32 = 3


@triton.jit
def fp4_nibble_to_f32(nib):
    """e2m1 code (uint8, 0..15) -> fp32: bit 3 sign, bits 2-1 exponent, bit 0 mantissa;
    codes 0/1 are the subnormals 0 and 0.5, the rest ``(1 + m/2) * 2^(e-1)``."""
    c = (nib & 7).to(tl.int32)
    neg = (nib & 8) != 0
    sub = c.to(tl.float32) * 0.5
    normal = (1.0 + 0.5 * (c & 1).to(tl.float32)) * tl.exp2(((c >> 1) - 1).to(tl.float32))
    mag = tl.where(c < 2, sub, normal)
    return tl.where(neg, -mag, mag)


@triton.jit
def fp4_f32_to_nibble(q):
    """fp32 value already on the e2m1 grid ({0, .5, 1, 1.5, 2, 3, 4, 6}, signed) -> e2m1 code.
    Thresholds sit between grid points, so a value a hair off the grid still maps to its point."""
    a = tl.abs(q)
    c = tl.where(
        a < 1.75,
        tl.where(a < 0.75, tl.where(a < 0.25, 0, 1), tl.where(a < 1.25, 2, 3)),
        tl.where(a < 3.5, tl.where(a < 2.5, 4, 5), tl.where(a < 5.0, 6, 7)),
    )
    return (c + tl.where(q < 0, 8, 0)).to(tl.uint8)


@triton.jit
def load_rows(pool_ptr, rows, valid, offs_d, D: tl.constexpr, ROW_BYTES: tl.constexpr, FMT: tl.constexpr):
    """Gather + dequantize ``[T, D]`` fp32 from a packed pool.

    ``pool_ptr`` uint8 pool base; ``rows`` [T] row ids (any int width; masked-out rows may be -1);
    ``valid`` [T] bool; ``offs_d`` [D] = ``tl.arange(0, D)``. Invalid rows read as 0.
    """
    row_i64 = tl.maximum(rows, 0).to(tl.int64)
    base = pool_ptr + row_i64[:, None] * ROW_BYTES
    m = valid[:, None] & (offs_d >= 0)[None, :]
    if FMT == 0:
        lo = tl.load(base + 2 * offs_d[None, :], mask=m, other=0).to(tl.uint16)
        hi = tl.load(base + 2 * offs_d[None, :] + 1, mask=m, other=0).to(tl.uint16)
        val = (lo | (hi << 8)).to(tl.bfloat16, bitcast=True).to(tl.float32)
    elif FMT == 1:
        v = tl.load(base + offs_d[None, :], mask=m, other=0)
        s = tl.load(base + D + (offs_d // 32)[None, :], mask=m, other=127)
        val = e4m3_u8_to_f32(v) * tl.exp2(s.to(tl.float32) - 127.0)
    else:
        b = tl.load(base + (offs_d // 2)[None, :], mask=m, other=0)
        nib = tl.where((offs_d % 2 == 0)[None, :], b & 0xF, b >> 4)
        q = fp4_nibble_to_f32(nib)
        if FMT == 2:
            s = e4m3_u8_to_f32(tl.load(base + D // 2 + (offs_d // 16)[None, :], mask=m, other=0))
        else:
            s = tl.exp2(tl.load(base + D // 2 + (offs_d // 32)[None, :], mask=m, other=127).to(tl.float32) - 127.0)
        val = q * s
    return val.to(tl.bfloat16).to(tl.float32)


__all__ = [
    "FMT_BF16",
    "FMT_FP8_E8M0_B32",
    "FMT_FP4_E4M3_B16",
    "FMT_FP4_E8M0_B32",
    "fp4_nibble_to_f32",
    "fp4_f32_to_nibble",
    "load_rows",
]
