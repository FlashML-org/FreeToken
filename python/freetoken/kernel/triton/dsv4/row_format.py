"""Device-side row-format helpers shared by the DSV4 and DSV41 kernels.

The Python-side descriptors live in ``freetoken.kvcache.dsv4.row_format``; the ``FMT`` constexpr a
kernel receives is ``RowFormat.code``:

  0  bf16          2 B / channel, no scale
  1  fp8_e8m0_b32  e4m3 byte / channel, ue8m0 scale byte per 32 channels
  2  fp4_e4m3_b16  e2m1 nibble / channel (even channel low nibble), e4m3 scale byte per 16
  3  fp4_e8m0_b32  e2m1 nibble / channel, ue8m0 scale byte per 32
  4  fp8_e8m0_b64_rope  e4m3 byte / channel, ue8m0 scale byte per 64, the last 64 (rope) channels as bf16

A row is ``[values | rope | scales | pad]``; ``ROW_BYTES`` is the pool's row stride. Every dequantized value
is rounded through bf16 before use -- the reference bakes its quant round-trip into a bf16 cache,
so this is what its attention / indexer kernels actually read.
"""

from __future__ import annotations

import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32
from freetoken.kernel.triton.kv_rows import stage_tile
from freetoken.kvcache.dsv4.row_format import FP8_E8M0_B64_ROPE

FMT_BF16 = 0
FMT_FP8_E8M0_B32 = 1
FMT_FP4_E4M3_B16 = 2
FMT_FP4_E8M0_B32 = 3
FMT_FP8_E8M0_B64_ROPE = 4

ROPE_DIM = tl.constexpr(FP8_E8M0_B64_ROPE.rope)
ROPE_GROUP = tl.constexpr(FP8_E8M0_B64_ROPE.group)


@triton.jit
def _fp8_b64_values(base, mask, offs_d, D: tl.constexpr):
    """FMT 4 e4m3 channels of the rows at ``base`` [T, 1] as fp32; masked lanes read 0."""
    NOPE: tl.constexpr = D - ROPE_DIM
    v = tl.load(base + offs_d[None, :], mask=mask, other=0)
    s = tl.load(base + NOPE + 2 * ROPE_DIM + (offs_d // ROPE_GROUP)[None, :], mask=mask, other=0)
    if e4m3_native_cx():
        x = v.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        x = e4m3_u8_to_f32(v)
    # 2^(s - 127) as fp32 bits: cheaper than exp2, and s = 0 (masked lanes) gives 0
    return x * (s.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


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
    elif FMT == 4:
        NOPE: tl.constexpr = D - ROPE_DIM
        nope = (offs_d < NOPE)[None, :]
        rope_ptr = (base + NOPE).to(tl.pointer_type(tl.bfloat16))
        r = tl.load(rope_ptr + (offs_d - NOPE)[None, :], mask=m & ~nope, other=0.0)
        val = tl.where(nope, _fp8_b64_values(base, m & nope, offs_d, D), r.to(tl.float32))
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


@triton.jit
def load_rope_split(row_ptr, valid, offs_d, D: tl.constexpr):
    """FMT 4 rows at ``row_ptr`` [T] as bf16 ``(values [T, D] with the rope channels zeroed, rope [T, ROPE_DIM])``.

    Exact: an e4m3 value times a power-of-two scale fits bf16. For kernels that dot the two parts
    separately; one [T, D] tile mixing the two decodes costs extra registers and shared memory.
    Invalid rows read as 0.
    """
    NOPE: tl.constexpr = D - ROPE_DIM
    base = row_ptr[:, None]
    rows = valid[:, None]
    nope = rows & (offs_d < NOPE)[None, :]
    # staged so the CTA decodes the tile once (see kv_rows.stage_tile)
    kv = stage_tile(_fp8_b64_values(base, nope, offs_d, D)).to(tl.bfloat16)
    offs_r = tl.arange(0, ROPE_DIM)
    rope_ptr = (base + NOPE).to(tl.pointer_type(tl.bfloat16))
    r = tl.load(rope_ptr + offs_r[None, :], mask=rows & (offs_r >= 0)[None, :], other=0.0)
    return kv, r


__all__ = [
    "FMT_BF16",
    "FMT_FP8_E8M0_B32",
    "FMT_FP4_E4M3_B16",
    "FMT_FP4_E8M0_B32",
    "FMT_FP8_E8M0_B64_ROPE",
    "fp4_nibble_to_f32",
    "fp4_f32_to_nibble",
    "load_rows",
    "load_rope_split",
]
