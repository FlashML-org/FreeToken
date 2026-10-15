"""Quantize rows into a packed pool and read them back (DeepSeek-V4.1 KV storage formats).

``pack_rows`` reproduces the reference quantizers (``inference/kernel.py``) exactly:

* fp8_e8m0_b32  ``act_quant(x, 32, "ue8m0", e8m0)``: per 32-group ``s = 2**ceil(log2(max(amax, 1e-4) / 448))``,
                ``q = round_e4m3(clamp(x / s, +-448))``
* fp4_e4m3_b16  ``fp4_act_quant(x, 16, scale_dtype=e4m3)``: ``s = e4m3(max(amax, 6 * 2^-9) / 6)`` (round to
                nearest e4m3), ``q = round_e2m1(clamp(x / s, +-6))``
* fp4_e8m0_b32  ``fp4_act_quant(x, 32)``: ``s = 2**ceil(log2(max(amax, 6 * 2^-126) / 6))``, same e2m1 rounding
* bf16          a plain byte copy

Rows land at ``pool[row_ids[m]]`` (or ``pool[m]`` without ids), so the same launch serves the KV write
path (scatter into slots) and plain buffer conversion. ``unpack_rows`` is the inverse gather + dequant
(bf16-rounded like the reference cache); the attention / indexer kernels use the same device helper
(``row_format.load_rows``) in-line instead of materializing rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import _log2_ceil, _round_fp4
from freetoken.kernel.triton.e4m3_compat import e4m3_f32_to_u8, e4m3_native_cx, round_e4m3
from freetoken.kvcache.dsv4.v41_row_format import BF16, RowFormat

from .row_format import fp4_f32_to_nibble, load_rows


@triton.jit
def _e4m3_codes(q):
    """fp32 (|q| <= 448) -> e4m3 bit pattern, RNE."""
    if e4m3_native_cx():
        return q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        return e4m3_f32_to_u8(round_e4m3(q))


@triton.jit
def _e4m3_round(q):
    """fp32 -> nearest e4m3 value, as fp32."""
    if e4m3_native_cx():
        return q.to(tl.float8e4nv).to(tl.float32)
    else:
        return round_e4m3(q)


@triton.jit
def _pack_rows_kernel(
    x_ptr, out_ptr, idx_ptr, stride_xm,
    D: tl.constexpr, GROUP: tl.constexpr, ROW_BYTES: tl.constexpr, FMT: tl.constexpr, HAS_IDX: tl.constexpr,
):
    """One program per input row ``m``; writes packed bytes to pool row ``idx[m]`` (or ``m``)."""
    m = tl.program_id(0).to(tl.int64)
    if HAS_IDX:
        row = tl.load(idx_ptr + m).to(tl.int64)
    else:
        row = m
    dst = out_ptr + row * ROW_BYTES
    src = x_ptr + m * stride_xm

    if FMT == 0:
        offs_d = tl.arange(0, D)
        bits = tl.load(src + offs_d).to(tl.bfloat16).to(tl.uint16, bitcast=True)
        tl.store(dst + 2 * offs_d, (bits & 0xFF).to(tl.uint8))
        tl.store(dst + 2 * offs_d + 1, (bits >> 8).to(tl.uint8))
    elif FMT == 1:
        NG: tl.constexpr = D // GROUP
        offs_d = tl.arange(0, D)
        offs_g = tl.arange(0, NG)
        x = tl.reshape(tl.load(src + offs_d).to(tl.float32), [NG, GROUP])
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        e = _log2_ceil(amax * (1.0 / 448.0))
        s = tl.exp2(e.to(tl.float32))
        q = tl.clamp(tl.math.div_rn(x, s[:, None]), -448.0, 448.0)
        tl.store(dst + offs_d, tl.reshape(_e4m3_codes(q), [D]))
        tl.store(dst + D + offs_g, (e + 127).to(tl.uint8))
    else:
        # e2m1 pairs: even channels from one strided load, odd from another, so a byte is (lo | hi << 4)
        NG: tl.constexpr = D // GROUP
        HALF: tl.constexpr = GROUP // 2
        offs_h = tl.arange(0, D // 2)
        offs_g = tl.arange(0, NG)
        xe = tl.reshape(tl.load(src + 2 * offs_h).to(tl.float32), [NG, HALF])
        xo = tl.reshape(tl.load(src + 2 * offs_h + 1).to(tl.float32), [NG, HALF])
        amax = tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1))
        if FMT == 2:
            amax = tl.maximum(amax, 6.0 * (2.0 ** -9))
            s = _e4m3_round(amax * (1.0 / 6.0))
            tl.store(dst + D // 2 + offs_g, _e4m3_codes(s))
        else:
            amax = tl.maximum(amax, 6.0 * (2.0 ** -126))
            e = _log2_ceil(amax * (1.0 / 6.0))
            s = tl.exp2(e.to(tl.float32))
            tl.store(dst + D // 2 + offs_g, (e + 127).to(tl.uint8))
        # IEEE division: the reference's CUDA ``x / s`` is correctly rounded, and with an e4m3
        # (non power-of-two) scale triton's default approximate div would land a hair off the
        # e2m1 midpoints and flip their round-to-even
        qe = fp4_f32_to_nibble(_round_fp4(tl.clamp(tl.math.div_rn(xe, s[:, None]), -6.0, 6.0)))
        qo = fp4_f32_to_nibble(_round_fp4(tl.clamp(tl.math.div_rn(xo, s[:, None]), -6.0, 6.0)))
        tl.store(dst + offs_h, tl.reshape(qe | (qo << 4), [D // 2]))


@triton.jit
def _unpack_rows_kernel(
    pool_ptr, idx_ptr, out_ptr, M,
    D: tl.constexpr, ROW_BYTES: tl.constexpr, FMT: tl.constexpr, BLOCK_T: tl.constexpr, HAS_IDX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = offs_t < M
    if HAS_IDX:
        rows = tl.load(idx_ptr + offs_t, mask=valid, other=0)
    else:
        rows = offs_t
    offs_d = tl.arange(0, D)
    vals = load_rows(pool_ptr, rows, valid, offs_d, D, ROW_BYTES, FMT)
    tl.store(
        out_ptr + offs_t[:, None].to(tl.int64) * D + offs_d[None, :],
        vals.to(out_ptr.dtype.element_ty), mask=valid[:, None],
    )


def pack_rows(
    x: torch.Tensor, fmt: RowFormat, pool: torch.Tensor | None = None, row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize ``x [M, D]`` (bf16 / fp16 / fp32) into packed rows.

    With ``pool [rows, row_bytes] uint8`` and ``row_ids [M]``, row ``m`` is written to ``pool[row_ids[m]]``
    (the KV write path). Without a pool a fresh ``[M, row_bytes]`` uint8 tensor is returned.
    """
    assert x.dim() == 2, x.shape
    m, d = x.shape
    fmt.validate_dim(d)
    row_bytes = fmt.row_bytes(d)
    assert d & (d - 1) == 0 and d >= 16, f"pack_rows wants a power-of-two D >= 16, got {d}"
    if pool is None:
        assert row_ids is None
        pool = torch.empty((m, row_bytes), dtype=torch.uint8, device=x.device)
    else:
        assert pool.dtype == torch.uint8 and pool.shape[1] == row_bytes and pool.is_contiguous(), (pool.shape, row_bytes)
        assert row_ids is not None and row_ids.shape == (m,), (None if row_ids is None else row_ids.shape, m)
    if m == 0:
        return pool
    if x.stride(1) != 1:
        x = x.contiguous()
    _pack_rows_kernel[(m,)](
        x, pool, row_ids if row_ids is not None else pool, x.stride(0),
        D=d, GROUP=fmt.group or d, ROW_BYTES=row_bytes, FMT=fmt.code, HAS_IDX=row_ids is not None,
        num_warps=4,
    )
    return pool


def unpack_rows(
    pool: torch.Tensor, fmt: RowFormat, dim: int, row_ids: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather ``pool[row_ids]`` (or every row) and dequantize to ``[M, dim]`` in ``out_dtype``."""
    fmt.validate_dim(dim)
    assert pool.dtype == torch.uint8 and pool.shape[1] == fmt.row_bytes(dim) and pool.is_contiguous(), pool.shape
    m = pool.shape[0] if row_ids is None else row_ids.numel()
    out = torch.empty((m, dim), dtype=out_dtype, device=pool.device)
    if m == 0:
        return out
    block_t = 16
    _unpack_rows_kernel[(triton.cdiv(m, block_t),)](
        pool, row_ids if row_ids is not None else pool, out, m,
        D=dim, ROW_BYTES=fmt.row_bytes(dim), FMT=fmt.code, BLOCK_T=block_t, HAS_IDX=row_ids is not None,
        num_warps=4,
    )
    return out


def quant_roundtrip(x: torch.Tensor, fmt: RowFormat) -> torch.Tensor:
    """``unpack(pack(x))`` -- the value the cache will hand back; the reference's ``inplace=True`` semantics."""
    if fmt is BF16:
        return x.to(torch.bfloat16)
    return unpack_rows(pack_rows(x, fmt), fmt, x.shape[1], out_dtype=torch.bfloat16)


__all__ = ["pack_rows", "unpack_rows", "quant_roundtrip"]
