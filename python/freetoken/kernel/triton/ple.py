# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (python/sglang/srt/models/qwen4_exp.py)
"""UVA row gather for the Qwen3.8-Flash-Next PLE n-gram table.

The table (320,001,536 rows x 160, FP8-e4m3 + one scalar scale = 47.7 GiB) stays in pinned
host memory and the GPU dereferences it in place over PCIe -- at its host VA on Linux/UVA, at
the mapped device address on WDDM (``kernel/pinned.device_ptr``). One program per requested
row: read the row, widen to fp32, apply the per-tensor scale, store bf16.

NVFP4 checkpoints pack the rows as e2m1 (two per byte) plus a per-16 e4m3 scale row and one
global scale: the gather reads both rows, decodes the nibbles, and multiplies in place.

Ids outside the table store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

# Latency-bound over PCIe, so keep the block small and let many of them be in flight.
_NUM_WARPS = 1


@triton.jit
def _e2m1_f32(n: tl.uint8) -> tl.float32:
    """The 16 e2m1 nibbles: 0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6 (sign = bit 3)."""
    n = n.to(tl.int32)
    mant = (n & 1).to(tl.float32)
    exp2 = (n >> 1) & 3
    # exponent 0 is subnormal (0.mant); otherwise (1.mant) * 2 ** (exp2 - 1)
    mag = tl.where(exp2 == 0, mant * 0.5, (1.0 + mant * 0.5) * tl.where(exp2 == 1, 1.0, tl.where(exp2 == 2, 2.0, 4.0)))
    sign = ((n >> 3) & 1) * mag
    return mag - 2 * sign


@triton.jit
def _ple_gather_kernel(
    table_ptr,
    scale_ptr,
    ids_ptr,
    out_ptr,
    scale,
    num_rows,
    EMB_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    IS_FP4: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM
    # the table is a host allocation: rebuild the typed pointer from the raw address
    if IS_FP8:
        if e4m3_native_cx():
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            # pre-sm_89 has no fp8e4nv type: load raw bytes and decode in software
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            values = e4m3_u8_to_f32(tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0))
    elif IS_FP4:
        # packed e2m1: two values per byte (low nibble first), per-16 e4m3 scale row, global scale
        pack_offsets = tl.arange(0, BLOCK_H)
        vbytes = tl.load(table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8)) + idx * (EMB_DIM // 2) + pack_offsets,
                         mask=pack_offsets < EMB_DIM // 2, other=0)
        lo = _e2m1_f32(vbytes & 0x0F)
        hi = _e2m1_f32((vbytes >> 4) & 0x0F)
        values = tl.join(lo, hi).reshape([BLOCK_D])
        scale_offsets = tl.arange(0, BLOCK_S)
        scale_mask = scale_offsets < EMB_DIM // 16
        if e4m3_native_cx():
            scale_base = scale_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            svals = tl.load(scale_base + idx * (EMB_DIM // 16) + scale_offsets, mask=scale_mask, other=0.0).to(tl.float32)
        else:
            scale_base = scale_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            svals = e4m3_u8_to_f32(tl.load(scale_base + idx * (EMB_DIM // 16) + scale_offsets, mask=scale_mask, other=0))
        svals = tl.broadcast_to(svals.reshape([BLOCK_S, 1]), [BLOCK_S, 16]).reshape([BLOCK_D])
        values = values * svals
    else:
        base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.where(in_range, values * scale, 0.0)
    tl.store(
        out_ptr + row * EMB_DIM + offsets,
        values.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_gather_rows(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
    scale_ptr: int | None = None,
) -> torch.Tensor:
    """Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), embed_dim]``
    bf16 on the same device. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``. Packed e2m1 tables
    (``is_fp4``) pass the per-16 e4m3 scale row bank as ``scale_ptr`` and the global scale as
    ``scale``.
    """
    assert not (is_fp8 and scale_ptr is not None), "a packed table is not also a native fp8 table"
    if scale_ptr is not None:
        is_fp8 = False
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _ple_gather_kernel[(n,)](
            table_ptr,
            scale_ptr if scale_ptr is not None else table_ptr,
            row_ids,
            out,
            float(scale),
            num_rows,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8,
            IS_FP4=scale_ptr is not None,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            BLOCK_H=triton.next_power_of_2(embed_dim // 2),
            BLOCK_S=triton.next_power_of_2(embed_dim // 16),
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["ple_gather_rows"]
