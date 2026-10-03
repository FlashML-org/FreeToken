# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (python/sglang/srt/models/qwen4_exp.py)
"""UVA row gather for the Qwen3.8-Flash-Next PLE n-gram table.

The table (320,001,536 rows x 160, FP8-e4m3 + one scalar scale = 47.7 GiB) stays in pinned
host memory and the GPU dereferences it in place over PCIe -- at its host VA on Linux/UVA, at
the mapped device address on WDDM (``kernel/pinned.device_ptr``). One program per requested
row: read the row, widen to fp32, apply the per-tensor scale, store bf16. NVFP4 checkpoints
pack the same rows two codes per byte with one fp8 block scale per 16 elements; that path
unpacks through the E2M1 LUT and applies block scale * global scale.

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
def _ple_gather_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    scale,
    num_rows,
    scale_ptr,
    lut_ptr,
    num_groups,
    EMB_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    IS_NVFP4: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    if IS_NVFP4:
        # NVFP4 row: one byte per two e2m1 codes (low nibble first) + one fp8 block scale
        # per 16 elements (8 bytes); the scalar `scale` is the checkpoint global scale.
        # Row offsets are int64: at 320M rows the byte offsets pass int32.
        half = EMB_DIM // 2
        offs_b = tl.arange(0, BLOCK_D // 2)
        mask_b = offs_b < half
        base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        packed = tl.load(base + idx * half + offs_b, mask=mask_b, other=0).to(tl.int32)
        if e4m3_native_cx():
            sb = scale_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            blk = tl.load(sb + idx * num_groups + offs_b // 8, mask=mask_b, other=0.0).to(tl.float32)
        else:
            sb = scale_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            blk = e4m3_u8_to_f32(tl.load(sb + idx * num_groups + offs_b // 8, mask=mask_b, other=0))
        g = blk * scale
        lo = tl.where(in_range, tl.load(lut_ptr + (packed & 0xF)) * g, 0.0)
        hi = tl.where(in_range, tl.load(lut_ptr + ((packed >> 4) & 0xF)) * g, 0.0)
        out_base = out_ptr + row.to(tl.int64) * EMB_DIM
        tl.store(out_base + 2 * offs_b, lo.to(out_ptr.dtype.element_ty), mask=mask_b)
        tl.store(out_base + 2 * offs_b + 1, hi.to(out_ptr.dtype.element_ty), mask=mask_b)
        return
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
    scales_ptr: int | None = None,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), embed_dim]``
    bf16 on the same device. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``.

    With ``scales_ptr`` (device-addressable fp8 block-scale bank, ``[num_rows,
    embed_dim // 16]``) and a device ``lut`` of the 16 E2M1 values, the packed table at
    ``table_ptr`` is read as NVFP4 and ``scale`` is its per-tensor global scale.
    """
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    assert (scales_ptr is None) == (lut is None), "NVFP4 gather needs both scales and lut"
    if n:
        _ple_gather_kernel[(n,)](
            table_ptr,
            row_ids,
            out,
            float(scale),
            num_rows,
            # constexpr-false arguments still need a plausibly typed operand; unused there
            scales_ptr if scales_ptr is not None else 0,
            lut if lut is not None else row_ids,
            embed_dim // 16,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8 and scales_ptr is None,  # the NVFP4 branch supersedes fp8
            IS_NVFP4=scales_ptr is not None,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["ple_gather_rows"]
