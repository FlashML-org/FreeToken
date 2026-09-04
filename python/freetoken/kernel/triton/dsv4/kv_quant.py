"""Quantizing store for the DSV4 window / compressed KV pools.

One program per row: read the bf16 row, take a max-abs per :data:`BLOCK`, and write the
fp8 bytes plus the fp16 scales into the pool at the row's slot. Replaces the plain
``index_copy_`` the unquantized pool does.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import round_e4m3
from freetoken.kvcache.dsv4_kv_quant import BLOCK, MAX_MAG, scale_width


@triton.jit
def _store_kv_quant_kernel(
    src_ptr, dst_ptr, sc_ptr, slot_ptr,
    stride_sm, stride_dm, stride_cm,
    D: tl.constexpr, NB: tl.constexpr, QBLOCK: tl.constexpr, MAXMAG: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slot_ptr + row).to(tl.int64)

    # Address the row as [NB, QBLOCK] directly rather than loading [D] and reshaping:
    # the block axis is the reduction axis for the max-abs, and triton reshapes from 1-D
    # are not free to express.
    offs2 = tl.arange(0, NB)[:, None] * QBLOCK + tl.arange(0, QBLOCK)[None, :]
    xb = tl.load(src_ptr + row * stride_sm + offs2).to(tl.float32)  # [NB, QBLOCK]

    amax = tl.max(tl.abs(xb), axis=1)
    # An all-zero block quantizes to zeros under any positive scale; 1.0 keeps the
    # division finite.
    scale = tl.where(amax > 0, amax / MAXMAG, 1.0)
    # Round the scale to its stored precision before dividing, so the value written here
    # and the value the attention kernel reads back are scaled by the identical number.
    scale = scale.to(tl.float16).to(tl.float32)

    # div_rn, not `/`: the plain operator may lower to a reciprocal multiply, which
    # disagrees with the torch reference on values sitting exactly between two steps.
    q = tl.math.div_rn(xb, scale[:, None])
    q = tl.minimum(tl.maximum(q, -MAXMAG), MAXMAG)
    # round_e4m3 before the cast: triton lowers fp32 -> float8e4nv as a double-round
    # (fp32 -> fp16 RTZ -> e4m3), which is one-sided toward zero. Putting the value on
    # the grid in fp32 first makes the cast exact.
    q = round_e4m3(q)

    tl.store(dst_ptr + slot * stride_dm + offs2, q.to(tl.float8e4nv))
    tl.store(sc_ptr + slot * stride_cm + tl.arange(0, NB), scale.to(tl.float16))


def store_kv_quant(
    pool: torch.Tensor,       # [slots, D] float8_e4m3fn
    scales: torch.Tensor,     # [slots, D//BLOCK] fp16
    slots: torch.Tensor,      # [rows] int
    kv: torch.Tensor,         # [rows, D] compute dtype
) -> None:
    """Quantize ``kv`` into ``pool``/``scales`` at ``slots``."""
    rows, d = kv.shape
    assert pool.shape[1] == d and pool.dtype == torch.float8_e4m3fn, (pool.shape, pool.dtype)
    nb = scale_width(d)
    assert scales.shape[1] == nb, (scales.shape, nb)
    if rows == 0:
        return
    kv = kv.contiguous()
    slots = slots.to(torch.int64).contiguous()
    _store_kv_quant_kernel[(rows,)](
        kv, pool, scales, slots,
        kv.stride(0), pool.stride(0), scales.stride(0),
        D=d, NB=nb, QBLOCK=BLOCK, MAXMAG=MAX_MAG,
        num_warps=4,
    )


__all__ = ["store_kv_quant"]
