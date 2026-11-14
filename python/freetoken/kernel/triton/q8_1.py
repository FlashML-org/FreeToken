"""ggml's q8_1 activation blocks written from a Triton kernel, so the kernel that makes an activation also makes the copy the ggml vector kernel reads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def store_q8_1(x, n, q_ptr, row, col, K: tl.constexpr, Q_BLOCKS: tl.constexpr):
    """Write the first ``n`` values of ``x`` as columns ``col .. col + n`` of row ``row`` in the ``empty_q8_1`` buffer of a ``[rows, K]`` tensor, byte for byte what ggml's quantize_q8_1 writes. ``n`` and ``col`` are multiples of 32."""
    NB: tl.constexpr = x.shape[0] // 32
    PAD_BYTES: tl.constexpr = (Q_BLOCKS - K // 32) * 36
    q_row = q_ptr + row * (Q_BLOCKS * 36)
    x = x.to(tl.float32)
    xb = tl.reshape(x, (NB, 32))
    amax = tl.max(tl.abs(xb), axis=1)
    d = amax / 127.0
    q = tl.where(amax[:, None] == 0.0, 0.0, libdevice.round(xb / d[:, None])).to(tl.int8)
    # ggml adds in the order of its warp shuffle: halves 16 apart, then 8, 4, 2, 1
    s = tl.sum(tl.reshape(x, (NB, 2, 16)), axis=1)
    s = tl.sum(tl.reshape(s, (NB, 2, 8)), axis=1)
    s = tl.sum(tl.reshape(s, (NB, 2, 4)), axis=1)
    s = tl.sum(tl.reshape(s, (NB, 2, 2)), axis=1)
    s = tl.sum(s, axis=1)
    blk = tl.arange(0, NB)
    keep = blk < n // 32
    # block_q8_1 is {half d, half sum, int8 qs[32]}
    ds_ptr = q_row.to(tl.pointer_type(tl.float16)) + (col // 32 + blk) * 18
    tl.store(ds_ptr, d.to(tl.float16), mask=keep)
    tl.store(ds_ptr + 1, s.to(tl.float16), mask=keep)
    qs_ptr = q_row.to(tl.pointer_type(tl.int8)) + (col // 32 + blk[:, None]) * 36 + 4 + tl.arange(0, 32)[None, :]
    tl.store(qs_ptr, q, mask=keep[:, None])
    if PAD_BYTES > 0:
        # ggml pads the row with zero blocks; the call that writes the last columns writes them
        PAD_SIZE: tl.constexpr = triton.next_power_of_2(PAD_BYTES)
        offs = tl.arange(0, PAD_SIZE)
        tl.store(q_row + (K // 32) * 36 + offs, tl.zeros([PAD_SIZE], tl.uint8), mask=(offs < PAD_BYTES) & (col + n == K))


def empty_q8_1(y: torch.Tensor, to, unit: int) -> tuple[torch.Tensor | None, int]:
    """Buffer for the q8_1 copy of ``y [rows, K]`` and its blocks per row if the linear ``to`` reads ``y`` that way, else ``(None, 0)``. ``unit`` is the ``n`` that the kernel passes to ``store_q8_1``."""
    if to is None or to.input_format(y) != "q8_1" or unit % 32:
        return None, 0
    from freetoken.kernel import ggml

    row_bytes = ggml.q8_1_row_bytes(y.shape[1])
    return torch.empty(y.shape[0], row_bytes, dtype=torch.uint8, device=y.device), row_bytes // 36


__all__ = ["empty_q8_1", "store_q8_1"]
