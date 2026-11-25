"""Gather expert rows of a narrow GPU shard out of wider host bank rows.

Each bank's narrow row is the bytes of ``ByteRuns`` (``layers/quantization/moe/shard.py``)
inside the wide source row: ``count`` runs of ``length`` bytes, ``stride`` apart from ``offset``.
``shard_copy`` moves scattered rows (decode misses) for all banks in one Triton launch; the rows
come as device-side ``(dst_ids, src_ids, num)`` like the fused slot copy, so it is fixed-shape and
CUDA-graph capturable. ``memcpy2d`` moves whole layers (prefill) on the copy engines instead: a
layer's runs are one regular 2D copy per bank, and the copy engines leave the SMs to the prefill
GEMMs it overlaps.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

_WORD_TYPES = {8: tl.uint64, 4: tl.uint32, 2: tl.uint16, 1: tl.uint8}


@triton.jit
def _shard_copy_kernel(
    dst_ptrs, src_ptrs, dst_row, src_row, run_offset, run_length, run_stride, run_zero,
    dst_ids, src_ids, num,
    WORD: tl.constexpr, BLOCK: tl.constexpr,
):
    bank = tl.program_id(1)
    dst = tl.load(dst_ptrs + bank).to(tl.pointer_type(WORD))
    src = tl.load(src_ptrs + bank).to(tl.pointer_type(WORD))
    drow = tl.load(dst_row + bank)
    srow = tl.load(src_row + bank)
    offset = tl.load(run_offset + bank)
    length = tl.load(run_length + bank)
    stride = tl.load(run_stride + bank)
    keep = tl.load(run_zero + bank) == 0
    chunks = tl.cdiv(drow, BLOCK)
    total = tl.load(num) * chunks
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    for tile in range(tl.program_id(0), total, tl.num_programs(0)):
        entry = tile // chunks
        k = (tile % chunks) * BLOCK + lanes
        mask = k < drow
        s = tl.load(src_ids + entry).to(tl.int64)
        d = tl.load(dst_ids + entry).to(tl.int64)
        values = tl.load(src + s * srow + offset + (k // length) * stride + k % length, mask=mask & keep, other=0)
        tl.store(dst + d * drow + k, values, mask=mask)


def copy_word(*sizes: int) -> int:
    """The widest word (8, 4, 2 or 1 bytes) every byte size / address in ``sizes`` is a multiple of."""
    return next(w for w in (8, 4, 2, 1) if all(s % w == 0 for s in sizes))


def shard_copy(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    runs: torch.Tensor,
    dst_ids: torch.Tensor,
    src_ids: torch.Tensor,
    num: torch.Tensor,
    *,
    word: int,
    programs: int = 64,
) -> None:
    """Copy rows ``src_ids[:num]`` of every bank's wide source to rows ``dst_ids[:num]`` of its narrow destination.

    ``runs`` is ``[6, banks]`` int64 in units of ``word`` bytes: destination row, source row,
    run offset, run length, run stride, and the zero flag (a narrow row that is all zeros)."""
    banks = dst_ptrs.numel()
    if banks == 0:
        return
    _shard_copy_kernel[(programs, banks)](
        dst_ptrs, src_ptrs, runs[0], runs[1], runs[2], runs[3], runs[4], runs[5],
        dst_ids, src_ids, num,
        WORD=_WORD_TYPES[word], BLOCK=1024, num_warps=4,
    )


@lru_cache(maxsize=None)
def _memcpy2d_module():
    from freetoken.kernel.utils import load_jit

    return load_jit("memcpy2d", cuda_files=["memcpy2d.cuh"], cuda_wrappers=[("memcpy2d", "&Memcpy2D::run")])


def memcpy2d(desc: torch.Tensor, stream: int) -> None:
    """Enqueue one 2D copy per row of ``desc`` (CPU int64 ``[n, 6]``: dst, dst pitch, src, src pitch,
    width, height; src 0 zeroes the rows) on the raw ``stream`` handle."""
    _memcpy2d_module().memcpy2d(desc, stream)


__all__ = ["copy_word", "memcpy2d", "shard_copy"]
