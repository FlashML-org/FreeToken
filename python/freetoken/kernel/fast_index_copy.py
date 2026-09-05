from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


DEFAULT_NUM_BLOCKS = 4
SKIP_FAST_INDEX_COPY_ENV = "FREETOKEN_SKIP_FAST_INDEX_COPY"
FUSED_COPY_BLOCKS_PER_BANK_ENV = "FREETOKEN_FUSED_COPY_BLOCKS_PER_BANK"
_TRUE_VALUES = {"1", "true", "yes", "on"}
# The legacy per-bank C++ kernel issues one 128-byte vectorized transaction per
# worker iteration. The fused multi-bank path has a tail-aware implementation
# and supports every 16-byte-aligned bank row, but this legacy specialization
# cannot represent a 240- or 400-byte worker row.
_LEGACY_COPY_ITERATION_BYTES = 128


def fused_copy_blocks_per_bank() -> int:
    """Return the AOT-compiled fused-copy grid width selected for one cache bank.

    Eight blocks is the established default. Sixty-four blocks widens the same
    vector-copy grid without changing indices, source bytes, destination slots,
    or arithmetic. Restricting the setting to the two cache-built variants keeps
    strict no-JIT launches reproducible on gfx1151.
    """

    raw = os.getenv(FUSED_COPY_BLOCKS_PER_BANK_ENV, "8")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{FUSED_COPY_BLOCKS_PER_BANK_ENV} must be 8 or 64, got {raw!r}") from exc
    if value not in (8, 64):
        raise ValueError(f"{FUSED_COPY_BLOCKS_PER_BANK_ENV} must be 8 or 64, got {value}")
    return value


def _skip_fast_index_copy_enabled() -> bool:
    return os.getenv(SKIP_FAST_INDEX_COPY_ENV, "").strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=None)
def _jit_update_flag_module() -> Module:
    return load_jit(
        "fast_index_copy_update_flag",
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("update_copy_flag", "update_copy_flag")],
    )


@lru_cache(maxsize=None)
def _jit_fast_index_copy_module(
    *,
    feature_size: int,
    worker_threads: int,
    worker_feature_size: int,
    num_block: int,
) -> Module:
    args = make_cpp_args(
        feature_size,
        worker_threads,
        worker_feature_size,
        1024,
        num_block,
        1,
    )
    return load_jit(
        "fast_index_copy",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[
            ("launch", f"&FastIndexCopyKernel<{args}>::run"),
            ("launch_high", f"&FastIndexCopyKernel<{args}>::run_high"),
            ("launch_normal", f"&FastIndexCopyKernel<{args}>::run_normal"),
        ],
    )


def _default_worker_threads(feature_size: int) -> int:
    if feature_size <= 1024:
        return 8
    if feature_size <= 2048:
        return 16
    return 32


def _shrink_worker_feature_size(feature_size: int, worker_feature_size: int) -> int:
    if feature_size < worker_feature_size:
        worker_feature_size = feature_size
    while feature_size % worker_feature_size != 0 and worker_feature_size > 128:
        worker_feature_size //= 2
    return worker_feature_size


def default_worker_args(feature_size: int) -> tuple[int, int, int]:
    """(worker_threads, worker_feature_size, num_block) of the default call path.

    The AOT spec builder (kernel/aot.py) compiles with exactly these values, so a
    default ``fast_index_copy_jit`` call hits the prebuilt cache by name.
    """
    return (
        _default_worker_threads(feature_size),
        _shrink_worker_feature_size(feature_size, 2048),
        DEFAULT_NUM_BLOCKS,
    )


def legacy_fast_index_copy_is_supported(feature_size: int) -> bool:
    """Return whether the legacy per-bank template can represent ``feature_size``.

    ``FastIndexCopyKernel`` has no scalar tail: every worker copies an integral
    number of 128-byte transactions. The fused multi-bank production path does
    support smaller 16-byte-aligned rows, so this predicate only controls AOT
    generation for the unused legacy fallback. Keeping the condition beside the
    runtime argument derivation prevents the cache catalog from emitting a HIP
    specialization that fails its own compile-time assertion.
    """

    _, worker_feature_size, _ = default_worker_args(feature_size)
    return worker_feature_size % _LEGACY_COPY_ITERATION_BYTES == 0


def fast_index_copy_jit(
    dst: torch.Tensor,
    dst_indices: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    worker_threads: int | None = None,
    worker_feature_size: int = 2048,
    num_block: int | None = None,
    priority: str | None = None,
    sync_flag: torch.Tensor | None = None,
) -> None:
    num_dst_feature = math.prod(dst.shape[1:])
    num_src_feature = math.prod(src.shape[1:])
    assert num_src_feature == num_dst_feature

    # Debug/perf ablation: keep miss bookkeeping intact, but make the copy free to
    # approximate a zero-copy-miss runtime. Outputs are only meaningful if callers
    # already have valid cache contents for the requested indices.
    if _skip_fast_index_copy_enabled():
        return

    dst = dst.as_strided(size=(dst.size(0), num_dst_feature), stride=(num_dst_feature, 1))
    src = src.as_strided(size=(src.size(0), num_src_feature), stride=(num_src_feature, 1))

    feature_size = dst.size(-1) * dst.element_size()
    num_block = num_block or DEFAULT_NUM_BLOCKS
    worker_threads = worker_threads or _default_worker_threads(feature_size)
    worker_feature_size = _shrink_worker_feature_size(feature_size, worker_feature_size)
    assert worker_threads in (8, 16, 32)
    assert feature_size % worker_feature_size == 0

    module = _jit_fast_index_copy_module(
        feature_size=feature_size,
        worker_threads=worker_threads,
        worker_feature_size=worker_feature_size,
        num_block=num_block,
    )
    if priority is None:
        module.launch(dst, dst_indices, src, src_indices, num_indices)
        return

    assert priority in ("high", "normal")
    assert sync_flag is not None
    if priority == "high":
        module.launch_high(dst, dst_indices, src, src_indices, num_indices, sync_flag)
        return
    module.launch_normal(dst, dst_indices, src, src_indices, num_indices, sync_flag)


@lru_cache(maxsize=None)
def _jit_fast_index_copy_multi_module(*, num_threads: int, blocks_per_bank: int) -> Module:
    args = make_cpp_args(num_threads, blocks_per_bank)
    return load_jit(
        "fast_index_copy_multi",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("launch", f"&MultiIndexCopyKernel<{args}>::run")],
    )


def fast_index_copy_multi_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    num_threads: int = 1024,
    blocks_per_bank: int | None = None,
) -> None:
    """Fused multi-bank index copy: copy the same rows for every bank in ONE launch.

    The copy is host->device PCIe-bandwidth bound (~31 GB/s measured, vs ~3 TB/s HBM), so what
    saturates the link is the TOTAL in-flight request count, blocks_per_bank*num_threads
    (~4096 threads/bank is the knee; blocks and threads are interchangeable -- 8x1024 ==
    16x512 == 4x1024 all measure the same across small/medium/large-expert workloads). PCIe
    is the bottleneck, not the GPU, so few blocks already feed it; the split only matters at
    the extreme (~100k+ threads bumps the zero-copy launch overhead). 8x1024 = 8192/bank sits
    ~2x past the knee. Launch count is one regardless of grid width, so the zero-copy launch-
    overhead win (and a strict >= over the legacy per-bank kernel across all workloads) holds.

    ``dst_ptrs``/``src_ptrs``/``feat_bytes`` are int64 [num_banks] device tensors built
    once by the caller (the per-bank slot-cache base addr, host-source base addr, and
    per-row byte size). Every bank's per-row byte size must be a multiple of 16, and the
    base addresses 16-byte aligned (true for contiguous torch allocations of these banks).
    """
    if _skip_fast_index_copy_enabled():
        return
    selected_blocks = fused_copy_blocks_per_bank() if blocks_per_bank is None else blocks_per_bank
    if selected_blocks not in (8, 64):
        raise ValueError(f"blocks_per_bank must be 8 or 64, got {selected_blocks}")
    module = _jit_fast_index_copy_multi_module(
        num_threads=num_threads, blocks_per_bank=selected_blocks
    )
    module.launch(dst_ptrs, src_ptrs, feat_bytes, dst_indices, src_indices, num_indices)


def update_copy_flag_jit(sync_flag: torch.Tensor, delta: int) -> None:
    assert sync_flag.is_cuda
    assert sync_flag.numel() == 1
    assert sync_flag.dtype == torch.int32
    _jit_update_flag_module().update_copy_flag(sync_flag, delta)
