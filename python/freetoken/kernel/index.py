from __future__ import annotations

import functools
import sys
import warnings
from typing import TYPE_CHECKING, Tuple

import torch

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
_TORCH_FALLBACK_KEYS: set[tuple[int, int]] = set()


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def num_splits_for(element_size: int) -> int:
    """Split factor for a row of ``element_size`` bytes; also used by the AOT
    shape table (kernel/aot_models.py), which must reproduce it exactly."""
    if element_size % 2048 == 0:
        return 4
    if element_size % 1024 == 0:
        return 2
    return 1


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    num_splits = num_splits_for(element_size)
    key = (element_size, num_splits)
    module = None
    if key not in _TORCH_FALLBACK_KEYS:
        try:
            module = _jit_index_module(element_size, num_splits=num_splits)
        except RuntimeError as exc:
            if sys.platform != "win32":
                raise
            _TORCH_FALLBACK_KEYS.add(key)
            warnings.warn(
                f"Falling back to torch.index_select for {element_size}-byte embedding rows "
                f"because the Windows CUDA index kernel is unavailable: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    if module is not None:
        module.launch(weights, indices, output, vocab_range)
        return output

    if vocab_range is None:
        torch.index_select(weights, 0, indices.to(torch.int64), out=output)
        return output

    start, length = vocab_range
    valid = (indices >= start) & (indices < start + length)
    local_indices = (indices - start).clamp(0, max(0, length - 1)).to(torch.int64)
    torch.index_select(weights, 0, local_indices, out=output)
    output.masked_fill_(~valid[:, None], 0)
    return output
