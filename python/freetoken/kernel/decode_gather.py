"""JIT launch for the opt-in TMA decode gather. The switch lives in ``freetoken.decode_gather``."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from freetoken.decode_gather import TUNED_DECODE_GATHER

from .fast_index_copy import _skip_fast_index_copy_enabled
from .utils import CppArgList, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


def module_spec(config: tuple) -> tuple[str, CppArgList, str]:
    """(load_jit name, template args, launch wrapper) for one gather config."""
    kind, *targs = config
    if kind != "tma" or len(targs) != 3:
        raise ValueError(f"unknown decode gather config {config!r}")
    args = make_cpp_args(*targs)
    return "decode_gather_tma", args, f"&DecodeGatherTmaKernel<{args}>::run"


@lru_cache(maxsize=None)
def _jit_decode_gather_module(config: tuple) -> Module:
    stem, args, wrapper = module_spec(config)
    return load_jit(
        stem,
        *args,
        cuda_files=["decode_gather.cuh"],
        cuda_wrappers=[("launch", wrapper)],
    )


def decode_gather_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    config: tuple = TUNED_DECODE_GATHER,
) -> None:
    """Same arguments and bytes as ``fast_index_copy_multi_jit``.

    The miss count is read on the device, so the launch stays CUDA-graph
    capturable. At most 8 banks; row sizes are multiples of 16 bytes.
    """
    if _skip_fast_index_copy_enabled():
        return
    module = _jit_decode_gather_module(tuple(config))
    module.launch(dst_ptrs, src_ptrs, feat_bytes, dst_indices, src_indices, num_indices)
