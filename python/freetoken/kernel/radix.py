from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from .utils import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> "Module":
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality; return the index of the first
    # differing element (prefix match length). On a GPU-less box we avoid the
    # AOT/ninja build and compute it with plain torch.
    if not torch.cuda.is_available():
        n = min(x.numel(), y.numel())
        if n == 0:
            return 0
        eq = x[:n] == y[:n]
        diff = (~eq).int().argmax().item()
        # if every compared element matches, the diff index points at a True only
        # when a mismatch exists; handle the all-equal case explicitly.
        if eq.all().item():
            return n
        return int(diff)
    return _load_radix_module().fast_compare_key(x, y)
