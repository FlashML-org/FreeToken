from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    from freetoken.utils.arch import is_rocm

    # ROCm torch has no torch.cuda.nvtx; mapping to roctx is future work. Under ROCm we
    # pass through (no-op decorator) so AMD runs are not coupled to NVIDIA-only tooling.
    if is_rocm():
        def passthrough(fn):
            return fn

        return passthrough

    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


def graph_capture():
    """Context manager that captures a CUDA (or, on ROCm, HIP) graph on the current
    stream via ``torch.cuda.graph``. Correctness is validated once by the Inc-1
    ``freetoken.utils.graph_gate`` probe; callers rely on that result."""
    import torch

    return torch.cuda.graph()
