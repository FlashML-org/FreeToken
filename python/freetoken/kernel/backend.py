"""Availability probes for the optional native kernel packages.

When flashinfer / sgl_kernel are installed the call-sites use their fused CUDA
ops; otherwise they fall back to the pure-Triton kernels in
``freetoken.kernel.triton``. ``find_spec`` only checks that the package is
importable (no import side effects), and the result is cached.
"""
from __future__ import annotations

import functools
import importlib.util


def _importable(name: str) -> bool:
    # find_spec normally returns None when a package is absent, but it can raise
    # (broken parent package, or a meta_path finder that blocks the name); treat
    # any failure as "not available" so callers cleanly fall back to triton.
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@functools.cache
def is_flashinfer_installed() -> bool:
    return _importable("flashinfer")


@functools.cache
def is_sgl_kernel_installed() -> bool:
    return _importable("sgl_kernel")


@functools.cache
def driver_cuda_version() -> int | None:
    """Max CUDA version the installed NVIDIA driver supports (``13000`` == CUDA 13.0),
    or None if undetermined. Driver-JIT kernels (PTX compiled at runtime, e.g.
    flashinfer's CuTe-DSL paths) are gated by this, not by any package's build-time
    toolkit version. Resolved through the ``_pinned_tensor`` extension's link-time
    cudart, so it works wherever the extension builds (including Windows) -- no dlopen
    by soname."""
    try:
        from freetoken.kernel.pinned import _load_pinned_extension

        version = int(_load_pinned_extension().driver_cuda_version())
    except Exception:
        return None
    return version or None  # 0 == no driver installed
