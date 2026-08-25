"""tinygrad-JIT fallback for FFI kernels that are not hand-ported to HIP.

FreeToken's hand-written tvm-ffi kernels (``store`` / ``index`` / ``fast_index_copy`` /
``batch_memcpy``) are CUDA source compiled via nvcc/JIT. The primary AMD port is the
``#if defined(USE_HIP)`` seam in ``device_api.h`` + ``LaunchKernel``/``warp.cuh``. This
module is the **documented fallback** for any kernel that proves intractable to hipify:
tinygrad's JIT compiles one logical kernel to PTX (CUDA) *and* AMDGPU/LLVM (ROCm), so the
same source covers both platforms.

Constraints (matching the FFI contract):

* Each fallback takes the same ``tvm.ffi.TensorView`` arguments as the hand-written
  kernel and returns the same output tensor(s), so the swap is invisible to callers.
* It runs on the *host* (tinygrad handles GPU dispatch); on ROCm it compiles to AMDGPU.
* It is **never a default**: ``kernel/utils.py`` only routes a kernel to the fallback
  when (a) ROCm is active and (b) the hand-HIP AOT/JIT variant is absent/unbuildable.
  If tinygrad is not installed, invoking the fallback raises a clear error.

Because tinygrad is an optional dependency (installed only when the fallback is actually
needed), all imports here are lazy and the module imports with zero third-party deps, so
it is safe to import on the CUDA-only path.
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = [
    "is_tinygrad_available",
    "kernel_fallback_available",
    "get_kernel_fallback",
]

# Kernel names the fallback registry knows how to build (mirrors the FFI kernel set).
_KNOWN_KERNELS = ("store", "index", "fast_index_copy", "batch_memcpy")

#: Which kernels currently have a *functional* tinygrad reimplementation. As HIP ports
#: land in Inc 7, names are removed from this set (the hand port wins); kernels left here
#: (if any) are the documented fallback set. Default: empty -- the hand-HIP port is the
#: primary path and the fallback is opt-in per kernel.
_FALLBACK_IMPLEMENTED: set[str] = set()


def is_tinygrad_available() -> bool:
    """True when the ``tinygrad`` package can be imported (JIT-to-ROCm available)."""
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("tinygrad") is not None
    except Exception:
        return False


def kernel_fallback_available(kernel: str) -> bool:
    """True when a tinygrad fallback for ``kernel`` is both implemented and usable
    (tinygrad installed). Always False on the CUDA path unless explicitly enabled, so
    the CUDA build never depends on tinygrad."""
    if kernel not in _FALLBACK_IMPLEMENTED:
        return False
    return is_tinygrad_available()


def get_kernel_fallback(kernel: str):
    """Return the tinygrad-backed fallback callable for ``kernel``, or raise a clear
    error explaining why it is unavailable. Never called on the CUDA path."""
    if kernel not in _FALLBACK_IMPLEMENTED:
        raise RuntimeError(
            f"FFI kernel {kernel!r} has no tinygrad fallback registered. On ROCm the "
            "preferred path is the hand-written HIP port (device_api.h); if you intend "
            "to use the tinygrad fallback you must register it in "
            "kernel/tinygrad_fallback.py._FALLBACK_IMPLEMENTED and implement the "
            "corresponding build function."
        )
    if not is_tinygrad_available():
        raise RuntimeError(
            f"FFI kernel {kernel!r} requires the tinygrad fallback, but tinygrad is not "
            "installed. Install it (`pip install tinygrad`) or provide a hand-written "
            "HIP port for this kernel."
        )
    from freetoken.kernel import tinygrad_impl  # noqa: PLC0415  (lazy; may be None)

    builder = getattr(tinygrad_impl, f"build_{kernel}", None)
    if builder is None:
        raise RuntimeError(
            f"tinygrad fallback for {kernel!r} is registered but has no "
            "tinygrad_impl.build_<kernel>() builder."
        )
    return builder


def _list_fallbacks() -> list[str]:
    return [k for k in _FALLBACK_IMPLEMENTED if kernel_fallback_available(k)]
