"""Exact-size pinned host tensors (e.g. offload expert banks).

The offload gather kernel (``fast_index_copy``) reads host memory zero-copy from the
GPU, so allocations must be pinned + device-mapped. We avoid
``torch.empty(pin_memory=True)`` because its caching allocator rounds sizes up to the
next power of two (a 70GB bank would reserve 128GB).

On HIP, if the ``_pinned_tensor`` extension was not built, ``host_register`` /
``device_ptr`` talk to libamdhip64 via ctypes. A silent no-op here is what caused
the RDNA4 offload-decode TDR (FlashML-org/FreeToken#122): the gather kernel
dereferenced unregistered host VAs.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_pinned_extension():
    try:
        return importlib.import_module("freetoken.kernel._pinned_tensor")
    except ImportError:
        return None


def _require_extension():
    ext = _load_pinned_extension()
    if ext is None:
        raise ImportError(
            "freetoken.kernel._pinned_tensor is not installed. Reinstall FreeToken "
            "so the pinned tensor CUDA/HIP extension is built at install time, or "
            "on AMD use the HIP ctypes fallback (host_register / device_ptr) "
            "documented in docs/amd.md."
        )
    return ext


def _hip_pin_memory_fallback() -> bool:
    """Exact-size hipHostMalloc needs the extension; skip-ext HIP may use pin_memory."""
    return getattr(torch.version, "hip", None) is not None


def create_pinned_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """Create a CPU pinned tensor with the same size, stride, and dtype as input."""
    ext = _load_pinned_extension()
    if ext is not None:
        return ext.create_pinned_tensor_like(input)
    if _hip_pin_memory_fallback():
        return torch.empty_like(input, pin_memory=True)
    return _require_extension().create_pinned_tensor_like(input)


def copy_to_pinned_tensor(input: torch.Tensor) -> torch.Tensor:
    """Copy a CPU tensor into exact-size cudaMallocHost / hipHostMalloc pinned storage."""

    output = create_pinned_tensor_like(input)
    with torch.no_grad():
        output.copy_(input)
    return output


def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate an exact-size, uninitialized pinned host tensor via cudaHostAlloc."""
    ext = _load_pinned_extension()
    if ext is not None:
        return ext.alloc_pinned_tensor(list(shape), dtype)
    if _hip_pin_memory_fallback():
        return torch.empty(*shape, dtype=dtype, pin_memory=True)
    return _require_extension().alloc_pinned_tensor(list(shape), dtype)


@lru_cache(maxsize=1)
def _hip_runtime():
    """ctypes HIP runtime when the extension is absent (Linux: libamdhip64.so)."""
    if getattr(torch.version, "hip", None) is None:
        return None
    from freetoken.runtime.gpu import load_hip_runtime

    return load_hip_runtime()


def host_register(addr: int, nbytes: int) -> None:
    """Register ``nbytes`` at ``addr`` as portable+mapped (pin-after-fill).

    Without the extension this used to be a silent no-op, which left expert banks
    pageable — the fused offload gather then died with `unspecified launch failure`
    (RDNA4 TDR, #122). On ROCm, register through the HIP runtime instead.
    """
    ext = _load_pinned_extension()
    if ext is not None:
        ext.host_register(addr, nbytes)
        return
    hip = _hip_runtime()
    if hip is not None:
        import ctypes

        # hipHostRegisterPortable (1) | hipHostRegisterMapped (2)
        status = hip.hipHostRegister(
            ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), ctypes.c_uint(3)
        )
        if status != 0:
            raise RuntimeError(
                f"hipHostRegister({nbytes} bytes) failed with hipError {status}"
            )
        return
    raise RuntimeError(
        "host_register requires the _pinned_tensor extension (NVIDIA) or a HIP "
        "runtime (AMD). Reinstall FreeToken, or see docs/amd.md."
    )


@lru_cache(maxsize=1)
def _host_ptr_identity() -> bool:
    ext = _load_pinned_extension()
    if ext is not None:
        return bool(ext.host_ptr_identity())
    hip = _hip_runtime()
    if hip is None:
        return False
    import ctypes
    import mmap

    # Probe with hipHostREGISTERed memory — how the expert banks are pinned.
    # On Windows/WDDM+ROCm, registered memory maps to a different device address
    # even though hipHostMalloc / torch pin_memory is unified (#122).
    buf = mmap.mmap(-1, 4096)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    if hip.hipHostRegister(
        ctypes.c_void_p(addr), ctypes.c_size_t(4096), ctypes.c_uint(3)
    ) != 0:
        return False
    try:
        dev = ctypes.c_void_p()
        ok = (
            hip.hipHostGetDevicePointer(
                ctypes.byref(dev), ctypes.c_void_p(addr), ctypes.c_uint(0)
            )
            == 0
        )
        return bool(ok and dev.value == addr)
    finally:
        hip.hipHostUnregister(ctypes.c_void_p(addr))


def device_ptr(t: torch.Tensor) -> int:
    """Base address of ``t`` as the GPU must dereference it.

    Equals ``data_ptr()`` on CUDA tensors and wherever pinned host memory is
    device-visible at its host VA (Linux/UVA). Where registered memory maps to a
    different device address, zero-copy consumers must use this, not
    ``data_ptr()``. Host tensors must be pinned+mapped.
    """
    if t.is_cuda or _host_ptr_identity():
        return t.data_ptr()
    ext = _load_pinned_extension()
    if ext is not None:
        return ext.host_device_ptr(t.data_ptr())
    hip = _hip_runtime()
    if hip is not None:
        import ctypes

        dev = ctypes.c_void_p()
        status = hip.hipHostGetDevicePointer(
            ctypes.byref(dev), ctypes.c_void_p(t.data_ptr()), ctypes.c_uint(0)
        )
        if status != 0:
            raise RuntimeError(f"hipHostGetDevicePointer failed with hipError {status}")
        if dev.value is None:
            raise RuntimeError("hipHostGetDevicePointer returned a null device pointer")
        return int(dev.value)
    return t.data_ptr()
