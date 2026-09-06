"""Exact-size pinned host tensors (e.g. offload expert banks).

The offload gather kernel (``fast_index_copy``) reads host memory zero-copy from the
GPU, so allocations must be pinned + device-mapped. We avoid
``torch.empty(pin_memory=True)`` because its caching allocator rounds sizes up to the
next power of two (a 70GB bank would reserve 128GB)."""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_pinned_extension():
    try:
        return importlib.import_module("freetoken.kernel._pinned_tensor")
    except ImportError as exc:
        if getattr(torch.version, "hip", None) is not None:
            # A source checkout may not have built the packaged extension yet.  HIP's
            # runtime API provides the same registration and address-translation
            # primitives, so ROCm can retain the safe mapped-memory contract.
            return None
        raise ImportError(
            "freetoken.kernel._pinned_tensor is not installed. Reinstall FreeToken "
            "so the pinned tensor CUDA extension is built at install time."
        ) from exc


def create_pinned_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """Create a CPU pinned tensor with the same size, stride, and dtype as input."""
    extension = _load_pinned_extension()
    if extension is None:
        return torch.empty_like(input, pin_memory=True)
    return extension.create_pinned_tensor_like(input)


def copy_to_pinned_tensor(input: torch.Tensor) -> torch.Tensor:
    """Copy a CPU tensor into exact-size cudaMallocHost pinned storage."""

    output = create_pinned_tensor_like(input)
    with torch.no_grad():
        output.copy_(input)
    return output


def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate an exact-size, uninitialized pinned host tensor via cudaHostAlloc."""
    extension = _load_pinned_extension()
    if extension is None:
        return torch.empty(*shape, dtype=dtype, pin_memory=True)
    return extension.alloc_pinned_tensor(list(shape), dtype)


@lru_cache(maxsize=1)
def _hip_runtime():
    """Return the loaded HIP runtime for source runs without the C++ extension."""
    if getattr(torch.version, "hip", None) is None:
        return None

    import ctypes

    for name in ("amdhip64_7.dll", "amdhip64.dll", "libamdhip64.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def host_register(addr: int, nbytes: int) -> None:
    """cudaHostRegister ``nbytes`` at ``addr`` as portable+mapped (pin-after-fill)."""
    extension = _load_pinned_extension()
    if extension is not None:
        extension.host_register(addr, nbytes)
        return

    hip = _hip_runtime()
    if hip is None:
        raise RuntimeError("ROCm runtime unavailable for mapped host registration")

    import ctypes

    # hipHostRegisterPortable | hipHostRegisterMapped
    status = hip.hipHostRegister(
        ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), ctypes.c_uint(3)
    )
    if status != 0:
        raise RuntimeError(f"hipHostRegister({nbytes} bytes) failed with hipError {status}")


@lru_cache(maxsize=1)
def _host_ptr_identity() -> bool:
    # cached per process: FreeToken pins one CUDA device per process (set at engine launch)
    extension = _load_pinned_extension()
    if extension is not None:
        return bool(extension.host_ptr_identity())

    hip = _hip_runtime()
    if hip is None:
        return False

    import ctypes
    import mmap

    # Probe the same mmap + hipHostRegister path used by HostBank.  On WDDM its
    # device-visible address can differ from the CPU virtual address.
    buf = mmap.mmap(-1, 4096)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    registered = (
        hip.hipHostRegister(
            ctypes.c_void_p(addr), ctypes.c_size_t(4096), ctypes.c_uint(3)
        )
        == 0
    )
    if not registered:
        buf.close()
        return False
    try:
        device = ctypes.c_void_p()
        translated = (
            hip.hipHostGetDevicePointer(
                ctypes.byref(device), ctypes.c_void_p(addr), ctypes.c_uint(0)
            )
            == 0
        )
        return translated and device.value == addr
    finally:
        hip.hipHostUnregister(ctypes.c_void_p(addr))
        buf.close()


def device_ptr(t: torch.Tensor) -> int:
    """Base address of ``t`` as the GPU must dereference it.

    Equals ``data_ptr()`` on CUDA tensors and wherever pinned host memory is
    device-visible at its host VA (Linux/UVA). On Windows/WDDM registered memory maps
    to a different device address, so zero-copy consumers must use this, not
    ``data_ptr()``. Host tensors must be pinned+mapped."""
    if t.is_cuda or _host_ptr_identity():
        return t.data_ptr()
    extension = _load_pinned_extension()
    if extension is not None:
        return extension.host_device_ptr(t.data_ptr())

    hip = _hip_runtime()
    if hip is None:
        raise RuntimeError("ROCm runtime unavailable for host-device pointer translation")

    import ctypes

    device = ctypes.c_void_p()
    status = hip.hipHostGetDevicePointer(
        ctypes.byref(device), ctypes.c_void_p(t.data_ptr()), ctypes.c_uint(0)
    )
    if status != 0 or not device.value:
        raise RuntimeError(f"hipHostGetDevicePointer failed with hipError {status}")
    return int(device.value)
