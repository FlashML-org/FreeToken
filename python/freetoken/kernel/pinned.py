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
        raise ImportError(
            "freetoken.kernel._pinned_tensor is not installed. Reinstall FreeToken "
            "so the pinned tensor CUDA extension is built at install time."
        ) from exc


def create_pinned_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """Create a CPU pinned tensor with the same size, stride, and dtype as input."""

    return _load_pinned_extension().create_pinned_tensor_like(input)


def copy_to_pinned_tensor(input: torch.Tensor) -> torch.Tensor:
    """Copy a CPU tensor into exact-size cudaMallocHost pinned storage."""

    output = create_pinned_tensor_like(input)
    with torch.no_grad():
        output.copy_(input)
    return output


def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate an exact-size, uninitialized pinned host tensor via cudaHostAlloc."""

    return _load_pinned_extension().alloc_pinned_tensor(list(shape), dtype)


def host_register(addr: int, nbytes: int) -> None:
    """cudaHostRegister ``nbytes`` at ``addr`` as portable+mapped (pin-after-fill)."""
    _load_pinned_extension().host_register(addr, nbytes)


@lru_cache(maxsize=1)
def _host_ptr_identity() -> bool:
    # cached per process: FreeToken pins one CUDA device per process (set at engine launch)
    return bool(_load_pinned_extension().host_ptr_identity())


def device_ptr(t: torch.Tensor) -> int | None:
    """Base address of ``t`` as the GPU must dereference it.

    Equals ``data_ptr()`` on CUDA tensors and wherever pinned host memory is
    device-visible at its host VA (Linux/UVA). On Windows/WDDM registered memory maps
    to a different device address, so zero-copy consumers must use this, not
    ``data_ptr()``. Host tensors must be pinned+mapped.

    Returns ``None`` when the host memory is not pinned+mapped (e.g. a bank whose
    cudaHostRegister failed and was downgraded to PAGEABLE). The caller treats
    ``None`` as "no device alias" and routes through the pageable copy path."""
    if t.is_cuda:
        return t.data_ptr()
    if _host_ptr_identity():
        return t.data_ptr()
    try:
        return _load_pinned_extension().host_device_ptr(t.data_ptr())
    except RuntimeError:
        return None
