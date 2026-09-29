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


class _MappedHostArray:
    """``__cuda_array_interface__`` over a pinned+mapped host tensor, so torch aliases it as a
    CUDA tensor without a copy; the alias holds this object, and this object holds the host."""

    def __init__(self, host: torch.Tensor) -> None:
        self.host = host
        self.__cuda_array_interface__ = {
            "shape": tuple(host.shape),
            "strides": None,
            # bfloat16 has no typestr: export the same-width integer and view the dtype back
            "typestr": f"<i{host.element_size()}",
            "data": (device_ptr(host), False),
            "version": 3,
        }


def mapped_cuda_view(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A CUDA tensor over pinned+mapped host memory: kernels read and write it in place over
    PCIe, and it keeps ``host`` alive for as long as it or any view of it lives."""
    with torch.cuda.device(device):
        # Inside the device context: torch reports no pointer as pinned before its CUDA context exists.
        assert host.is_pinned() and host.is_contiguous(), "only contiguous pinned+mapped host memory aliases"
        return torch.as_tensor(_MappedHostArray(host), device=device).view(host.dtype)


def device_ptr(t: torch.Tensor) -> int:
    """Base address of ``t`` as the GPU must dereference it.

    Equals ``data_ptr()`` on CUDA tensors and wherever pinned host memory is
    device-visible at its host VA (Linux/UVA). On Windows/WDDM registered memory maps
    to a different device address, so zero-copy consumers must use this, not
    ``data_ptr()``. Host tensors must be pinned+mapped."""
    if t.is_cuda or _host_ptr_identity():
        return t.data_ptr()
    return _load_pinned_extension().host_device_ptr(t.data_ptr())
