"""Regression coverage for the CUDA-namespace compatibility boundary on ROCm.

PyTorch exposes AMD devices through ``torch.cuda`` so CUDA-oriented Python
programs can run on HIP.  FreeToken must not mistake a ``gfx11xx`` capability
for a newer NVIDIA SM capability, nor select optional CUDA binaries merely
because a stale package happens to be installed in the environment.
"""

import torch

from freetoken.kernel import backend
from freetoken.utils import arch


def test_rocm_never_satisfies_nvidia_sm_gates(monkeypatch):
    """HIP hardware is excluded before numerical NVIDIA capability comparison."""
    monkeypatch.setattr(torch.version, "hip", "7.15")
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (11, 5))
    arch.is_rocm_runtime.cache_clear()
    arch._get_torch_cuda_version.cache_clear()

    try:
        assert arch.is_rocm_runtime() is True
        assert arch._get_torch_cuda_version() is None
        assert arch.is_sm90_supported() is False
        assert arch.is_sm100_supported() is False
    finally:
        # Cached runtime detection must not leak the synthetic HIP state into
        # unrelated test modules that run later in the same interpreter.
        arch.is_rocm_runtime.cache_clear()
        arch._get_torch_cuda_version.cache_clear()


def test_rocm_disables_cuda_only_optional_backends(monkeypatch):
    """Triton remains available, while CUDA binary packages are bypassed on HIP."""
    monkeypatch.setattr(backend, "is_rocm_runtime", lambda: True)
    monkeypatch.setattr(backend, "_importable", lambda _name: True)
    backend.is_flashinfer_installed.cache_clear()
    backend.is_sgl_kernel_installed.cache_clear()
    backend.is_triton_kernels_installed.cache_clear()
    backend.driver_cuda_version.cache_clear()

    try:
        assert backend.is_flashinfer_installed() is False
        assert backend.is_sgl_kernel_installed() is False
        assert backend.is_triton_kernels_installed() is False
        assert backend.driver_cuda_version() is None
    finally:
        backend.is_flashinfer_installed.cache_clear()
        backend.is_sgl_kernel_installed.cache_clear()
        backend.is_triton_kernels_installed.cache_clear()
        backend.driver_cuda_version.cache_clear()
