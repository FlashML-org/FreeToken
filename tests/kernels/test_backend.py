import importlib.util

import pytest
import torch

import freetoken.kernel.backend as backend
import freetoken.moe.nvfp4_backends as nvfp4_backends


_CUDA_ONLY_PROBES = (
    backend.is_flashinfer_installed,
    backend.is_sgl_kernel_installed,
    backend.is_triton_kernels_installed,
)


def _clear_probe_caches() -> None:
    for probe in _CUDA_ONLY_PROBES:
        probe.cache_clear()
    backend.driver_cuda_version.cache_clear()


def test_rocm_never_selects_cuda_only_backends(monkeypatch):
    def unexpected_probe(_name: str) -> bool:
        raise AssertionError("unexpected CUDA-only package probe on ROCm")

    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(backend, "_importable", unexpected_probe)
    _clear_probe_caches()

    assert all(not probe() for probe in _CUDA_ONLY_PROBES)
    assert backend.driver_cuda_version() is None

    _clear_probe_caches()


def test_cuda_keeps_optional_package_probes(monkeypatch):
    monkeypatch.setattr(backend, "is_rocm", lambda: False)
    monkeypatch.setattr(backend, "_importable", lambda _name: True)
    _clear_probe_caches()

    assert all(probe() for probe in _CUDA_ONLY_PROBES)

    _clear_probe_caches()


def _reject_cuda_donor_probe(*_args, **_kwargs):
    raise AssertionError("unexpected CUDA-only donor probe on ROCm")


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_rocm_nvfp4_uses_triton_without_cuda_donor_probes(monkeypatch, requested):
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", _reject_cuda_donor_probe)
    monkeypatch.setattr(nvfp4_backends, "_donor_symbols_ok", _reject_cuda_donor_probe)

    selected = nvfp4_backends.select_nvfp4_backend(
        torch.device("cuda"), intermediate_size=1024, requested=requested
    )

    assert selected == "triton"


@pytest.mark.parametrize("requested", ["marlin", "flashinfer"])
def test_rocm_nvfp4_rejects_forced_cuda_backend(monkeypatch, requested):
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", _reject_cuda_donor_probe)
    monkeypatch.setattr(nvfp4_backends, "_donor_symbols_ok", _reject_cuda_donor_probe)

    with pytest.raises(RuntimeError, match="CUDA-only and unavailable on ROCm"):
        nvfp4_backends.select_nvfp4_backend(
            torch.device("cuda"), intermediate_size=1024, requested=requested
        )


def test_rocm_b12x_reason_short_circuits_cuda_probes(monkeypatch):
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", _reject_cuda_donor_probe)
    monkeypatch.setattr(nvfp4_backends, "_donor_symbols_ok", _reject_cuda_donor_probe)

    assert nvfp4_backends._b12x_unusable_reason((12, 0)) == (
        "flashinfer b12x is CUDA-only and unavailable on ROCm"
    )
