import pytest

import freetoken.kernel.backend as backend
import freetoken.layers.quantization.moe.nvfp4 as nvfp4
from freetoken.layers.quantization import KernelSelectionError, select_kernel
from freetoken.layers.quantization.moe import MoEConfig, Nvfp4MoEMethod


_CUDA_ONLY_PROBES = (
    backend.is_flashinfer_installed,
    backend.is_sgl_kernel_installed,
    backend.is_vllm_installed,
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


def _nvfp4_moe_config() -> MoEConfig:
    return MoEConfig(
        num_experts=128,
        hidden=4096,
        intermediate=1536,
        top_k=8,
        strategy="offload",
    )


def test_rocm_nvfp4_auto_uses_triton_without_cuda_donor_probes(monkeypatch):
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(backend, "_importable", _reject_cuda_donor_probe)
    monkeypatch.setattr(backend, "device_capability", lambda: (12, 0))
    monkeypatch.setattr(nvfp4, "_marlin_symbols_ok", _reject_cuda_donor_probe)
    monkeypatch.setattr(nvfp4, "_b12x_symbols_ok", _reject_cuda_donor_probe)
    _clear_probe_caches()

    selected = select_kernel(
        Nvfp4MoEMethod.candidates,
        "auto",
        _nvfp4_moe_config(),
    )

    assert selected.name == "triton"
    _clear_probe_caches()


@pytest.mark.parametrize("requested", ["marlin", "b12x"])
def test_rocm_nvfp4_rejects_forced_cuda_kernel(monkeypatch, requested):
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(backend, "_importable", _reject_cuda_donor_probe)
    monkeypatch.setattr(backend, "device_capability", lambda: (12, 0))
    monkeypatch.setattr(nvfp4, "_marlin_symbols_ok", _reject_cuda_donor_probe)
    monkeypatch.setattr(nvfp4, "_b12x_symbols_ok", _reject_cuda_donor_probe)
    _clear_probe_caches()

    with pytest.raises(KernelSelectionError, match="was requested but cannot run"):
        select_kernel(
            Nvfp4MoEMethod.candidates,
            requested,
            _nvfp4_moe_config(),
        )

    _clear_probe_caches()
