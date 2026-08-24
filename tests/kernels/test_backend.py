import freetoken.kernel.backend as backend


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
