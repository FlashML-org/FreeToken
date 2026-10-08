from types import SimpleNamespace

import pytest
import torch

import freetoken.kernel.backend as backend
import freetoken.layers.quantization.moe.nvfp4 as nvfp4
from freetoken.layers.quantization import KernelSelectionError, select_kernel
from freetoken.layers.quantization.linear import LinearConfig, Nvfp4LinearMethod
from freetoken.layers.quantization.moe import MoEConfig, Nvfp4MoEMethod


_CUDA_ONLY_PROBES = (
    backend.is_flashinfer_installed,
    backend.is_sgl_kernel_installed,
    backend.is_vllm_installed,
)
_CACHED_PROBES = (*_CUDA_ONLY_PROBES, backend.driver_cuda_version, backend.device_capability, backend.is_rocm)


def _clear_probe_caches() -> None:
    for probe in _CACHED_PROBES:
        probe.cache_clear()


@pytest.fixture(autouse=True)
def _isolated_probe_caches():
    _clear_probe_caches()
    yield
    _clear_probe_caches()


@pytest.mark.parametrize("hip_version, expected", [(None, False), ("7.14.60850", True)])
def test_rocm_detection_uses_torch_build(monkeypatch, hip_version, expected):
    monkeypatch.setattr(torch.version, "hip", hip_version)
    assert backend.is_rocm() is expected


def test_rocm_never_selects_cuda_only_backends(monkeypatch):
    import freetoken.kernel.pinned as pinned

    def unexpected_probe(_name: str) -> bool:
        pytest.fail("unexpected CUDA-only package probe on ROCm")

    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(backend, "_importable", unexpected_probe)
    monkeypatch.setattr(pinned, "_load_pinned_extension", _reject_cuda_donor_probe)

    assert all(not probe() for probe in _CUDA_ONLY_PROBES)
    assert backend.driver_cuda_version() is None


@pytest.mark.parametrize("available", [False, True])
def test_cuda_keeps_optional_package_probes(monkeypatch, available):
    queried = []

    def probe(name):
        queried.append(name)
        return available

    monkeypatch.setattr(backend, "is_rocm", lambda: False)
    monkeypatch.setattr(backend, "_importable", probe)

    for _ in range(2):
        assert [probe() for probe in _CUDA_ONLY_PROBES] == [available] * 3
    assert queried == ["flashinfer", "sgl_kernel", "vllm"]


@pytest.mark.parametrize("value, expected", [(13000, 13000), (0, None), (None, None)])
def test_cuda_driver_probe_retains_result_and_failure_handling(monkeypatch, value, expected):
    import freetoken.kernel.pinned as pinned

    def load():
        if value is None:
            raise ImportError("extension unavailable")
        return SimpleNamespace(driver_cuda_version=lambda: value)

    monkeypatch.setattr(backend, "is_rocm", lambda: False)
    monkeypatch.setattr(pinned, "_load_pinned_extension", load)
    assert backend.driver_cuda_version() == expected


@pytest.mark.parametrize("error", [ImportError, ValueError, RuntimeError])
def test_broken_cuda_package_discovery_is_unavailable(monkeypatch, error):
    def broken_spec(_name):
        raise error("broken optional package")

    monkeypatch.setattr(backend, "is_rocm", lambda: False)
    monkeypatch.setattr(backend.importlib.util, "find_spec", broken_spec)
    assert not any(probe() for probe in _CUDA_ONLY_PROBES)


def _reject_cuda_donor_probe(*_args, **_kwargs):
    pytest.fail("unexpected CUDA-only donor probe on ROCm")


def _block_rocm_donors(monkeypatch):
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(backend, "_importable", _reject_cuda_donor_probe)
    monkeypatch.setattr(backend, "device_capability", _reject_cuda_donor_probe)
    monkeypatch.setattr(nvfp4, "_marlin_symbols_ok", _reject_cuda_donor_probe)
    monkeypatch.setattr(nvfp4, "_b12x_symbols_ok", _reject_cuda_donor_probe)


def _nvfp4_moe_config() -> MoEConfig:
    return MoEConfig(
        num_experts=128,
        hidden=4096,
        intermediate=1536,
        top_k=8,
        strategy="offload",
    )


@pytest.mark.parametrize("requested", ["auto", "triton"])
@pytest.mark.parametrize("layer", ["moe", "linear"])
def test_rocm_nvfp4_uses_triton_without_cuda_donor_probes(monkeypatch, requested, layer):
    _block_rocm_donors(monkeypatch)
    method, config = (
        (Nvfp4MoEMethod, _nvfp4_moe_config()) if layer == "moe"
        else (Nvfp4LinearMethod, LinearConfig(4096, 1536))
    )

    selected = select_kernel(method.candidates, requested, config)

    assert selected.name == "triton"


@pytest.mark.parametrize(("layer", "requested"), [("moe", "marlin"), ("moe", "b12x"), ("linear", "marlin")])
def test_rocm_nvfp4_rejects_forced_cuda_kernel(monkeypatch, layer, requested):
    _block_rocm_donors(monkeypatch)
    method, config = (
        (Nvfp4MoEMethod, _nvfp4_moe_config()) if layer == "moe"
        else (Nvfp4LinearMethod, LinearConfig(4096, 1536))
    )

    with pytest.raises(KernelSelectionError, match="CUDA-only.*ROCm"):
        select_kernel(method.candidates, requested, config)


@pytest.mark.parametrize(("cc", "requested", "expected"), [
    ((8, 0), "auto", "triton"),
    ((9, 0), "auto", "triton"),
    ((12, 0), "auto", "triton"),
    ((8, 0), "marlin", "marlin"),
    ((9, 0), "marlin", "marlin"),
    ((12, 0), "b12x", "b12x"),
    ((9, 0), "triton", "triton"),
])
def test_cuda_nvfp4_moe_selection_is_preserved(monkeypatch, cc, requested, expected):
    monkeypatch.setattr(backend, "is_rocm", lambda: False)
    monkeypatch.setattr(backend, "_importable", lambda _name: True)
    monkeypatch.setattr(backend, "device_capability", lambda: cc)
    monkeypatch.setattr(backend, "driver_cuda_version", lambda: 13000)
    monkeypatch.setattr(nvfp4, "_marlin_symbols_ok", lambda: True)
    monkeypatch.setattr(nvfp4, "_b12x_symbols_ok", lambda: True)
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda _name: object())

    selected = select_kernel(Nvfp4MoEMethod.candidates, requested, _nvfp4_moe_config())
    assert selected.name == expected


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_rocm_activation_fallback_matches_torch(monkeypatch, dtype, activation):
    from freetoken.layers.activation import gated_act_and_mul

    _block_rocm_donors(monkeypatch)
    torch.manual_seed(0)
    x = torch.randn((7, 256), device="cuda", dtype=dtype)
    out = torch.empty((7, 128), device="cuda", dtype=dtype)
    gate, up = x.float().chunk(2, dim=-1)
    if activation == "silu":
        expected = torch.nn.functional.silu(gate) * up
    else:
        approx = "tanh" if activation == "gelu_tanh" else "none"
        expected = torch.nn.functional.gelu(gate, approximate=approx) * up

    gated_act_and_mul(activation, x, out)
    torch.testing.assert_close(out, expected.to(dtype), rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["rmsnorm", "gemma"])
def test_rocm_norm_fallback_matches_torch(monkeypatch, dtype, kind):
    from freetoken.layers.norm import GemmaRMSNorm, RMSNorm

    _block_rocm_donors(monkeypatch)
    torch.manual_seed(0)
    norm = (RMSNorm if kind == "rmsnorm" else GemmaRMSNorm)(128, 1e-6)
    norm.weight = torch.randn(128, device="cuda", dtype=dtype)
    x = torch.randn((7, 128), device="cuda", dtype=dtype)
    xf = x.float()
    expected = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    expected *= norm.weight.float()

    torch.testing.assert_close(norm.forward(x), expected.to(dtype), rtol=1e-2, atol=1e-3)
