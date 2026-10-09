"""Numerical checks for the built-in activation and norm fallback paths."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)


@pytest.fixture(autouse=True)
def _use_triton_fallbacks(monkeypatch):
    from freetoken.kernel import backend

    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: False)
    monkeypatch.setattr(backend, "is_sgl_kernel_installed", lambda: False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_activation_fallback_matches_torch(dtype, activation):
    from freetoken.layers.activation import gated_act_and_mul

    torch.manual_seed(0)
    x = torch.randn((7, 256), device="cuda", dtype=dtype)
    out = torch.full((7, 128), float("nan"), device="cuda", dtype=dtype)
    gate, up = x.float().chunk(2, dim=-1)
    if activation == "silu":
        expected = torch.nn.functional.silu(gate) * up
    else:
        approx = "tanh" if activation == "gelu_tanh" else "none"
        expected = torch.nn.functional.gelu(gate, approximate=approx) * up

    gated_act_and_mul(activation, x, out)

    # CUDA intentionally uses approximate math (notably tanh.approx.f32),
    # while ROCm uses libdevice; allow for both against the Torch reference.
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-3)
    torch.testing.assert_close(out, expected.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["rmsnorm", "gemma"])
def test_norm_fallback_matches_torch(dtype, kind):
    from freetoken.layers.norm import GemmaRMSNorm, RMSNorm

    torch.manual_seed(0)
    norm = (RMSNorm if kind == "rmsnorm" else GemmaRMSNorm)(128, 1e-6)
    norm.weight = torch.randn(128, device="cuda", dtype=dtype)
    x = torch.randn((7, 128), device="cuda", dtype=dtype)
    xf = x.float()
    expected = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    expected *= norm.weight.float()

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-3)
    torch.testing.assert_close(norm.forward(x), expected.to(dtype), rtol=rtol, atol=atol)
