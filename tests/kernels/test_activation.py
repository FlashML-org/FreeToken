"""Triton *_and_mul activation kernels vs torch references.

Guards the GELU_TANH fallback in particular: `tanh.approx.f32` is an sm_75+ PTX
instruction, so pre-Turing takes a libdevice path that must agree with it.
"""

import pytest
import torch


def _xy(rows, d, dtype):
    torch.manual_seed(0)
    x = torch.randn(rows, 2 * d, device="cuda", dtype=dtype)
    return x, x[:, :d].float(), x[:, d:].float()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("d", [128, 1000])
def test_silu_and_mul_matches_torch(dtype, d):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from freetoken.kernel.triton.activation import silu_and_mul

    x, gate, up = _xy(17, d, dtype)
    ref = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(silu_and_mul(x).float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("d", [128, 1000])
def test_gelu_and_mul_matches_torch(dtype, d):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from freetoken.kernel.triton.activation import gelu_and_mul

    x, gate, up = _xy(17, d, dtype)
    ref = torch.nn.functional.gelu(gate, approximate="none") * up
    torch.testing.assert_close(gelu_and_mul(x).float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("d", [128, 1000])
def test_gelu_tanh_and_mul_matches_torch(dtype, d):
    """The sm_75+ `tanh.approx.f32` path and the libdevice fallback must agree here."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from freetoken.kernel.triton.activation import gelu_tanh_and_mul

    x, gate, up = _xy(17, d, dtype)
    ref = torch.nn.functional.gelu(gate, approximate="tanh") * up
    torch.testing.assert_close(gelu_tanh_and_mul(x).float(), ref, atol=2e-2, rtol=2e-2)


def test_gelu_tanh_saturates_at_both_tails():
    """tanh must saturate to +-1 rather than overflow: gelu_tanh(x) -> x for large +x
    and -> 0 for large -x. An exp-based fallback that overflows would produce nan."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from freetoken.kernel.triton.activation import gelu_tanh_and_mul

    gate = torch.tensor([-1e4, -60.0, -8.0, 0.0, 8.0, 60.0, 1e4], device="cuda")
    x = torch.cat([gate, torch.ones_like(gate)]).reshape(1, -1)
    out = gelu_tanh_and_mul(x).float().flatten()
    assert torch.isfinite(out).all(), out
    ref = torch.nn.functional.gelu(gate, approximate="tanh")
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swigluoai_and_mul_matches_reference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from freetoken.kernel.triton.activation import swigluoai_and_mul

    alpha, limit = 1.702, 7.0
    x, gate, up = _xy(17, 256, dtype)
    g = gate.clamp(max=limit)
    u = up.clamp(-limit, limit)
    ref = g * torch.sigmoid(alpha * g) * (u + 1.0)
    got = swigluoai_and_mul(x, alpha=alpha, limit=limit).float()
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
