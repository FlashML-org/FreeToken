import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a CUDA or ROCm GPU is required",
)


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return (x.float() * inv * weight.float()).to(x.dtype)


@pytest.mark.parametrize("shape", [(2, 64), (2, 3, 64)])
def test_rmsnorm_matches_pytorch_reference(shape):
    from freetoken.kernel.triton.norm import rmsnorm

    torch.manual_seed(11)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    output = rmsnorm(x, weight, eps=1e-5)
    expected = _reference_rmsnorm(x, weight, 1e-5)

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_fused_add_rmsnorm_updates_both_tensors_from_reference():
    from freetoken.kernel.triton.norm import fused_add_rmsnorm

    torch.manual_seed(12)
    x = torch.randn((3, 64), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    expected_residual = residual.float() + x.float()
    weight = torch.randn(64, device="cuda", dtype=x.dtype)
    expected_x = _reference_rmsnorm(expected_residual.to(x.dtype), weight, 1e-5)

    fused_add_rmsnorm(x, residual, weight, eps=1e-5)

    torch.testing.assert_close(residual, expected_residual.to(residual.dtype), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x, expected_x, rtol=2e-2, atol=2e-2)
