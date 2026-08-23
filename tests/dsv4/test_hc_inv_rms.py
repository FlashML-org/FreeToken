"""``inv_rms`` must agree with the expression it replaced, and be no less accurate.

The old form was ``torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)`` over the
fp32 upcast. ``inv_rms`` reduces the bf16 source directly: the upcast is exact, so the
per-element squares are identical and only the reduction order differs. Bit-equality is
not asserted -- ATen's order varies with M and is not reproducible -- so the contract
is (a) agreement to fp32 rounding and (b) accuracy no worse than ATen's against fp64.
"""
import pytest
import torch

from freetoken.kernel.triton.dsv4.norm import inv_rms

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel")

HC_DIM = 4 * 4096  # hc_mult * dim at DSV4-Flash
EPS = 1e-6


def _old(xf, eps):
    return torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 7, 512])
def test_matches_square_mean(dtype, rows):
    dev = "cuda"
    x = torch.randn(rows, HC_DIM, device=dev, dtype=dtype)
    got = inv_rms(x, EPS)
    ref = _old(x.float(), EPS)
    assert got.shape == ref.shape == (rows, 1)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=0)


def test_keeps_leading_dims():
    """hc_pre feeds a [B, T, hc_dim] view and broadcasts the result against [B, T, mix]."""
    dev = "cuda"
    x = torch.randn(2, 3, HC_DIM, device=dev, dtype=torch.bfloat16)
    got = inv_rms(x, EPS)
    assert got.shape == (2, 3, 1)
    torch.testing.assert_close(got, _old(x.float(), EPS), rtol=1e-5, atol=0)


def test_tiny_and_large_magnitudes():
    """Check the scale range a hidden state can reach."""
    dev = "cuda"
    for scale in (1e-3, 1.0, 1e3):
        x = torch.randn(4, HC_DIM, device=dev, dtype=torch.bfloat16) * scale
        torch.testing.assert_close(inv_rms(x, EPS), _old(x.float(), EPS),
                                   rtol=1e-5, atol=0)


def test_accuracy_not_worse_than_aten():
    """The point of the Triton reduce over ``linalg.vector_norm``: no sqrt round-trip,
    so it stays level with ATen against an fp64 reference instead of ~23% behind."""
    torch.manual_seed(0)
    for rows in (64, 1024):
        x = torch.randn(rows, HC_DIM, device="cuda", dtype=torch.bfloat16)
        xf = x.float()
        xd = xf.double()
        truth = torch.rsqrt((xd * xd).mean(-1, keepdim=True) + EPS)
        err_new = ((inv_rms(x, EPS).double() - truth).abs() / truth.abs()).mean()
        err_aten = ((_old(xf, EPS).double() - truth).abs() / truth.abs()).mean()
        # same quality of fp32 answer -- allow a small margin either way, but catch a
        # real regression such as accumulating in bf16 or re-squaring a norm.
        assert err_new < err_aten * 1.25, (rows, err_new.item(), err_aten.item())
