"""linear_fp32: the decode-M path must equal F.linear (an fp32 sum of the same products)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.deepseek_v4.layers import SMALL_M_LINEAR_MAX, linear_fp32


@pytest.mark.parametrize("M", [1, 4, SMALL_M_LINEAR_MAX, SMALL_M_LINEAR_MAX + 1, 64])
def test_matches_f_linear(M):
    g = torch.Generator().manual_seed(M)
    x = torch.randn(M, 16384, generator=g)
    w = torch.randn(24, 16384, generator=g)
    torch.testing.assert_close(linear_fp32(x, w), F.linear(x, w), rtol=1e-4, atol=1e-2)


def test_keeps_leading_dims():
    x = torch.randn(1, 3, 64)  # [B, T, K] as hc_pre passes it
    w = torch.randn(5, 64)
    out = linear_fp32(x, w)
    assert out.shape == (1, 3, 5)
    torch.testing.assert_close(out, F.linear(x, w), rtol=1e-4, atol=1e-3)
