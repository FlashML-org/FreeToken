"""The W4A4 NVFP4 linears: cublas (our activation quant with 128x4-swizzled scales feeding torch's NVFP4 scaled_mm) and b12x (flashinfer only),
and the sm_120 W4A16 GEMV / skinny GEMM both run for small batches on the same row-major weight."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0), reason="needs NVFP4 tensor cores (sm_100+)"
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value, ties to the even code, saturating at 6 (cvt.rn.satfinite)."""
    grid = _E2M1[:8].to(v.device)
    a = v.abs().clamp(max=6.0)
    hi = torch.searchsorted(grid, a.contiguous()).clamp(max=7)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    code = torch.where(d_lo < d_hi, lo, torch.where(d_hi < d_lo, hi, torch.where(lo % 2 == 0, lo, hi)))
    return torch.copysign(grid[code], v)


def _qdq(x: torch.Tensor, g: float) -> torch.Tensor:
    xb = x.float().view(x.shape[0], -1, 16)
    g = torch.tensor(g, dtype=torch.float32, device=x.device)
    sf = ((xb.abs().amax(-1) / (6.0 * g)).to(torch.float8_e4m3fn).float() * g).unsqueeze(-1)
    inv = torch.where(sf > 0, 1.0 / sf, torch.zeros_like(sf))
    return (_round_e2m1(xb * inv) * sf).view_as(x)


def _dequant(packed: torch.Tensor, scale: torch.Tensor, row_global: torch.Tensor) -> torch.Tensor:
    n, k2 = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, 2 * k2).long()
    return _E2M1.to(packed.device)[codes] * scale.float().repeat_interleave(16, dim=1) * row_global.float().unsqueeze(1)


def _small_batch_route(N, K, M, max_tokens=16):
    if torch.cuda.get_device_capability() != (12, 0):
        return None
    from freetoken.kernel import gluon

    if not gluon.available():
        return None
    from freetoken.kernel.gluon.nvfp4.linear.sm120_gemv import gemv_config

    return gemv_config(N, K, M, max_tokens)


@pytest.mark.parametrize("rows,K", [(1, 2048), (7, 2080), (300, 5376)])
def test_swizzled_scales_are_the_cublas_128x4_layout(rows, K):
    """Same codes either way; the swizzled scales are the linear ones in cuBLAS's [row tile, k tile, 32, 4, 4] order, zero padded."""
    from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4
    from freetoken.layers.quantization.linear.nvfp4 import _swizzle_128x4

    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    g = (x.float().abs().max() / 2688).reshape(1)
    q, s = quantize_nvfp4(x, g)
    q_swz, s_swz = quantize_nvfp4(x, g, swizzled=True)
    assert torch.equal(q, q_swz)
    assert torch.equal(_swizzle_128x4(s).view(torch.uint8), s_swz.view(torch.uint8))


@pytest.mark.parametrize("kernel_name", ["cublas", "b12x"])
@pytest.mark.parametrize("M", [1, 2, 7, 16, 300])
@pytest.mark.parametrize("N,K", [(512, 1024), (528, 2080), (12, 2048)])
@pytest.mark.parametrize("parts_differ", [False, True])
def test_w4a4_linear_matches_nvfp4_activation_reference(kernel_name, M, N, K, parts_differ):
    """finalize + apply against a torch emulation: W4A4, or W4A16 where the sm_120 small-batch kernels take the batch (M <= 16 for cublas,
    M <= 8 for b12x, at 512 x 1024; 528 x 2080 fits neither and keeps the GEMM; N = 12 must be refused, b12x returns garbage there);
    parts_differ gives the halves of a fused weight different globals, and the larger one applies to both."""
    from freetoken.layers.quantization.linear.base import LinearConfig
    from freetoken.layers.quantization.linear.nvfp4 import Nvfp4LinearMethod
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    kernel = next(c for c in Nvfp4LinearMethod.candidates if c.name == kernel_name)()
    reason = kernel.unusable_reason(LinearConfig(K, N, (N // 2, N // 2), nvfp4_scheme(input_scale=True)))
    if reason:
        pytest.skip(reason)
    g = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    scale = (torch.rand(N, K // 16, device="cuda", generator=g) + 0.25).to(torch.float8_e4m3fn)
    row_global = torch.full((N,), 1e-3, dtype=torch.float32, device="cuda")
    if parts_differ:
        row_global[N // 2:] = 4e-3
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    # off amax / 2688 by an irrational factor: with amax / 2688 many values sit exactly on an e2m1 midpoint, where the kernel's
    # approximate fp32 reciprocal and torch's exact one round apart
    g_in = float(x.float().abs().max()) / 2688 * math.pi / 3
    # one tensor-wise global: the largest part's, applied to every row
    w_ref = _dequant(weight, scale, torch.full_like(row_global, float(row_global.max())))
    small = _small_batch_route(N, K, M, 16 if kernel_name == "cublas" else 8) is not None
    ref = (x.float() if small else _qdq(x, g_in)) @ w_ref.T + bias.float()

    layer = SimpleNamespace(weight=weight.clone(), weight_scale=scale.clone(), weight_global=row_global.clone(),
                            input_scale=torch.tensor(g_in, device="cuda"), bias=bias)
    kernel.finalize(layer)
    out = kernel.apply(layer, x)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


@pytest.mark.parametrize("N,K,T", [
    (16384, 1024, 1),  # fma, 4 rows per warp
    (16384, 768, 2),  # fma, one word per lane
    (640, 2048, 1),  # fma, four words per lane
    (640, 17408, 1),  # long K: mma
    (640, 4224, 12),  # mma bm32, masked last K block
    (256, 16512, 4),  # mma, masked last K block
    (640, 2048, 5),  # mma bm16, padded tokens
])
def test_sm120_gemv_tiles_match_dequant_reference(N, K, T):
    """Each tile gemv_config hands out, against x @ dequant(W)^T on the flat 128x4 scales."""
    route = _small_batch_route(N, K, T)
    if route is None:
        pytest.skip("needs sm_120 with Gluon")
    from freetoken.layers.quantization.linear.nvfp4 import _swizzle_128x4

    g = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    scale = (torch.rand(N, K // 16, device="cuda", generator=g) + 0.25).to(torch.float8_e4m3fn)
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16, generator=g)
    g_w = torch.tensor(1e-3, device="cuda")
    ref = x.float() @ _dequant(weight, scale, torch.full((N,), 1e-3, device="cuda")).T
    kernel, tile = route
    out = kernel(x, weight, _swizzle_128x4(scale), g_w, **tile)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


@pytest.mark.parametrize("kernel_name", ["cublas", "b12x"])
@pytest.mark.parametrize("M", [0, 4, 64])
def test_w4a4_linear_takes_row_strided_and_empty_inputs(kernel_name, M):
    """A row-strided slice and an empty batch give the same output as a contiguous input, on both the small-batch and the GEMM route."""
    from freetoken.layers.quantization.linear.base import LinearConfig
    from freetoken.layers.quantization.linear.nvfp4 import Nvfp4LinearMethod
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    N, K = 512, 1024
    kernel = next(c for c in Nvfp4LinearMethod.candidates if c.name == kernel_name)()
    reason = kernel.unusable_reason(LinearConfig(K, N, (N,), nvfp4_scheme(input_scale=True)))
    if reason:
        pytest.skip(reason)
    g = torch.Generator(device="cuda").manual_seed(0)
    layer = SimpleNamespace(weight=torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g),
                            weight_scale=(torch.rand(N, K // 16, device="cuda", generator=g) + 0.25).to(torch.float8_e4m3fn),
                            weight_global=torch.full((N,), 1e-3, dtype=torch.float32, device="cuda"),
                            input_scale=torch.tensor(0.01, device="cuda"), bias=None)
    kernel.finalize(layer)
    buf = torch.randn(M, K + 64, device="cuda", dtype=torch.bfloat16, generator=g)
    out = kernel.apply(layer, buf[:, :K])
    assert out.shape == (M, N)
    torch.testing.assert_close(out, kernel.apply(layer, buf[:, :K].contiguous()), rtol=0, atol=0)
