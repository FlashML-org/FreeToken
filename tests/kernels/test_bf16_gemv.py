"""The single-row bf16 GEMV, and the bf16 linear kernel that sends decode's projections through it.

A decode step at batch 1 multiplies one activation row by every dense weight, so each projection
is a GEMV bound by how fast the weight streams in. The kernel keeps F.linear's contract -- fp32
accumulation, bf16 or fp32 out -- and the linear kernel keeps F.linear for every other input.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda"

# Flash-Next's GDN in_proj, attention qkv, hyper-connection down and up projections; the last
# leaves a tail in both N and K.
SHAPES = [(16480, 2560), (13312, 2560), (336, 10240), (10240, 320), (1001, 6112)]


def _weight(N: int, K: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("out_dtype,tolerance", [(torch.bfloat16, 1e-2), (torch.float32, 1e-5)])
def test_the_gemv_matches_an_fp32_reference(N: int, K: int, out_dtype: torch.dtype, tolerance: float):
    from freetoken.kernel.triton.bf16_gemv import bf16_gemv

    weight = _weight(N, K, seed=N + K)
    x = torch.randn(1, K, device=DEV, dtype=torch.bfloat16)

    y = bf16_gemv(x, weight, out_dtype)

    reference = x.float() @ weight.float().t()
    assert y.shape == (1, N)
    assert y.dtype == out_dtype
    relative = (y.float() - reference).abs().max() / reference.abs().max()
    assert relative.item() < tolerance, relative.item()


def test_the_bf16_linear_method_prefers_the_triton_kernel():
    from freetoken.layers.quantization.linear.base import LinearConfig
    from freetoken.layers.quantization.linear.unquantized import UnquantizedLinearMethod
    from freetoken.layers.quantization.method import select_kernel

    kernel = select_kernel(UnquantizedLinearMethod.candidates, "auto", LinearConfig(2560, 16480))

    assert kernel.name == "triton"


@pytest.fixture
def gemv_calls(monkeypatch) -> list[tuple[int, ...]]:
    """Every shape the linear kernel hands the GEMV, with the GEMV still doing the work."""
    import freetoken.kernel.triton.bf16_gemv as module

    calls: list[tuple[int, ...]] = []
    real = module.bf16_gemv

    def recording(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        calls.append(tuple(x.shape))
        return real(x, weight, out_dtype)

    monkeypatch.setattr(module, "bf16_gemv", recording)
    return calls


def test_a_single_bf16_row_runs_the_gemv(gemv_calls: list[tuple[int, ...]]):
    from freetoken.layers.quantization.linear.unquantized import TritonLinearKernel

    layer = SimpleNamespace(weight=_weight(4096, 2560, seed=1), bias=None)
    x = torch.randn(1, 2560, device=DEV, dtype=torch.bfloat16)

    y = TritonLinearKernel().apply(layer, x)

    assert gemv_calls == [(1, 2560)]
    reference = F.linear(x.float(), layer.weight.float())
    assert ((y.float() - reference).abs().max() / reference.abs().max()).item() < 1e-2


@pytest.mark.parametrize(
    "rows,activation,with_bias",
    [(2, torch.bfloat16, False), (64, torch.bfloat16, False), (1, torch.float32, False), (1, torch.bfloat16, True)],
)
def test_every_other_input_keeps_the_torch_path(
    gemv_calls: list[tuple[int, ...]], rows: int, activation: torch.dtype, with_bias: bool
):
    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel, TritonLinearKernel

    weight = _weight(512, 2560, seed=2)
    bias = torch.randn(512, device=DEV, dtype=torch.bfloat16) if with_bias else None
    layer = SimpleNamespace(weight=weight, bias=bias)
    x = torch.randn(rows, 2560, device=DEV, dtype=activation)

    y = TritonLinearKernel().apply(layer, x)

    assert gemv_calls == []
    assert torch.equal(y, TorchLinearKernel().apply(layer, x))


def test_the_gemv_replays_in_a_cuda_graph():
    from freetoken.layers.quantization.linear.unquantized import TritonLinearKernel

    kernel = TritonLinearKernel()
    layer = SimpleNamespace(weight=_weight(13312, 2560, seed=3), bias=None)
    x = torch.randn(1, 2560, device=DEV, dtype=torch.bfloat16)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        kernel.apply(layer, x)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = kernel.apply(layer, x)

    x.copy_(torch.randn(1, 2560, device=DEV, dtype=torch.bfloat16))
    graph.replay()

    assert torch.equal(y, kernel.apply(layer, x))
