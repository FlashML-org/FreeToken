"""fp8 e4m3 weight with one fp32 scale per output row (per-tensor scales are broadcast)."""

from __future__ import annotations

from typing import Any

import torch

from freetoken.kernel import backend

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod

FP8 = torch.float8_e4m3fn


class TorchFp8TensorLinearKernel(LinearKernel):
    """Static W8A8 through torch._scaled_mm; needs the checkpoint's input_scale and fp8 tensor cores."""

    name = "torch"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if cfg.scheme is None or not cfg.scheme.has("input_scale"):
            return "the checkpoint has no static activation scale"
        cc = backend.device_capability()
        if cc < (8, 9):
            return f"fp8 tensor cores need sm_89+, got sm_{cc[0]}{cc[1]}"
        from freetoken.kernel.triton.e4m3_compat import e4m3_native

        if not e4m3_native():
            return "e4m3 emulation mode has no fp8 GEMM"
        return None

    def finalize(self, layer: Any) -> None:
        from freetoken.kernel.triton.fp8_pertensor_linear import rowwise_scaled_mm_ok, weight_scale_segments

        # a fused projection carries one scalar per part; decide the GEMM shape once, not under graph capture
        scale = layer.weight_scale
        uniform = bool((scale == scale[0]).all().item())
        layer._fp8_uniform_scale = uniform
        layer._fp8_scale_segments = None if uniform else weight_scale_segments(scale)
        if not uniform:
            rowwise_scaled_mm_ok()

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        return fp8_pertensor_linear(
            x, layer.weight, layer.weight_scale, layer.bias,
            layer.input_scale, layer._fp8_uniform_scale, scale_segments=layer._fp8_scale_segments,
        )


class TritonFp8TensorLinearKernel(LinearKernel):
    """W8A16: split-K GEMV at M=1, GEMM above."""

    name = "triton"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        return fp8_pertensor_linear(x, layer.weight, layer.weight_scale, layer.bias)


class EmulationFp8TensorLinearKernel(LinearKernel):
    name = "emulation"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        *lead, K = x.shape
        w = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)[:, None]
        out = (x.reshape(-1, K) @ w.t()).reshape(*lead, w.shape[0])
        return out + layer.bias.to(out.dtype) if layer.bias is not None else out


@register_method(QuantKind.FP8_TENSOR, LayerKind.LINEAR)
class Fp8TensorLinearMethod(LinearMethod):
    candidates = (TorchFp8TensorLinearKernel, TritonFp8TensorLinearKernel, EmulationFp8TensorLinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features, dtype=FP8)
        layer.weight_scale = torch.empty(g.out_features, dtype=torch.float32)
        layer.input_scale = torch.empty((), dtype=torch.float32) if self.scheme.has("input_scale") else None
