"""fp8 e4m3 weight with e8m0 scales per 32 input elements (OCP MXFP8)."""

from __future__ import annotations

from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import MX_GROUP as BLOCK, QuantKind
from .base import LinearKernel, LinearMethod

FP8 = torch.float8_e4m3fn


class TritonMxfp8LinearKernel(LinearKernel):
    """W8A16: split-K GEMV up to M=256, bf16 dequant + cuBLAS above."""

    name = "triton"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.mxfp8_linear import mxfp8_linear

        return mxfp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)


class EmulationMxfp8LinearKernel(LinearKernel):
    name = "emulation"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant

        *lead, K = x.shape
        w = mxfp8_dequant(layer.weight, layer.weight_scale_inv, dtype=torch.float32)
        out = (x.reshape(-1, K).float() @ w.t()).to(x.dtype).reshape(*lead, w.shape[0])
        return out + layer.bias.to(out.dtype) if layer.bias is not None else out


@register_method(QuantKind.MXFP8, LayerKind.LINEAR)
class Mxfp8LinearMethod(LinearMethod):
    candidates = (TritonMxfp8LinearKernel, EmulationMxfp8LinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        if g.in_features % BLOCK:
            raise ValueError(f"mxfp8 needs in_features divisible by {BLOCK}, got {g.in_features}")
        layer.weight = torch.empty(g.out_features, g.in_features, dtype=FP8)
        layer.weight_scale_inv = torch.empty(g.out_features, g.in_features // BLOCK, dtype=torch.uint8)
