"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            b = b.to(x.dtype) if b is not None else None
        return F.linear(x, w, b)


class TritonLinearKernel(TorchLinearKernel):
    """A single bf16 row, decode at batch 1, as one GEMV; every other input is the torch kernel's."""

    name = "triton"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        w = layer.weight
        single_row = x.numel() == x.shape[-1]
        if single_row and x.is_cuda and layer.bias is None and x.dtype == w.dtype == torch.bfloat16:
            from freetoken.kernel.triton.bf16_gemv import bf16_gemv

            return bf16_gemv(x, w, x.dtype)
        return super().apply(layer, x)


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TritonLinearKernel, TorchLinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)
