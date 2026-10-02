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

    def apply_out_dtype(self, layer: Any, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        w, b = layer.weight, layer.bias
        if x.is_cuda and x.dtype == w.dtype and x.dtype in (torch.float16, torch.bfloat16) and out_dtype == torch.float32:
            # Preserve fp32 logits without storing or reading a second, fp32 weight copy.
            y = torch.mm(x.reshape(-1, x.shape[-1]), w.T, out_dtype=out_dtype)
            y = y.reshape(*x.shape[:-1], w.shape[0])
            return y if b is None else y + b.to(out_dtype)
        return F.linear(x.to(out_dtype), w.to(out_dtype), b.to(out_dtype) if b is not None else None)


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TorchLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)
