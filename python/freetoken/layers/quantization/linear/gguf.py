"""GGUF Linear: packed ggml rows multiplied by llama.cpp's MMVQ / MMQ kernels; a fused module runs one GEMM per ggml type, float parts as plain bf16 rows."""

from __future__ import annotations

from typing import Any

import torch

from ..ggml_types import is_quantized, pad_rows, row_bytes, segment_param, segments
from ..registry import LayerKind, register_method
from ..scheme import QuantKind, gguf_types
from .base import LinearConfig, LinearKernel, LinearMethod



class GgmlLinearKernel(LinearKernel):
    name = "ggml"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        return None

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import ggml

        x2 = x.reshape(-1, x.shape[-1])
        segs = layer._ggml_segments
        m = x2.shape[0]
        y = torch.empty(m, sum(s.rows for s in segs), dtype=x.dtype, device=x.device)
        # every segment reads the same x: quantize it once for all the vector-kernel segments
        xq = ggml.quantize_q8_1(x2) if sum(is_quantized(s.ggml_type) and ggml.uses_mmvq(s.ggml_type, m) for s in segs) > 1 else None
        for i, s in enumerate(segs):
            w = getattr(layer, segment_param(i))[: s.rows]
            out = y[:, s.out_start : s.out_start + s.rows]
            if is_quantized(s.ggml_type):
                ggml.mul_mat(w, s.ggml_type, x2, out=out, xq=xq)
            else:
                torch.mm(x2, w.t(), out=out)
        if layer.bias is not None:
            y = y + layer.bias
        return y.reshape(*x.shape[:-1], y.shape[-1])


@register_method(QuantKind.GGUF, LayerKind.LINEAR)
class GGUFLinearMethod(LinearMethod):
    candidates = (GgmlLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        segs = segments(gguf_types(g.scheme), g.output_sizes)
        layer._ggml_segments = segs
        k = g.in_features
        for i, s in enumerate(segs):
            if is_quantized(s.ggml_type):
                setattr(layer, segment_param(i), torch.empty(s.rows + pad_rows(k), row_bytes(s.ggml_type, k), dtype=torch.uint8))
            else:
                setattr(layer, segment_param(i), torch.empty(s.rows, k))
