"""fp8 e4m3 weight with square block scales: 128x128 (DeepSeek-V3 style) or 32x32 (DeepSeek-V4.1)."""

from __future__ import annotations

from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import FP8_BLOCK, QuantKind, fp8_block_size
from .base import LinearConfig, LinearKernel, LinearMethod

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def _e8m0(cfg: LinearConfig) -> bool:
    return cfg.scheme is not None and cfg.scheme.weight.scale == "e8m0"


def layer_block_size(layer: Any) -> int:
    """The block edge a finalized layer was declared with: ``K / (scale columns)``, exact by construction."""
    return layer.weight.shape[1] // layer.weight_scale_inv.shape[1]


class Dsv4Fp8BlockLinearKernel(LinearKernel):
    """DeepSeek-V4's reference path: activations quantized to fp8 with power-of-two block scales, e8m0 weight scales read as codes."""

    name = "dsv4"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        return None if _e8m0(cfg) else "serves e8m0 block scales only"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear

        return block_fp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias, block=layer_block_size(layer))


class TritonFp8BlockLinearKernel(LinearKernel):
    """W8A16 GEMV at M=1, dynamic 1x128 W8A8 GEMM above."""

    name = "triton"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if _e8m0(cfg):
            return "reads float block scales; e8m0 codes go to the dsv4 kernel"
        if fp8_block_size(cfg.scheme) != FP8_BLOCK:
            return f"float-scale kernel serves {FP8_BLOCK}x{FP8_BLOCK} blocks only"
        return None

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear

        return block_fp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)


@register_method(QuantKind.FP8_BLOCK, LayerKind.LINEAR)
class Fp8BlockLinearMethod(LinearMethod):
    candidates = (Dsv4Fp8BlockLinearKernel, TritonFp8BlockLinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        block = fp8_block_size(g.scheme)
        if g.in_features % block or any(o % block for o in g.output_sizes):
            raise ValueError(f"block-fp8 needs in/out sizes divisible by {block}, got K={g.in_features} N={g.output_sizes}")
        layer.weight = torch.empty(g.out_features, g.in_features, dtype=FP8)
        # e8m0 codes stay codes for the dsv4 kernel; float scales are bf16 as the readers push them today
        scale_dtype = E8M0 if _e8m0(g) else torch.bfloat16
        layer.weight_scale_inv = torch.empty(g.out_features // block, g.in_features // block, dtype=scale_dtype)


class TritonFp8BlockQuantizeAtLoadKernel(TritonFp8BlockLinearKernel):
    """Same W8A16 GEMV, but the weight arrives bf16 and is block-quantized to fp8 + bf16 scale once,
    post-load, in ``finalize`` -- for weights the checkpoint ships unquantized (e.g. lm_head)."""

    name = "triton_qat"

    def finalize(self, layer: Any) -> None:
        from freetoken.kernel.triton.fp8_block_linear import per_block_quant_fp8

        if layer.weight.dtype is FP8:  # idempotent guard
            return
        w_fp8, scale = per_block_quant_fp8(layer.weight)
        layer.weight = w_fp8
        layer.weight_scale_inv = scale


@register_method(QuantKind.FP8_BLOCK_QAT, LayerKind.LINEAR)
class Fp8BlockQuantizeAtLoadLinearMethod(LinearMethod):
    """fp8-block linear whose checkpoint weight is bf16; quantized to fp8 at load (see the kernel's finalize)."""

    candidates = (TritonFp8BlockQuantizeAtLoadKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        if g.in_features % FP8_BLOCK or any(o % FP8_BLOCK for o in g.output_sizes):
            raise ValueError(f"block-fp8(QAT) needs in/out divisible by {FP8_BLOCK}, got K={g.in_features} N={g.output_sizes}")
        layer.weight = torch.empty(g.out_features, g.in_features, dtype=torch.bfloat16)
