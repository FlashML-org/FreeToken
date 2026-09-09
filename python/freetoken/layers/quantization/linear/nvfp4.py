"""e2m1 weight packed two per byte, e4m3 scales per 16 input elements, one fp16 global per output row."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from freetoken.kernel import backend

from ..registry import LayerKind, register_method
from ..scheme import NVFP4_GROUP as GROUP, QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod

FP8 = torch.float8_e4m3fn


class TritonNvfp4LinearKernel(LinearKernel):
    """W4A16 on the K-major resident layout."""

    name = "triton"

    def finalize(self, layer: Any) -> None:
        from freetoken.kernel.triton.nvfp4_linear import nvfp4_transpose_resident

        layer.weight, layer.weight_scale = nvfp4_transpose_resident(layer.weight, layer.weight_scale)

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t

        return nvfp4_dense_linear_t(x, layer.weight, layer.weight_scale, layer.weight_global, layer.bias)


class MarlinNvfp4LinearKernel(LinearKernel):
    """vLLM's fused FP4 Marlin GEMM (W4A16); needs one global scale for the whole weight."""

    name = "marlin"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if not backend.is_vllm_installed():
            return "vLLM is not installed"
        if len(cfg.output_sizes) > 1:
            return "fused projection carries one global scale per part"
        try:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: F401
                apply_fp4_marlin_linear,
                is_fp4_marlin_supported,
                prepare_fp4_layer_for_marlin,
            )
        except Exception as exc:
            return f"vLLM Marlin FP4 symbols are unusable ({exc!r})"
        if not is_fp4_marlin_supported():
            return "vLLM reports FP4 Marlin unsupported on this device"
        return None

    def finalize(self, layer: Any) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin

        # Marlin wants the per-tensor scalar; weight_global is that scalar broadcast to [N]
        packed = SimpleNamespace(
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_scale_2=layer.weight_global.reshape(-1)[0].to(torch.bfloat16),
            output_size_per_partition=layer.out_features,
            input_size_per_partition=layer.in_features,
            params_dtype=torch.bfloat16,
        )
        prepare_fp4_layer_for_marlin(packed)
        layer._marlin_weight = packed.weight
        layer._marlin_scale = packed.weight_scale
        layer._marlin_global = packed.weight_scale_2
        layer._marlin_workspace = packed.workspace
        dev = layer.weight.device
        layer.weight = torch.empty(0, dtype=torch.uint8, device=dev)
        layer.weight_scale = torch.empty(0, dtype=FP8, device=dev)
        layer.weight_global = torch.empty(0, dtype=torch.float16, device=dev)

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

        out = apply_fp4_marlin_linear(
            input=x,
            weight=layer._marlin_weight,
            weight_scale=layer._marlin_scale,
            weight_scale_2=layer._marlin_global,
            workspace=layer._marlin_workspace,
            size_n=layer.out_features,
            size_k=layer.in_features,
        )
        return out + layer.bias.to(out.dtype) if layer.bias is not None else out


class EmulationNvfp4LinearKernel(LinearKernel):
    """Dequantize to bf16 with dequant_nvfp4, then a plain matmul; keeps the row-major layout."""

    name = "emulation"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

        *lead, K = x.shape
        slots = torch.zeros(1, dtype=torch.int32, device=x.device)
        w = dequant_nvfp4(
            layer.weight.unsqueeze(0), layer.weight_scale.unsqueeze(0), layer.weight_global.unsqueeze(0),
            slots, dtype=torch.bfloat16,
        )[0]
        out = (x.reshape(-1, K) @ w.t()).to(x.dtype).reshape(*lead, w.shape[0])
        return out + layer.bias.to(out.dtype) if layer.bias is not None else out


@register_method(QuantKind.NVFP4, LayerKind.LINEAR)
class Nvfp4LinearMethod(LinearMethod):
    candidates = (TritonNvfp4LinearKernel, MarlinNvfp4LinearKernel, EmulationNvfp4LinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        if g.in_features % GROUP:
            raise ValueError(f"nvfp4 needs in_features divisible by {GROUP}, got {g.in_features}")
        layer.weight = torch.empty(g.out_features, g.in_features // 2, dtype=torch.uint8)
        layer.weight_scale = torch.empty(g.out_features, g.in_features // GROUP, dtype=FP8)
        layer.weight_global = torch.empty(g.out_features, dtype=torch.float16)
        # input_scale stays undeclared: the W4A16 kernels never read it and today's readers drop it
