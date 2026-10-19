"""e2m1 weight packed two per byte, e4m3 scales per 16 input elements, one fp32 global per output row."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any

import torch

from freetoken.kernel import backend
from freetoken.utils import init_logger

from ..registry import LayerKind, register_method
from ..scheme import NVFP4_GROUP as GROUP, QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod

logger = init_logger(__name__)

FP8 = torch.float8_e4m3fn
# fused parts whose weight globals are further apart than this ratio get a load-time warning
_PART_GLOBAL_WARN_RATIO = 1.05


class TritonNvfp4LinearKernel(LinearKernel):
    """W4A16 on the K-major resident layout."""

    name = "triton"

    def finalize(self, layer: Any) -> None:
        from freetoken.kernel.triton.nvfp4_linear import nvfp4_transpose_resident

        layer.weight, layer.weight_scale = nvfp4_transpose_resident(layer.weight, layer.weight_scale)

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t

        return nvfp4_dense_linear_t(x, layer.weight, layer.weight_scale, layer.weight_global, layer.bias)


@functools.cache
def _warn_part_globals() -> None:
    logger.warning_rank0(
        "an NVFP4 linear fuses parts whose weight global scales differ by more than %d%%; the W4A4 kernels use the largest "
        "for every part, which scales the other parts' outputs by their ratio; --quant-backend linear.nvfp4=triton keeps "
        "each part's global", round((_PART_GLOBAL_WARN_RATIO - 1) * 100),
    )


def _tensor_global(layer: Any) -> torch.Tensor:
    """The one weight global the W4A4 GEMMs and the small-batch kernels take: the largest per-part global, the block scales unchanged."""
    g = layer.weight_global.float()
    g_max = g.max()
    if g_max > _PART_GLOBAL_WARN_RATIO * g.min():
        _warn_part_globals()
    return g_max


def _swizzle_128x4(scale: torch.Tensor) -> torch.Tensor:
    """[N, K/16] block scales -> the flat cuBLAS / CUTLASS 128x4 layout, rows padded to 128 and columns to 4 with zeros."""
    rows, cols = scale.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(rb * 128, cb * 4, dtype=scale.dtype, device=scale.device)
    padded[:rows, :cols] = scale
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


@functools.cache
def _sm120_gemv(device_index: int):
    """The row-major W4A16 GEMV / skinny GEMM module for this GPU, or None."""
    from freetoken.kernel import gluon

    if torch.cuda.get_device_capability(device_index) != (12, 0) or not gluon.available():
        return None
    from freetoken.kernel.gluon.nvfp4.linear import sm120_gemv

    return sm120_gemv


def _small_batch(layer: Any, x: torch.Tensor, max_tokens: int) -> torch.Tensor | None:
    """W4A16 on the W4A4 kernels' own weight for decode-sized batches, where the W4A4 GEMMs leave bandwidth unused; None to run the GEMM."""
    gemv = _sm120_gemv(x.device.index)
    if gemv is None or x.dtype != torch.bfloat16 or x.shape[0] == 0:
        return None
    route = gemv.gemv_config(layer.weight.shape[0], x.shape[1], x.shape[0], max_tokens)
    if route is None:
        return None
    kernel, tile = route
    return kernel(x if x.stride(1) == 1 else x.contiguous(), layer.weight, layer.weight_scale, layer.weight_global, **tile)


@functools.cache
def _flashinfer_cuda_major() -> int:
    """The CUDA version flashinfer's b12x GEMM checks at call time: nvcc's, else torch's."""
    from flashinfer.jit.cpp_ext import get_cuda_version

    return get_cuda_version().major


def _small_batch_usable() -> bool:
    return torch.cuda.is_available() and _sm120_gemv(torch.cuda.current_device()) is not None


class CublasNvfp4LinearKernel(LinearKernel):
    """W4A4: activations quantized under the checkpoint's input_scale, then torch's NVFP4 scaled_mm (cuBLASLt) on the row-major weight."""

    name = "cublas"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if backend.is_rocm() or not torch.cuda.is_available():
            return "needs CUDA"
        if cfg.scheme is None or not cfg.scheme.has("input_scale"):
            return "the weight declares no static activation scale (W4A16 checkpoint)"
        if backend.device_capability() < (10, 0):
            return "NVFP4 tensor cores need sm_100+"
        if not hasattr(torch.nn.functional, "scaled_mm") or not hasattr(torch, "float4_e2m1fn_x2"):
            return f"torch {torch.__version__} has no NVFP4 scaled_mm"
        if cfg.out_features % 16 or cfg.in_features % 32:
            return f"scaled_mm needs N % 16 == 0 and K % 32 == 0, got N={cfg.out_features} K={cfg.in_features}"
        return None

    def worth_it(self, cfg: LinearConfig) -> bool:
        # cuBLAS alone runs decode-sized batches up to 1.4x slower than the triton GEMV; the small-batch kernels cover those on sm_120
        return _small_batch_usable()

    def finalize(self, layer: Any) -> None:
        g_w = _tensor_global(layer)
        layer.weight_scale = _swizzle_128x4(layer.weight_scale)
        layer.weight_global = g_w.reshape(())
        layer.input_scale = layer.input_scale.float().reshape(())

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F

        from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4

        *lead, K = x.shape
        out = _small_batch(layer, x.reshape(-1, K), 16)
        if out is not None:
            out = out.reshape(*lead, layer.weight.shape[0])
            return out + layer.bias.to(out.dtype) if layer.bias is not None else out
        xq, xs = quantize_nvfp4(x.reshape(-1, K), layer.input_scale.reshape(1), swizzled=True)
        block, tensor = F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise
        swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
        out = F.scaled_mm(
            xq.view(torch.float4_e2m1fn_x2), layer.weight.view(torch.float4_e2m1fn_x2).t(),
            [xs, layer.input_scale], [block, tensor], [layer.weight_scale, layer.weight_global], [block, tensor],
            swizzle_a=swizzle, swizzle_b=swizzle, output_dtype=x.dtype,
        ).reshape(*lead, layer.weight.shape[0])
        return out + layer.bias.to(out.dtype) if layer.bias is not None else out


class B12xNvfp4LinearKernel(LinearKernel):
    """W4A4 through flashinfer alone: its activation quant and its SM12x CuTe-DSL GEMM (mm_fp4 backend b12x) on the row-major weight."""

    name = "b12x"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if backend.is_rocm() or not torch.cuda.is_available():
            return "needs CUDA"
        if cfg.scheme is None or not cfg.scheme.has("input_scale"):
            return "the weight declares no static activation scale (W4A16 checkpoint)"
        cc = backend.device_capability()
        if cc[0] != 12:
            return f"b12x requires sm_120 / sm_121, got sm_{cc[0]}{cc[1]}"
        if not backend.is_flashinfer_installed():
            return "flashinfer is not installed"
        drv = backend.driver_cuda_version()
        if drv is not None and drv < 13000:
            return f"b12x JIT-compiles through the driver and needs CUDA 13+, the driver supports {drv // 1000}.{drv % 1000 // 10}"
        if cfg.in_features % 32 or cfg.out_features % 8:
            # flashinfer checks only K; N % 8 != 0 returns wrong output
            return f"b12x needs K % 32 == 0 and N % 8 == 0, got K={cfg.in_features} N={cfg.out_features}"
        try:
            from flashinfer import fp4_quantize, mm_fp4, nvfp4_block_scale_interleave  # noqa: F401

            cuda_major = _flashinfer_cuda_major()
        except Exception as exc:
            return f"flashinfer FP4 GEMM symbols are unusable ({exc!r})"
        if cuda_major < 13:
            return f"flashinfer's b12x GEMM needs a CUDA 13 toolkit (nvcc, else torch), got CUDA {cuda_major}"
        return None

    def worth_it(self, cfg: LinearConfig) -> bool:
        return _small_batch_usable()

    def finalize(self, layer: Any) -> None:
        from flashinfer import nvfp4_block_scale_interleave

        g_w = _tensor_global(layer)
        layer.weight_scale = nvfp4_block_scale_interleave(layer.weight_scale.view(torch.uint8)).view(FP8)
        # flashinfer takes the quant-side activation global and alpha = 1 / (that * the weight's quant-side global)
        layer.input_scale_inv = (1.0 / layer.input_scale.float()).reshape(1)
        layer.alpha = (layer.input_scale.float() * g_w).reshape(1)
        layer.weight_global = g_w.reshape(())

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from flashinfer import fp4_quantize, mm_fp4

        *lead, K = x.shape
        x2 = x.reshape(-1, K)
        if x2.shape[0] == 0:
            return x.new_empty(*lead, layer.weight.shape[0])
        # b12x's W4A4 GEMM beats the W4A16 kernels from 9 tokens on GeForce, where bf16 mma accumulates in fp32 at half rate
        out = _small_batch(layer, x2, 8)
        if out is not None:
            out = out.reshape(*lead, layer.weight.shape[0])
            return out + layer.bias.to(out.dtype) if layer.bias is not None else out
        xq, xs = fp4_quantize(x2.contiguous(), layer.input_scale_inv, is_sf_swizzled_layout=True)
        out = mm_fp4(xq, layer.weight.T, xs, layer.weight_scale, layer.alpha, x.dtype, backend="b12x").reshape(*lead, layer.weight.shape[0])
        return out + layer.bias.to(out.dtype) if layer.bias is not None else out


class MarlinNvfp4LinearKernel(LinearKernel):
    """vLLM's fused FP4 Marlin GEMM (W4A16); needs one global scale for the whole weight."""

    name = "marlin"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if backend.is_rocm():
            return "Marlin is CUDA-only and unavailable on ROCm; use triton"
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
        layer.weight_global = torch.empty(0, dtype=torch.float32, device=dev)

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
    candidates = (B12xNvfp4LinearKernel, CublasNvfp4LinearKernel, TritonNvfp4LinearKernel, MarlinNvfp4LinearKernel, EmulationNvfp4LinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        if g.in_features % GROUP:
            raise ValueError(f"nvfp4 needs in_features divisible by {GROUP}, got {g.in_features}")
        layer.weight = torch.empty(g.out_features, g.in_features // 2, dtype=torch.uint8)
        layer.weight_scale = torch.empty(g.out_features, g.in_features // GROUP, dtype=FP8)
        layer.weight_global = torch.empty(g.out_features, dtype=torch.float32)
        # no W4A16 kernel reads it; declared so a W4A4 checkpoint loads complete and a W4A4 kernel finds it in place
        layer.input_scale = torch.empty((), dtype=torch.float32) if self.scheme.has("input_scale") else None
