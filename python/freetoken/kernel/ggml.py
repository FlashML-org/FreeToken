"""llama.cpp's ggml-cuda kernels (vendored under ``csrc/ggml``) as a tvm-ffi module: prebuilt in the kernel cache, else compiled on first use.

Weights stay in their packed ggml blocks. ``mul_mat`` / ``mul_mat_id`` pick MMVQ or MMQ the way llama.cpp's mul_mat / mul_mat_id do.
Activations are fp32, bf16 or fp16 and are quantized to q8_1 first; results come out in the activation dtype and are accumulated in fp32.
"""

from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING

import torch

from .utils import _is_rocm, load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module

_ROOT = pathlib.Path(__file__).parent / "csrc" / "ggml"


def load_module(build_directory: str | None = None) -> Module:
    cuda_dir = _ROOT / "ggml-cuda"
    sources = [
        _ROOT / "freetoken" / "ggml_shim.cu",
        _ROOT / "freetoken" / "ggml_ops.cu",
        *(cuda_dir / f for f in ("mmvq.cu", "mmq.cu", "mmid.cu", "quantize.cu", "convert.cu", "getrows.cu")),
        *sorted((cuda_dir / "template-instances").glob("mmq-instance-*.cu")),
    ]
    if _is_rocm():
        flags = ["-ffast-math", "-DGGML_USE_HIP"]
    else:
        # a multi-arch kernel-cache build is ~6x smaller with compressed SASS
        flags = ["-use_fast_math", "--extended-lambda", "--compress-mode=size"]
    return load_aot(
        "ggml",
        cuda_files=[str(s) for s in sources],
        extra_cuda_cflags=flags,
        extra_include_paths=[str(_ROOT / "include"), str(_ROOT / "src"), str(cuda_dir)],
        build_directory=build_directory,
    )


@functools.cache
def _module() -> Module:
    return load_module()


@functools.cache
def _mmq_tail_bytes(ggml_type: int, k: int) -> int:
    return _module().mmq_tail_bytes(ggml_type, k)


def _check_mmq_tail(w: torch.Tensor, ggml_type: int, k: int) -> None:
    """MMQ tiles read up to MATRIX_ROW_PADDING values past the last row of ``w``, so its storage must hold them."""
    tail = _mmq_tail_bytes(ggml_type, k)
    if tail == 0 or w.numel() == 0:
        return
    end = w.storage_offset() + sum((n - 1) * s for n, s in zip(w.shape, w.stride())) + 1
    if end + tail > w.untyped_storage().nbytes():
        raise ValueError(f"ggml type {ggml_type} K={k} weight needs {tail} bytes of zero padding after its last row for MMQ")


def mul_mat(w: torch.Tensor, ggml_type: int, x: torch.Tensor, *, out: torch.Tensor | None = None, xq: torch.Tensor | None = None) -> torch.Tensor:
    """``x[M, K] @ w[N, K]^T`` in ``x``'s dtype with ``w`` packed as ``[N, row_bytes]`` uint8, or written into the ``[M, N]`` view ``out``.

    ``xq`` is ``quantize_q8_1(x)``: the vector kernel reads it instead of quantizing ``x`` again.
    """
    m, k = x.shape
    if out is None:
        out = torch.empty(m, w.shape[0], dtype=x.dtype, device=x.device)
    if not uses_mmvq(ggml_type, m):
        _check_mmq_tail(w, ggml_type, k)
    _module().mul_mat(w, ggml_type, x.contiguous(), out, xq)
    return out


def quantize_q8_1(x: torch.Tensor) -> torch.Tensor:
    """``x[M, K]`` as the vector kernel's q8_1 blocks, to share among GEMVs that read the same ``x``."""
    m, k = x.shape
    out = torch.empty(m, _q8_1_row_bytes(k), dtype=torch.uint8, device=x.device)
    _module().quantize_q8_1(x.contiguous(), out)
    return out


@functools.cache
def _q8_1_row_bytes(k: int) -> int:
    return _module().q8_1_row_bytes(k)


@functools.cache
def uses_mmvq(ggml_type: int, m: int) -> bool:
    """Whether ``mul_mat`` of ``m`` rows on this GPU runs the vector kernel, the one that reads ``xq``."""
    return _module().uses_mmvq(ggml_type, m)


def mul_mat_id(w: torch.Tensor, ggml_type: int, x: torch.Tensor, ids: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Per-route expert GEMM over ``w[E, N, row_bytes]``: returns ``[T, k, N]`` in ``x``'s dtype; ``x`` is ``[T, K]`` (shared by the k routes) or ``[T, k, K]``.

    ``scale`` is an fp32 ``[E]`` view (strided is fine) that multiplies each NVFP4 expert's rows; it needs ``T <= mmvq_max_tokens``.
    """
    t, top_k = ids.shape
    out = torch.empty(t, top_k, w.shape[1], dtype=x.dtype, device=x.device)
    if t > mmvq_max_tokens(ggml_type):
        _check_mmq_tail(w, ggml_type, x.shape[-1])
    _module().mul_mat_id(w, ggml_type, (x.unsqueeze(1) if x.dim() == 2 else x).contiguous(), ids, out, scale)
    return out


def mul_mat_id_glu(
    w: torch.Tensor,
    ggml_type: int,
    x: torch.Tensor,
    ids: torch.Tensor,
    glu: str,
    limit: float = 0.0,
    gate_scale: torch.Tensor | None = None,
    up_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``glu(gate, up)`` per route in one pass for small ``T``: ``w[E, 2I, row_bytes]`` holds gate rows then up rows; returns ``[T, k, I]`` in ``x``'s dtype.

    ``glu`` is one of ``swiglu``, ``geglu`` (tanh GELU), ``swiglu_oai`` (alpha 1.702, limit 7) and ``swiglu_clamp`` (alpha 1, clamped at ``limit``).
    ``gate_scale`` / ``up_scale`` are fp32 ``[E]`` views that multiply each NVFP4 expert's gate / up results before the glu.
    """
    t, top_k = ids.shape
    out = torch.empty(t, top_k, w.shape[1] // 2, dtype=x.dtype, device=x.device)
    _module().mul_mat_id_glu(w, ggml_type, x.contiguous(), ids, out, glu, limit, gate_scale, up_scale)
    return out


def moe_weighted_sum(y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``sum_j weights[t, j] * y[t, j, :]`` over ``y[T, k, N]``, accumulated in fp32 and returned in ``y``'s dtype."""
    out = torch.empty(y.shape[0], y.shape[2], dtype=y.dtype, device=y.device)
    _module().moe_weighted_sum(y.contiguous(), weights.float().contiguous(), out)
    return out


@functools.cache
def mmvq_max_tokens(ggml_type: int) -> int:
    """Largest token count the small-batch MoE kernels take on this GPU."""
    return _module().mmvq_max_tokens(ggml_type)


def dequantize(w: torch.Tensor, ggml_type: int, k: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    out = torch.empty(w.numel() // w.shape[-1], k, dtype=dtype, device=w.device)
    _module().dequantize(w, ggml_type, k, out)
    return out


def get_rows(w: torch.Tensor, ggml_type: int, k: int, ids: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    out = torch.empty(ids.shape[0], k, dtype=dtype, device=ids.device)
    _module().get_rows(w, ggml_type, k, ids, out)
    return out


__all__ = [
    "dequantize",
    "get_rows",
    "load_module",
    "mmvq_max_tokens",
    "moe_weighted_sum",
    "mul_mat",
    "mul_mat_id",
    "mul_mat_id_glu",
    "quantize_q8_1",
    "uses_mmvq",
]
