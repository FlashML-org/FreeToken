"""llama.cpp's ggml-cuda kernels (vendored under ``csrc/ggml``) as a tvm-ffi module: prebuilt in the kernel cache, else compiled on first use.

Weights stay in their packed ggml blocks. ``mul_mat`` / ``mul_mat_id`` pick MMVQ or MMQ the way llama.cpp's mul_mat / mul_mat_id do.
"""

from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING

import torch

from .utils import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module

_ROOT = pathlib.Path(__file__).parent / "csrc" / "ggml"
# dtypes the kernels read and write; others go through fp32
_NATIVE = (torch.bfloat16, torch.float32)


def load_module(build_directory: str | None = None) -> Module:
    cuda_dir = _ROOT / "ggml-cuda"
    sources = [
        _ROOT / "freetoken" / "ggml_shim.cu",
        _ROOT / "freetoken" / "ggml_ops.cu",
        *(cuda_dir / f for f in ("mmvq.cu", "mmq.cu", "mmid.cu", "quantize.cu", "convert.cu", "getrows.cu")),
        *sorted((cuda_dir / "template-instances").glob("mmq-instance-*.cu")),
    ]
    return load_aot(
        "ggml",
        cuda_files=[str(s) for s in sources],
        extra_cuda_cflags=["-use_fast_math", "--extended-lambda"],
        extra_include_paths=[str(_ROOT / "include"), str(_ROOT / "src"), str(cuda_dir)],
        build_directory=build_directory,
    )


@functools.cache
def _module() -> Module:
    return load_module()


def _activation(x: torch.Tensor) -> torch.Tensor:
    return (x if x.dtype in _NATIVE else x.float()).contiguous()


def _writable(out: torch.Tensor) -> torch.Tensor:
    """``out``, or an fp32 buffer to copy into it when the kernels cannot write its dtype."""
    return out if out.dtype in _NATIVE else torch.empty(out.shape, dtype=torch.float32, device=out.device)


def _result_dtype(x: torch.Tensor, out_f32: bool) -> torch.dtype:
    return torch.bfloat16 if x.dtype == torch.bfloat16 and not out_f32 else torch.float32


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


def mul_mat(
    w: torch.Tensor, ggml_type: int, x: torch.Tensor, *, out_f32: bool = False, out: torch.Tensor | None = None, xq: torch.Tensor | None = None
) -> torch.Tensor:
    """``x[M, K] @ w[N, K]^T`` with ``w`` packed as ``[N, row_bytes]`` uint8; the result is in ``x``'s dtype unless ``out_f32``, or written into the ``[M, N]`` view ``out``.

    ``xq`` is ``quantize_q8_1(x)``: the vector kernel reads it instead of quantizing ``x`` again.
    """
    m, k = x.shape
    if out is None:
        out = torch.empty(m, w.shape[0], dtype=_result_dtype(x, out_f32), device=x.device)
    if not uses_mmvq(ggml_type, m):
        _check_mmq_tail(w, ggml_type, k)
    res = _writable(out)
    _module().mul_mat(w, ggml_type, _activation(x), res, xq)
    if res is not out:
        out.copy_(res)
    return out


def quantize_q8_1(x: torch.Tensor) -> torch.Tensor:
    """``x[M, K]`` as the vector kernel's q8_1 blocks, to share among GEMVs that read the same ``x``."""
    m, k = x.shape
    out = torch.empty(m, _q8_1_row_bytes(k), dtype=torch.uint8, device=x.device)
    _module().quantize_q8_1(_activation(x), out)
    return out


@functools.cache
def _q8_1_row_bytes(k: int) -> int:
    return _module().q8_1_row_bytes(k)


@functools.cache
def uses_mmvq(ggml_type: int, m: int) -> bool:
    """Whether ``mul_mat`` of ``m`` rows on this GPU runs the vector kernel, the one that reads ``xq``."""
    return _module().uses_mmvq(ggml_type, m)


def mul_mat_id(
    w: torch.Tensor, ggml_type: int, x: torch.Tensor, ids: torch.Tensor, *, out_f32: bool = False, ncols_max: int | None = None
) -> torch.Tensor:
    """Per-route expert GEMM over ``w[E, N, row_bytes]``: returns ``[T, k, N]``; ``x`` is ``[T, K]`` (shared by the k routes) or ``[T, k, K]``.

    ``ncols_max`` is an upper bound on the routes any one expert gets; without it the matrix kernel runs only the tiles that hold routes.
    """
    t, top_k = ids.shape
    out = torch.empty(t, top_k, w.shape[1], dtype=torch.float32 if out_f32 else x.dtype, device=x.device)
    if t > mmvq_max_tokens(ggml_type):
        _check_mmq_tail(w, ggml_type, x.shape[-1])
    res = _writable(out)
    _module().mul_mat_id(w, ggml_type, _activation(x.unsqueeze(1) if x.dim() == 2 else x), ids, res, ncols_max or -1)
    if res is not out:
        out.copy_(res)
    return out


def mul_mat_id_glu(w: torch.Tensor, ggml_type: int, x: torch.Tensor, ids: torch.Tensor, glu: str, limit: float = 0.0) -> torch.Tensor:
    """``glu(gate, up)`` per route in one pass for small ``T``: ``w[E, 2I, row_bytes]`` holds gate rows then up rows; returns fp32 ``[T, k, I]``.

    ``glu`` is one of ``swiglu``, ``geglu`` (tanh GELU), ``swiglu_oai`` (alpha 1.702, limit 7) and ``swiglu_clamp`` (alpha 1, clamped at ``limit``).
    """
    t, top_k = ids.shape
    out = torch.empty(t, top_k, w.shape[1] // 2, dtype=torch.float32, device=x.device)
    _module().mul_mat_id_glu(w, ggml_type, _activation(x), ids, out, glu, limit)
    return out


def moe_weighted_sum(y: torch.Tensor, weights: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``sum_j weights[t, j] * y[t, j, :]`` over fp32 ``y[T, k, N]``, accumulated in fp32."""
    out = torch.empty(y.shape[0], y.shape[2], dtype=dtype, device=y.device)
    res = _writable(out)
    _module().moe_weighted_sum(y, weights.float().contiguous(), res)
    if res is not out:
        out.copy_(res)
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
    out = torch.empty(ids.shape[0], k, dtype=dtype, device=w.device)
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
