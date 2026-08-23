from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..gguf_gemm.utils import (
    BLOCK_BYTES_BY_TYPE,
    BLOCK_QK_BY_TYPE,
    TRITON_SUPPORTED_TYPES,
)

TRITON_DEQUANT_BLOCK_SIZE = 256
TRITON_DEQUANT_SUPPORTED_TYPES = TRITON_SUPPORTED_TYPES
TRITON_DEQUANT_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


@triton.jit
def dequant_offsets(total, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    return offs, offs < total


@triton.jit
def load_scale_min_k4_vector(scales_ptr, j, mask):
    scale_lo = tl.load(scales_ptr + j, mask=mask & (j < 4), other=0) & 0x3F
    min_lo = tl.load(scales_ptr + j + 4, mask=mask & (j < 4), other=0) & 0x3F

    hi = tl.load(scales_ptr + j + 4, mask=mask & (j >= 4), other=0)
    lo_d = tl.load(scales_ptr + j - 4, mask=mask & (j >= 4), other=0)
    lo_m = tl.load(scales_ptr + j, mask=mask & (j >= 4), other=0)
    scale_hi = (hi & 0x0F) | ((lo_d >> 6) << 4)
    min_hi = (hi >> 4) | ((lo_m >> 6) << 4)

    return tl.where(j < 4, scale_lo, scale_hi), tl.where(j < 4, min_lo, min_hi)


@triton.jit
def load_q3_k_scale_vector(scales_ptr, is_idx, mask):
    is_0 = is_idx < 4
    is_1 = (is_idx >= 4) & (is_idx < 8)
    is_2 = (is_idx >= 8) & (is_idx < 12)
    is_3 = is_idx >= 12

    lo0 = tl.load(scales_ptr + is_idx, mask=mask & is_0, other=0)
    hi0 = tl.load(scales_ptr + is_idx + 8, mask=mask & is_0, other=0)
    scale0 = (lo0 & 0x0F) | (((hi0 >> 0) & 0x03) << 4)

    lo1 = tl.load(scales_ptr + is_idx, mask=mask & is_1, other=0)
    hi1 = tl.load(scales_ptr + is_idx + 4, mask=mask & is_1, other=0)
    scale1 = (lo1 & 0x0F) | (((hi1 >> 2) & 0x03) << 4)

    lo2 = tl.load(scales_ptr + is_idx - 8, mask=mask & is_2, other=0)
    hi2 = tl.load(scales_ptr + is_idx, mask=mask & is_2, other=0)
    scale2 = ((lo2 >> 4) & 0x0F) | (((hi2 >> 4) & 0x03) << 4)

    lo3 = tl.load(scales_ptr + is_idx - 8, mask=mask & is_3, other=0)
    hi3 = tl.load(scales_ptr + is_idx - 4, mask=mask & is_3, other=0)
    scale3 = ((lo3 >> 4) & 0x0F) | (((hi3 >> 6) & 0x03) << 4)

    return tl.where(
        is_0, scale0, tl.where(is_1, scale1, tl.where(is_2, scale2, scale3))
    )


def validate_dequant_args(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, int, torch.dtype]:
    quant_type = int(quant_type)
    if quant_type not in TRITON_DEQUANT_SUPPORTED_TYPES:
        raise ValueError(f"Unsupported Triton dequant quant type: {quant_type}")
    if not W.is_cuda:
        raise ValueError("Triton dequant kernels require CUDA tensors")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")

    dtype = dtype or torch.float16
    if dtype not in TRITON_DEQUANT_SUPPORTED_DTYPES:
        raise TypeError(
            "Triton dequant kernels support torch.float16, torch.bfloat16, and "
            f"torch.float32 outputs, got {dtype}"
        )

    total = int(m) * int(n)
    if total < 0:
        raise ValueError(f"Invalid dequantized shape ({m}, {n})")

    block_qk = BLOCK_QK_BY_TYPE[quant_type]
    if total % block_qk != 0:
        raise ValueError(
            f"Dequantized element count {total} must be divisible by "
            f"{block_qk} for quant type {quant_type}"
        )

    expected_bytes = total // block_qk * BLOCK_BYTES_BY_TYPE[quant_type]
    if W.numel() < expected_bytes:
        raise ValueError(
            f"Quantized weights have {W.numel()} bytes, but quant type "
            f"{quant_type} and shape ({m}, {n}) require {expected_bytes} bytes"
        )

    return W.contiguous(), total, dtype


def run_dequantize_kernel(
    kernel,
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None,
    quant_type: int,
    extra_args: tuple = (),
) -> torch.Tensor:
    W, total, dtype = validate_dequant_args(W, quant_type, m, n, dtype)
    Y = torch.empty((m, n), device=W.device, dtype=dtype)

    if total == 0:
        return Y

    grid = (triton.cdiv(total, TRITON_DEQUANT_BLOCK_SIZE),)
    kernel[grid](
        W,
        Y,
        total,
        *extra_args,
        BLOCK_SIZE=TRITON_DEQUANT_BLOCK_SIZE,
        num_warps=4,
    )
    return Y
