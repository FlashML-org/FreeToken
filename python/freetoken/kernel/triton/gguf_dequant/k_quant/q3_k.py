import torch
import triton
import triton.language as tl

from ...gguf_gemm.utils import GGML_TYPE_Q3_K, load_f16_from_u8
from ..utils import (
    dequant_offsets,
    load_q3_k_scale_vector,
    run_dequantize_kernel,
)


@triton.jit
def q3_k_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 110
    pos = offs % 256
    group = pos // 128
    rem = pos % 128
    j = rem // 32
    half = (rem % 32) // 16
    sub_idx = rem % 16

    is_idx = 8 * group + 2 * j + half
    hm_mask = 1 << (4 * group + j)
    scale = load_q3_k_scale_vector(block_ptrs + 96, is_idx, mask)
    ql = tl.load(block_ptrs + 32 + 32 * group + 16 * half + sub_idx, mask=mask, other=0)
    hm = tl.load(block_ptrs + 16 * half + sub_idx, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 108, mask).to(tl.float32)
    q = ((ql >> (2 * j)) & 0x03).to(tl.int16)
    q = (q - tl.where((hm & hm_mask) != 0, 0, 4).to(tl.int16)).to(tl.float32)
    out = q * ((scale.to(tl.int16) - 32).to(tl.float32) * d)
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q3_k_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q3_k_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q3_K)
