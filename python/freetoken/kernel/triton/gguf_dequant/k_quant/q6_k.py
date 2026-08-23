import torch
import triton
import triton.language as tl

from ...gguf_gemm.utils import GGML_TYPE_Q6_K, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def q6_k_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 210
    pos = offs % 256
    half_block = pos // 128
    rem = pos % 128
    group = rem // 32
    half = (rem % 32) // 16
    sub_idx = rem % 16
    il = 16 * half + sub_idx

    scale = tl.load(
        block_ptrs + 192 + 8 * half_block + 2 * group + half,
        mask=mask,
        other=0,
    )
    d = load_f16_from_u8(block_ptrs + 208, mask).to(tl.float32)
    scale = tl.cast(scale, tl.int8, bitcast=True).to(tl.float32) * d
    qh = tl.load(block_ptrs + 128 + 32 * half_block + il, mask=mask, other=0)
    ql_idx = 64 * half_block + tl.where((group % 2) == 1, 32, 0) + il
    ql = tl.load(block_ptrs + ql_idx, mask=mask, other=0)
    q = tl.where(group < 2, ql & 0x0F, ql >> 4)
    q = q.to(tl.float32) + 16.0 * (((qh >> (2 * group)) & 0x03).to(tl.float32))
    out = (q - 32.0) * scale
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q6_k_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q6_k_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q6_K)
