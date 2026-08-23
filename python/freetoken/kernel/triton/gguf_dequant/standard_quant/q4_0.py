import torch
import triton
import triton.language as tl

from ...gguf_gemm.utils import GGML_TYPE_Q4_0, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def q4_0_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 32) * 18
    pos = offs % 32
    packed = tl.load(block_ptrs + 2 + (pos % 16), mask=mask, other=0)
    q = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    out = (q.to(tl.float32) - 8.0) * d
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q4_0_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q4_0_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q4_0)
