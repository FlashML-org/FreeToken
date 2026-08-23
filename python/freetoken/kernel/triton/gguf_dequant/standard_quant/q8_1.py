import torch
import triton
import triton.language as tl

from ...gguf_gemm.utils import GGML_TYPE_Q8_1, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def q8_1_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 32) * 36
    pos = offs % 32
    q = tl.load(block_ptrs + 4 + pos, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    out = tl.cast(q, tl.int8, bitcast=True).to(tl.float32) * d
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q8_1_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q8_1_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q8_1)
