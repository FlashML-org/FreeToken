# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-gguf-plugin; see
# third_party/vllm-gguf-plugin/ATTRIBUTION.md and PROVENANCE.json.
# Modified by Maxritz: relocated relative GEMM imports for FreeToken.

import torch
import triton
import triton.language as tl

from ...gguf_gemm.utils import GGML_TYPE_Q5_1, load_f16_from_u8, load_u32_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def q5_1_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 32) * 24
    pos = offs % 32
    packed = tl.load(block_ptrs + 8 + (pos % 16), mask=mask, other=0)
    qh = load_u32_from_u8(block_ptrs + 4, mask)
    q = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    q = q | (((qh >> pos) & 0x01) << 4)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    m0 = load_f16_from_u8(block_ptrs + 2, mask).to(tl.float32)
    out = q.to(tl.float32) * d + m0
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q5_1_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q5_1_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q5_1)
