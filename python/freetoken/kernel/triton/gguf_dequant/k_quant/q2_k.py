# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-gguf-plugin; see
# third_party/vllm-gguf-plugin/ATTRIBUTION.md and PROVENANCE.json.
# Modified by Maxritz: relocated relative GEMM imports for FreeToken.

import torch
import triton
import triton.language as tl

from ...gguf_gemm.utils import GGML_TYPE_Q2_K, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def q2_k_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 84
    pos = offs % 256
    group = pos // 128
    rem = pos % 128
    part = rem // 32
    q_idx = rem % 32

    scale_idx = 8 * group + 2 * part + tl.where(q_idx >= 16, 1, 0)
    q = tl.load(block_ptrs + 16 + 32 * group + q_idx, mask=mask, other=0)
    scale_byte = tl.load(block_ptrs + scale_idx, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 80, mask).to(tl.float32)
    dmin = load_f16_from_u8(block_ptrs + 82, mask).to(tl.float32)
    scale = (scale_byte & 0x0F).to(tl.float32) * d
    minv = (scale_byte >> 4).to(tl.float32) * dmin
    out = (((q >> (2 * part)) & 0x03).to(tl.float32) * scale) - minv
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q2_k_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q2_k_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q2_K)
