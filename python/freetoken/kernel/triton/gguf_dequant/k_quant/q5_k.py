# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-gguf-plugin; see
# third_party/vllm-gguf-plugin/ATTRIBUTION.md and PROVENANCE.json.
# Modified by Maxritz: relocated relative GEMM imports for FreeToken.

import torch
import triton
import triton.language as tl

from ...gguf_gemm.utils import GGML_TYPE_Q5_K, load_f16_from_u8
from ..utils import (
    dequant_offsets,
    load_scale_min_k4_vector,
    run_dequantize_kernel,
)


@triton.jit
def q5_k_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 176
    pos = offs % 256
    group = pos // 64
    rem = pos % 64
    part = rem // 32
    q_idx = rem % 32
    j = 2 * group + part

    scale_q, min_q = load_scale_min_k4_vector(block_ptrs + 4, j, mask)
    qh = tl.load(block_ptrs + 16 + q_idx, mask=mask, other=0)
    q = tl.load(block_ptrs + 48 + 32 * group + q_idx, mask=mask, other=0)
    q = tl.where(part == 0, q & 0x0F, q >> 4)
    q = q.to(tl.float32) + 16.0 * ((qh & (1 << (2 * group + part))) != 0).to(tl.float32)
    dall = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    dmin = load_f16_from_u8(block_ptrs + 2, mask).to(tl.float32)
    out = q * (scale_q.to(tl.float32) * dall)
    out = out - min_q.to(tl.float32) * dmin
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q5_k_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q5_k_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q5_K)
