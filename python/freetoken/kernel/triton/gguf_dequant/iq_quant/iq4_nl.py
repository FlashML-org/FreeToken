# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-gguf-plugin; see
# third_party/vllm-gguf-plugin/ATTRIBUTION.md and PROVENANCE.json.
# Modified by Maxritz: relocated relative GEMM imports for FreeToken.

import torch
import triton
import triton.language as tl

from ...gguf_gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gguf_gemm.utils import GGML_TYPE_IQ4_NL, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def iq4_nl_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    values_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 32) * 18
    pos = offs % 32
    packed = tl.load(block_ptrs + 2 + (pos % 16), mask=mask, other=0)
    nibble = tl.where(pos < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    value = tl.load(values_ptr + nibble.to(tl.int32), mask=mask, other=0)
    out = value.to(tl.float32) * d
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_iq4_nl_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_dequantize_kernel(
        iq4_nl_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_IQ4_NL,
        extra_args=(tables["kvalues_iq4nl"],),
    )
