# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-gguf-plugin; see
# third_party/vllm-gguf-plugin/ATTRIBUTION.md and PROVENANCE.json.
# Modified by Maxritz: relocated relative GEMM imports for FreeToken.

import torch
import triton
import triton.language as tl

from ...gguf_gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gguf_gemm.utils import GGML_TYPE_IQ3_S, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def iq3_s_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    grid_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 110
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8
    half = k // 4
    k4 = k % 4

    qh = tl.load(block_ptrs + 66 + ib, mask=mask, other=0).to(tl.int32)
    idx = tl.load(block_ptrs + 2 + 8 * ib + 2 * il + half, mask=mask, other=0).to(
        tl.int32
    )
    idx = idx | tl.where(
        half == 0,
        (qh << (8 - 2 * il)) & 0x100,
        (qh << (7 - 2 * il)) & 0x100,
    )
    signs = tl.load(block_ptrs + 74 + 4 * ib + il, mask=mask, other=0)
    scale_byte = tl.load(block_ptrs + 106 + (ib // 2), mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    scale = (scale_byte >> (4 * (ib % 2))) & 0x0F
    dscale = d * (1.0 + 2.0 * scale.to(tl.float32))
    grid = tl.load(grid_ptr + idx * 4 + k4, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * dscale
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_iq3_s_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_dequantize_kernel(
        iq3_s_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_IQ3_S,
        extra_args=(tables["iq3s_grid"],),
    )
