import torch
import triton
import triton.language as tl

from ...gguf_gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gguf_gemm.utils import GGML_TYPE_IQ2_XXS, load_f16_from_u8, load_u32_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def iq2_xxs_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    grid_ptr,
    sign_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 66
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    q2_base = block_ptrs + 2 + 8 * ib
    aux32 = load_u32_from_u8(q2_base + 4, mask)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    dscale = d * ((aux32 >> 28).to(tl.float32) + 0.5) * 0.25
    grid_idx = tl.load(q2_base + il, mask=mask, other=0).to(tl.int32)
    signs = tl.load(sign_ptr + ((aux32 >> (7 * il)) & 127), mask=mask, other=0)
    grid = tl.load(grid_ptr + grid_idx * 8 + k, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * dscale
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_iq2_xxs_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_dequantize_kernel(
        iq2_xxs_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_IQ2_XXS,
        extra_args=(tables["iq2xxs_grid"], tables["ksigns_iq2xs"]),
    )
