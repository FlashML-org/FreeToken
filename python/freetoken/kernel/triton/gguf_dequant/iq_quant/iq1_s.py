import torch
import triton
import triton.language as tl

from ...gguf_gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gguf_gemm.utils import GGML_TYPE_IQ1_S, load_f16_from_u8, load_u16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def iq1_s_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    grid_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 50
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    qh = load_u16_from_u8(block_ptrs + 34 + 2 * ib, mask)
    idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=mask, other=0).to(tl.int32)
    idx = idx | ((((qh >> (3 * il)) & 0x7).to(tl.int32)) << 8)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    scale = d * (2.0 * ((qh >> 12) & 0x7).to(tl.float32) + 1.0) * 0.125
    delta_num = tl.where((qh & 0x8000) != 0, -9.0, -7.0)
    grid = tl.load(grid_ptr + idx * 8 + k, mask=mask, other=0).to(tl.float32)
    out = ((grid * 8.0) + delta_num) * scale
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_iq1_s_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_dequantize_kernel(
        iq1_s_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_IQ1_S,
        extra_args=(tables["iq1s_grid"],),
    )
