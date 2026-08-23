import torch
import triton
import triton.language as tl

from ...gguf_gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gguf_gemm.utils import GGML_TYPE_IQ2_S, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def iq2_s_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    grid_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 82
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    qh = tl.load(block_ptrs + 66 + ib, mask=mask, other=0).to(tl.int32)
    grid_idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=mask, other=0).to(tl.int32)
    grid_idx = grid_idx | ((qh << (8 - 2 * il)) & 0x300)
    signs = tl.load(block_ptrs + 34 + 4 * ib + il, mask=mask, other=0)
    scale_byte = tl.load(block_ptrs + 74 + ib, mask=mask, other=0)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    scale = (((scale_byte >> (4 * (il // 2))) & 0x0F).to(tl.float32) + 0.5) * 0.25
    grid = tl.load(grid_ptr + grid_idx * 8 + k, mask=mask, other=0).to(tl.float32)
    sign = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
    out = grid * sign * d * scale
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_iq2_s_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_dequantize_kernel(
        iq2_s_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_IQ2_S,
        extra_args=(tables["iq2s_grid"],),
    )
