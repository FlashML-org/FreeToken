import torch
import triton
import triton.language as tl

from ...gguf_gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gguf_gemm.utils import GGML_TYPE_IQ1_M, load_u16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def iq1_m_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    grid_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 56
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    il = rem // 8
    k = rem % 8

    sc0 = load_u16_from_u8(block_ptrs + 48 + 0, mask)
    sc1 = load_u16_from_u8(block_ptrs + 48 + 2, mask)
    sc2 = load_u16_from_u8(block_ptrs + 48 + 4, mask)
    sc3 = load_u16_from_u8(block_ptrs + 48 + 6, mask)
    base_bits = (
        (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
    )
    qh = tl.load(block_ptrs + 32 + 2 * ib + (il // 2), mask=mask, other=0).to(tl.int32)
    idx = tl.load(block_ptrs + 4 * ib + il, mask=mask, other=0).to(tl.int32)
    idx = idx | (((qh >> (4 * (il % 2))) & 0x07) << 8)
    delta_num = tl.where((qh & (0x08 << (4 * (il % 2)))) != 0, -9.0, -7.0)
    ib16 = 2 * ib + (il // 2)
    sc_sel = tl.where(
        ib16 // 4 == 0,
        sc0,
        tl.where(ib16 // 4 == 1, sc1, tl.where(ib16 // 4 == 2, sc2, sc3)),
    )
    base = tl.cast(base_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
    scale = (
        base
        * (2.0 * (((sc_sel >> (3 * (ib16 % 4))) & 0x07).to(tl.float32)) + 1.0)
        * 0.125
    )
    grid = tl.load(grid_ptr + idx * 8 + k, mask=mask, other=0).to(tl.float32)
    out = ((grid * 8.0) + delta_num) * scale
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_iq1_m_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_dequantize_kernel(
        iq1_m_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_IQ1_M,
        extra_args=(tables["iq1s_grid"],),
    )
