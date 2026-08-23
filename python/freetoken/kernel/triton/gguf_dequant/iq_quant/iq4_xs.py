import torch
import triton
import triton.language as tl

from ...gguf_gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gguf_gemm.utils import GGML_TYPE_IQ4_XS, load_f16_from_u8, load_u16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def iq4_xs_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    values_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 256) * 136
    pos = offs % 256
    ib = pos // 32
    rem = pos % 32
    packed = tl.load(block_ptrs + 8 + 16 * ib + (rem % 16), mask=mask, other=0)
    nibble = tl.where(rem < 16, packed & 0x0F, (packed >> 4) & 0x0F)
    scales_h = load_u16_from_u8(block_ptrs + 2, mask)
    scales_l = tl.load(block_ptrs + 4 + (ib // 2), mask=mask, other=0).to(tl.int32)
    scale = (
        ((scales_l >> (4 * (ib % 2))) & 0x0F)
        | (((scales_h.to(tl.int32) >> (2 * ib)) & 0x03) << 4)
    ).to(tl.int16)
    scale = (scale - 32).to(tl.float32)
    d = load_f16_from_u8(block_ptrs + 0, mask).to(tl.float32)
    value = tl.load(values_ptr + nibble.to(tl.int32), mask=mask, other=0)
    out = value.to(tl.float32) * d * scale
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_iq4_xs_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_dequantize_kernel(
        iq4_xs_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_IQ4_XS,
        extra_args=(tables["kvalues_iq4nl"],),
    )
