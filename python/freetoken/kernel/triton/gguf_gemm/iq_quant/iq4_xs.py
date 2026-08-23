import torch
import triton
import triton.language as tl

from ..utils import (
    GGML_TYPE_IQ4_XS,
    load_f16_from_u8,
    load_u16_from_u8,
    load_x_chunk,
    run_triton_kernel,
)
from .iq_tables import get_iq_table_tensors


@triton.jit
def iq4_xs_gemm_kernel(
    x_ptr,
    w_u8_ptr,
    y_ptr,
    m,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_ym,
    stride_yn,
    values_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_nibble = tl.arange(0, 16)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 136
        scales_h = load_u16_from_u8(block_ptrs + 2, n_mask)

        for ib in range(8):
            packed = tl.load(
                block_ptrs[:, None] + 8 + 16 * ib + offs_nibble[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            x1 = load_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_m,
                m,
                kb * 256 + 32 * ib,
                CHUNK=16,
            )
            x2 = load_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_m,
                m,
                kb * 256 + 32 * ib + 16,
                CHUNK=16,
            )
            x_dtype = x1.dtype
            d = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
            scales_l = tl.load(block_ptrs + 4 + (ib // 2), mask=n_mask, other=0).to(
                tl.int32
            )
            scale = (
                (
                    ((scales_l >> (4 * (ib % 2))) & 0x0F)
                    | (((scales_h.to(tl.int32) >> (2 * ib)) & 0x03) << 4)
                ).to(tl.int16)
                - 32
            ).to(x_dtype)
            low = tl.load(values_ptr + (packed & 0x0F).to(tl.int32)).to(x_dtype)
            high = tl.load(values_ptr + ((packed >> 4) & 0x0F).to(tl.int32)).to(x_dtype)
            dscale = (d * scale).to(x_dtype)
            q1 = (low * dscale[:, None]).to(x_dtype)
            q2 = (high * dscale[:, None]).to(x_dtype)
            acc = tl.dot(x1, tl.trans(q1), acc=acc)
            acc = tl.dot(x2, tl.trans(q2), acc=acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_iq4_xs_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_triton_kernel(
        iq4_xs_gemm_kernel,
        W,
        X,
        row,
        GGML_TYPE_IQ4_XS,
        extra_args=(tables["kvalues_iq4nl"],),
    )
