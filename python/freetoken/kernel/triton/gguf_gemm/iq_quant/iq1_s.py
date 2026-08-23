import torch
import triton
import triton.language as tl

from ..utils import (
    GGML_TYPE_IQ1_S,
    load_f16_from_u8,
    load_u16_from_u8,
    load_x_chunk,
    run_triton_kernel,
)
from .iq_tables import get_iq_table_tensors


@triton.jit
def iq1_s_gemm_kernel(
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
    grid_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_8 = tl.arange(0, 8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 50

        for ib in range(8):
            qh = load_u16_from_u8(block_ptrs + 34 + 2 * ib, n_mask)
            delta_num = tl.where((qh & 0x8000) != 0, -9, -7).to(tl.int16)
            for il in range(4):
                idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=n_mask, other=0).to(
                    tl.int32
                )
                idx = idx | ((((qh >> (3 * il)) & 0x7).to(tl.int32)) << 8)
                x_tile = load_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_m,
                    m,
                    kb * 256 + 32 * ib + 8 * il,
                    CHUNK=8,
                )
                x_dtype = x_tile.dtype
                d = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
                scale = (d * (2 * ((qh >> 12) & 0x7).to(x_dtype) + 1.0) * 0.125).to(
                    x_dtype
                )
                grid = tl.load(
                    grid_ptr + idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                q_tile = (
                    (((grid.to(tl.int16) * 8) + delta_num[:, None]).to(x_dtype))
                    * scale[:, None]
                ).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_iq1_s_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_triton_kernel(
        iq1_s_gemm_kernel,
        W,
        X,
        row,
        GGML_TYPE_IQ1_S,
        extra_args=(tables["iq1s_grid"],),
    )
