import torch
import triton
import triton.language as tl

from ..utils import (
    GGML_TYPE_IQ2_XS,
    load_f16_from_u8,
    load_u16_from_u8,
    load_x_chunk,
    run_triton_kernel,
)
from .iq_tables import get_iq_table_tensors


@triton.jit
def iq2_xs_gemm_kernel(
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
    sign_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_8 = tl.arange(0, 8)
    sign_mask = (1 << offs_8).to(tl.uint8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 74

        for ib in range(8):
            scale_byte = tl.load(block_ptrs + 66 + ib, mask=n_mask, other=0)
            for il in range(4):
                q2 = load_u16_from_u8(block_ptrs + 2 + 2 * (4 * ib + il), n_mask).to(
                    tl.int32
                )
                grid_idx = q2 & 0x1FF
                signs = tl.load(sign_ptr + (q2 >> 9), mask=n_mask, other=0)
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
                scale = (
                    ((scale_byte >> (4 * (il // 2))) & 0x0F).to(x_dtype) + 0.5
                ) * 0.25
                grid = tl.load(
                    grid_ptr + grid_idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                sign = tl.where((signs[:, None] & sign_mask[None, :]) != 0, -1, 1).to(
                    x_dtype
                )
                q_tile = (grid * sign * (d * scale).to(x_dtype)[:, None]).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_iq2_xs_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_triton_kernel(
        iq2_xs_gemm_kernel,
        W,
        X,
        row,
        GGML_TYPE_IQ2_XS,
        extra_args=(tables["iq2xs_grid"], tables["ksigns_iq2xs"]),
    )
