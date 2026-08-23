import torch
import triton
import triton.language as tl

from ..utils import (
    GGML_TYPE_IQ3_XXS,
    load_f16_from_u8,
    load_u32_from_u8,
    load_x_chunk,
    run_triton_kernel,
)
from .iq_tables import get_iq_table_tensors


@triton.jit
def iq3_xxs_gemm_kernel(
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
    offs_4 = tl.arange(0, 4)
    sign_mask_lo = (1 << offs_4).to(tl.uint8)
    sign_mask_hi = (1 << (offs_4 + 4)).to(tl.uint8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 98

        for ib in range(8):
            aux32 = load_u32_from_u8(block_ptrs + 66 + 4 * ib, n_mask)
            for il in range(4):
                signs = tl.load(
                    sign_ptr + ((aux32 >> (7 * il)) & 127), mask=n_mask, other=0
                )
                idx1 = tl.load(
                    block_ptrs + 2 + 8 * ib + 2 * il + 0, mask=n_mask, other=0
                ).to(tl.int32)
                idx2 = tl.load(
                    block_ptrs + 2 + 8 * ib + 2 * il + 1, mask=n_mask, other=0
                ).to(tl.int32)
                x1 = load_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_m,
                    m,
                    kb * 256 + 32 * ib + 8 * il,
                    CHUNK=4,
                )
                x2 = load_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_m,
                    m,
                    kb * 256 + 32 * ib + 8 * il + 4,
                    CHUNK=4,
                )
                x_dtype = x1.dtype
                d = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
                dscale = (d * ((aux32 >> 28).to(x_dtype) + 0.5) * 0.5).to(x_dtype)
                grid1 = tl.load(
                    grid_ptr + idx1[:, None] * 4 + offs_4[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                grid2 = tl.load(
                    grid_ptr + idx2[:, None] * 4 + offs_4[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                q1 = (
                    grid1
                    * tl.where((signs[:, None] & sign_mask_lo[None, :]) != 0, -1, 1).to(
                        x_dtype
                    )
                    * dscale[:, None]
                ).to(x_dtype)
                q2 = (
                    grid2
                    * tl.where((signs[:, None] & sign_mask_hi[None, :]) != 0, -1, 1).to(
                        x_dtype
                    )
                    * dscale[:, None]
                ).to(x_dtype)
                acc += tl.sum(x1[:, None, :] * q1[None, :, :], axis=2)
                acc += tl.sum(x2[:, None, :] * q2[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_iq3_xxs_triton(
    W: torch.Tensor, X: torch.Tensor, row: int
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_triton_kernel(
        iq3_xxs_gemm_kernel,
        W,
        X,
        row,
        GGML_TYPE_IQ3_XXS,
        extra_args=(tables["iq3xxs_grid"], tables["ksigns_iq2xs"]),
    )
