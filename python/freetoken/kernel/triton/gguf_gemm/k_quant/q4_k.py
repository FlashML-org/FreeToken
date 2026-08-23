import torch
import triton
import triton.language as tl

from ..utils import (
    GGML_TYPE_Q4_K,
    load_f16_from_u8,
    load_x_chunk,
    run_triton_kernel,
)


@triton.jit
def q4_k_gemm_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_q = tl.arange(0, 32)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 144
        scales_ptr = block_ptrs + 4

        for chunk in range(8):
            group = chunk // 2
            part = chunk % 2
            j = 2 * group + part
            if j < 4:
                scale_q = tl.load(scales_ptr + j, mask=n_mask, other=0) & 0x3F
                min_q = tl.load(scales_ptr + j + 4, mask=n_mask, other=0) & 0x3F
            else:
                hi = tl.load(scales_ptr + j + 4, mask=n_mask, other=0)
                lo_d = tl.load(scales_ptr + j - 4, mask=n_mask, other=0)
                lo_m = tl.load(scales_ptr + j, mask=n_mask, other=0)
                scale_q = (hi & 0x0F) | ((lo_d >> 6) << 4)
                min_q = (hi >> 4) | ((lo_m >> 6) << 4)

            x_tile = load_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_m,
                m,
                kb * 256 + 64 * group + 32 * part,
                CHUNK=32,
            )
            x_dtype = x_tile.dtype
            dall = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
            dmin = load_f16_from_u8(block_ptrs + 2, n_mask).to(x_dtype)
            q = tl.load(
                block_ptrs[:, None] + 16 + 32 * group + offs_q[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            q = (q & 0x0F) if part == 0 else (q >> 4)
            q_tile = q.to(x_dtype) * (scale_q.to(x_dtype)[:, None] * dall[:, None])
            q_tile = q_tile - min_q.to(x_dtype)[:, None] * dmin[:, None]
            acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_q4_k_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    return run_triton_kernel(q4_k_gemm_kernel, W, X, row, GGML_TYPE_Q4_K)
