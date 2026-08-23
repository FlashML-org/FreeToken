import torch
import triton
import triton.language as tl

from ..utils import GGML_TYPE_Q6_K, load_f16_from_u8, load_x_chunk, run_triton_kernel


@triton.jit
def q6_k_gemm_kernel(
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
    offs_l = tl.arange(0, 16)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 210

        for chunk in range(16):
            half_block = chunk // 8
            rem = chunk % 8
            group = rem // 2
            half = rem % 2
            il = 16 * half + offs_l
            scale = tl.load(
                block_ptrs + 192 + 8 * half_block + 2 * group + half,
                mask=n_mask,
                other=0,
            )
            x_tile = load_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_m,
                m,
                kb * 256 + 128 * half_block + 32 * group + 16 * half,
                CHUNK=16,
            )
            x_dtype = x_tile.dtype
            d = load_f16_from_u8(block_ptrs + 208, n_mask).to(x_dtype)
            scale = tl.cast(scale, tl.int8, bitcast=True).to(x_dtype) * d
            qh = tl.load(
                block_ptrs[:, None] + 128 + 32 * half_block + il[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            ql_idx = 64 * half_block + (32 if group % 2 == 1 else 0) + il
            ql = tl.load(
                block_ptrs[:, None] + ql_idx[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            q = (ql & 0x0F) if group < 2 else (ql >> 4)
            q = q.to(x_dtype) + 16.0 * (((qh >> (2 * group)) & 0x03).to(x_dtype))
            q_tile = (q - 32.0) * scale[:, None]
            acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_q6_k_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    return run_triton_kernel(q6_k_gemm_kernel, W, X, row, GGML_TYPE_Q6_K)
