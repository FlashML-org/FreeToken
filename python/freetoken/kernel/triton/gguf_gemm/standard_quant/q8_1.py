import torch
import triton
import triton.language as tl

from ..utils import GGML_TYPE_Q8_1, load_f16_from_u8, load_x_tile, run_triton_kernel


@triton.jit
def q8_1_gemm_kernel(
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
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_nibble = tl.arange(0, 16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        x_tile, cur_kb, kb_mask = load_x_tile(
            x_ptr,
            m,
            num_k_blocks,
            stride_xm,
            stride_xk,
            offs_m,
            kb_start,
            offs_kb,
            offs_nibble,
            BLOCK_M=BLOCK_M,
            BLOCK_K_BLOCKS=BLOCK_K_BLOCKS,
        )
        x_dtype = x_tile.dtype

        block_ptrs = w_row_ptrs + cur_kb[None, :] * 36
        scale_mask = (offs_n[:, None] < n) & kb_mask[None, :]
        d = load_f16_from_u8(block_ptrs + 0, scale_mask).to(x_dtype)
        q_low = tl.load(
            block_ptrs[:, :, None] + 4 + offs_nibble[None, None, :],
            mask=(offs_n[:, None, None] < n) & kb_mask[None, :, None],
            other=0,
        )
        q_high = tl.load(
            block_ptrs[:, :, None] + 20 + offs_nibble[None, None, :],
            mask=(offs_n[:, None, None] < n) & kb_mask[None, :, None],
            other=0,
        )
        q_tile = tl.reshape(
            tl.join(
                tl.cast(q_low, tl.int8, bitcast=True).to(x_dtype) * d[:, :, None],
                tl.cast(q_high, tl.int8, bitcast=True).to(x_dtype) * d[:, :, None],
            ),
            (BLOCK_N, BLOCK_K_BLOCKS * 32),
        )
        acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_q8_1_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    return run_triton_kernel(q8_1_gemm_kernel, W, X, row, GGML_TYPE_Q8_1)
