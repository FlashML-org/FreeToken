import torch
import triton
import triton.language as tl

from ..utils import GGML_TYPE_Q3_K, load_f16_from_u8, load_x_chunk, run_triton_kernel


@triton.jit
def q3_k_gemm_kernel(
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
        block_ptrs = w_row_ptrs + kb * 110

        for chunk in range(16):
            group = chunk // 8
            rem = chunk % 8
            j = rem // 2
            half = rem % 2
            hm_mask = 1 << (4 * group + j)
            is_idx = 8 * group + 2 * j + half
            scales_ptr = block_ptrs + 96
            if is_idx < 4:
                lo = tl.load(scales_ptr + is_idx, mask=n_mask, other=0)
                hi = tl.load(scales_ptr + is_idx + 8, mask=n_mask, other=0)
                scale = (lo & 0x0F) | (((hi >> 0) & 0x03) << 4)
            elif is_idx < 8:
                lo = tl.load(scales_ptr + is_idx, mask=n_mask, other=0)
                hi = tl.load(scales_ptr + is_idx + 4, mask=n_mask, other=0)
                scale = (lo & 0x0F) | (((hi >> 2) & 0x03) << 4)
            elif is_idx < 12:
                lo = tl.load(scales_ptr + is_idx - 8, mask=n_mask, other=0)
                hi = tl.load(scales_ptr + is_idx, mask=n_mask, other=0)
                scale = ((lo >> 4) & 0x0F) | (((hi >> 4) & 0x03) << 4)
            else:
                lo = tl.load(scales_ptr + is_idx - 8, mask=n_mask, other=0)
                hi = tl.load(scales_ptr + is_idx - 4, mask=n_mask, other=0)
                scale = ((lo >> 4) & 0x0F) | (((hi >> 6) & 0x03) << 4)

            x_tile = load_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_m,
                m,
                kb * 256 + 128 * group + 32 * j + 16 * half,
                CHUNK=16,
            )
            x_dtype = x_tile.dtype
            d = load_f16_from_u8(block_ptrs + 108, n_mask).to(x_dtype)
            ql = tl.load(
                block_ptrs[:, None] + 32 + 32 * group + 16 * half + offs_l[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            hm = tl.load(
                block_ptrs[:, None] + 16 * half + offs_l[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            q = ((ql >> (2 * j)) & 0x03).to(tl.int16)
            q = (q - tl.where((hm & hm_mask) != 0, 0, 4).to(tl.int16)).to(x_dtype)
            dq = ((scale.to(tl.int16) - 32).to(x_dtype) * d)[:, None]
            q_tile = q * dq
            acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_q3_k_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    return run_triton_kernel(q3_k_gemm_kernel, W, X, row, GGML_TYPE_Q3_K)
