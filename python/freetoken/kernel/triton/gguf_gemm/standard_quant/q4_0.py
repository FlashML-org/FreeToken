import torch
import triton
import triton.language as tl

Q4_0_BLOCK_SIZE = 32
Q4_0_BLOCK_BYTES = 18

Q4_0_BLOCK_M = 32
Q4_0_BLOCK_N = 128
Q4_0_BLOCK_K_BLOCKS = 4
Q4_0_NUM_WARPS = 2
Q4_0_NUM_STAGES = 2
Q4_0_SUPPORTED_ACTIVATION_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


@triton.jit
def q4_0_gemm_kernel(
    x_ptr,
    w_u8_ptr,
    y_ptr,
    m,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_w_u8n,
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
    offs_byte = tl.arange(0, 16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_row_ptrs = x_ptr + offs_m[:, None, None] * stride_xm
    w_packed_row_ptrs = w_u8_ptr + offs_n[:, None, None] * stride_w_u8n
    w_block_row_ptrs = w_u8_ptr + offs_n[:, None] * stride_w_u8n

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks

        x_k_low = cur_kb[None, :, None] * 32 + offs_byte[None, None, :]
        x_k_high = x_k_low + 16

        x_even_ptrs = x_row_ptrs + x_k_low * stride_xk
        x_odd_ptrs = x_row_ptrs + x_k_high * stride_xk
        x_mask = (offs_m[:, None, None] < m) & kb_mask[None, :, None]

        x_even = tl.load(x_even_ptrs, mask=x_mask, other=0.0)
        x_odd = tl.load(x_odd_ptrs, mask=x_mask, other=0.0)
        x_dtype = x_even.dtype

        scale_ptrs = w_block_row_ptrs + cur_kb[None, :] * 18
        scale_mask = (offs_n[:, None] < n) & kb_mask[None, :]
        scale_lo = tl.load(scale_ptrs + 0, mask=scale_mask, other=0)
        scale_hi = tl.load(scale_ptrs + 1, mask=scale_mask, other=0)
        scale_bits = scale_lo.to(tl.uint16) | (scale_hi.to(tl.uint16) << 8)
        scales = tl.cast(scale_bits, tl.float16, bitcast=True).to(x_dtype)

        packed_ptrs = (
            w_packed_row_ptrs
            + cur_kb[None, :, None] * 18
            + 2
            + offs_byte[None, None, :]
        )
        packed_mask = (offs_n[:, None, None] < n) & kb_mask[None, :, None]
        packed = tl.load(packed_ptrs, mask=packed_mask, other=0)

        low = ((packed & 0x0F).to(x_dtype) - 8.0) * scales[:, :, None]
        high = (((packed >> 4) & 0x0F).to(x_dtype) - 8.0) * scales[:, :, None]

        x_tile = tl.join(x_even, x_odd)
        w_tile = tl.join(low, high)

        x_tile = tl.reshape(x_tile, (BLOCK_M, BLOCK_K_BLOCKS * 32))
        w_tile = tl.reshape(w_tile, (BLOCK_N, BLOCK_K_BLOCKS * 32))

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_q4_0_triton(
    W: torch.Tensor,
    X: torch.Tensor,
    row: int,
) -> torch.Tensor:
    if not W.is_cuda or not X.is_cuda:
        raise ValueError("Q4_0 Triton kernel requires CUDA tensors")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Q4_0 weights must be torch.uint8, got {W.dtype}")
    if X.dtype not in Q4_0_SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Triton Q4_0 kernel supports torch.float16, torch.bfloat16, and "
            f"torch.float32 activations, got {X.dtype}"
        )
    if X.dim() not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.dim()}D")
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )
    if W.shape[1] % Q4_0_BLOCK_BYTES != 0:
        raise ValueError(
            f"Invalid Q4_0 row width {W.shape[1]}: "
            f"must be divisible by {Q4_0_BLOCK_BYTES}"
        )

    num_k_blocks = W.shape[1] // Q4_0_BLOCK_BYTES
    hidden_size = num_k_blocks * Q4_0_BLOCK_SIZE
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match "
            f"Q4_0 weight width {hidden_size}"
        )

    W = W.contiguous()
    X_shape = X.shape
    X_2d = X.reshape(-1, hidden_size).contiguous()
    Y_2d = torch.empty((X_2d.shape[0], row), device=X.device, dtype=X.dtype)

    grid = (
        triton.cdiv(X_2d.shape[0], Q4_0_BLOCK_M),
        triton.cdiv(row, Q4_0_BLOCK_N),
    )
    q4_0_gemm_kernel[grid](
        X_2d,
        W,
        Y_2d,
        X_2d.shape[0],
        row,
        num_k_blocks,
        X_2d.stride(0),
        X_2d.stride(1),
        W.stride(0),
        Y_2d.stride(0),
        Y_2d.stride(1),
        BLOCK_M=Q4_0_BLOCK_M,
        BLOCK_N=Q4_0_BLOCK_N,
        BLOCK_K_BLOCKS=Q4_0_BLOCK_K_BLOCKS,
        num_warps=Q4_0_NUM_WARPS,
        num_stages=Q4_0_NUM_STAGES,
    )

    if X.dim() == 2:
        return Y_2d
    return Y_2d.view(*X_shape[:-1], row)
