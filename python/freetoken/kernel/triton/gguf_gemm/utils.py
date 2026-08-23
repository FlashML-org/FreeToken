import torch
import triton
import triton.language as tl

GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_IQ1_M = 29

QK = 32
QK_K = 256
Q4_0_BLOCK_BYTES = 18
Q4_1_BLOCK_BYTES = 20
Q5_0_BLOCK_BYTES = 22
Q5_1_BLOCK_BYTES = 24
Q8_0_BLOCK_BYTES = 34
Q8_1_BLOCK_BYTES = 36
Q2_K_BLOCK_BYTES = 84
Q3_K_BLOCK_BYTES = 110
Q4_K_BLOCK_BYTES = 144
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210
IQ2_XXS_BLOCK_BYTES = 66
IQ2_XS_BLOCK_BYTES = 74
IQ3_XXS_BLOCK_BYTES = 98
IQ1_S_BLOCK_BYTES = 50
IQ4_NL_BLOCK_BYTES = 18
IQ3_S_BLOCK_BYTES = 110
IQ2_S_BLOCK_BYTES = 82
IQ4_XS_BLOCK_BYTES = 136
IQ1_M_BLOCK_BYTES = 56

BLOCK_BYTES_BY_TYPE = {
    GGML_TYPE_Q4_0: Q4_0_BLOCK_BYTES,
    GGML_TYPE_Q4_1: Q4_1_BLOCK_BYTES,
    GGML_TYPE_Q5_0: Q5_0_BLOCK_BYTES,
    GGML_TYPE_Q5_1: Q5_1_BLOCK_BYTES,
    GGML_TYPE_Q8_0: Q8_0_BLOCK_BYTES,
    GGML_TYPE_Q8_1: Q8_1_BLOCK_BYTES,
    GGML_TYPE_Q2_K: Q2_K_BLOCK_BYTES,
    GGML_TYPE_Q3_K: Q3_K_BLOCK_BYTES,
    GGML_TYPE_Q4_K: Q4_K_BLOCK_BYTES,
    GGML_TYPE_Q5_K: Q5_K_BLOCK_BYTES,
    GGML_TYPE_Q6_K: Q6_K_BLOCK_BYTES,
    GGML_TYPE_IQ2_XXS: IQ2_XXS_BLOCK_BYTES,
    GGML_TYPE_IQ2_XS: IQ2_XS_BLOCK_BYTES,
    GGML_TYPE_IQ3_XXS: IQ3_XXS_BLOCK_BYTES,
    GGML_TYPE_IQ1_S: IQ1_S_BLOCK_BYTES,
    GGML_TYPE_IQ4_NL: IQ4_NL_BLOCK_BYTES,
    GGML_TYPE_IQ3_S: IQ3_S_BLOCK_BYTES,
    GGML_TYPE_IQ2_S: IQ2_S_BLOCK_BYTES,
    GGML_TYPE_IQ4_XS: IQ4_XS_BLOCK_BYTES,
    GGML_TYPE_IQ1_M: IQ1_M_BLOCK_BYTES,
}

BLOCK_QK_BY_TYPE = {
    GGML_TYPE_Q4_0: QK,
    GGML_TYPE_Q4_1: QK,
    GGML_TYPE_Q5_0: QK,
    GGML_TYPE_Q5_1: QK,
    GGML_TYPE_Q8_0: QK,
    GGML_TYPE_Q8_1: QK,
    GGML_TYPE_Q2_K: QK_K,
    GGML_TYPE_Q3_K: QK_K,
    GGML_TYPE_Q4_K: QK_K,
    GGML_TYPE_Q5_K: QK_K,
    GGML_TYPE_Q6_K: QK_K,
    GGML_TYPE_IQ2_XXS: QK_K,
    GGML_TYPE_IQ2_XS: QK_K,
    GGML_TYPE_IQ3_XXS: QK_K,
    GGML_TYPE_IQ1_S: QK_K,
    GGML_TYPE_IQ4_NL: QK,
    GGML_TYPE_IQ3_S: QK_K,
    GGML_TYPE_IQ2_S: QK_K,
    GGML_TYPE_IQ4_XS: QK_K,
    GGML_TYPE_IQ1_M: QK_K,
}

TRITON_SUPPORTED_TYPES = frozenset(BLOCK_BYTES_BY_TYPE)

TRITON_BLOCK_M = 32
TRITON_BLOCK_N = 128
TRITON_BLOCK_K_BLOCKS = 4
TRITON_NUM_WARPS = 2
TRITON_NUM_STAGES = 2
TRITON_SUPPORTED_ACTIVATION_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


@triton.jit
def load_f16_from_u8(ptrs, mask):
    lo = tl.load(ptrs + 0, mask=mask, other=0)
    hi = tl.load(ptrs + 1, mask=mask, other=0)
    bits = lo.to(tl.uint16) | (hi.to(tl.uint16) << 8)
    return tl.cast(bits, tl.float16, bitcast=True)


@triton.jit
def load_u32_from_u8(ptrs, mask):
    b0 = tl.load(ptrs + 0, mask=mask, other=0).to(tl.uint32)
    b1 = tl.load(ptrs + 1, mask=mask, other=0).to(tl.uint32)
    b2 = tl.load(ptrs + 2, mask=mask, other=0).to(tl.uint32)
    b3 = tl.load(ptrs + 3, mask=mask, other=0).to(tl.uint32)
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


@triton.jit
def load_u16_from_u8(ptrs, mask):
    lo = tl.load(ptrs + 0, mask=mask, other=0).to(tl.uint16)
    hi = tl.load(ptrs + 1, mask=mask, other=0).to(tl.uint16)
    return lo | (hi << 8)


@triton.jit
def load_x_tile(
    x_ptr,
    m,
    num_k_blocks,
    stride_xm,
    stride_xk,
    offs_m,
    kb_start,
    offs_kb,
    offs_nibble,
    BLOCK_M: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    cur_kb = kb_start + offs_kb
    kb_mask = cur_kb < num_k_blocks
    x_row_ptrs = x_ptr + offs_m[:, None, None] * stride_xm
    x_k_low = cur_kb[None, :, None] * 32 + offs_nibble[None, None, :]
    x_k_high = x_k_low + 16
    x_even = tl.load(
        x_row_ptrs + x_k_low * stride_xk,
        mask=(offs_m[:, None, None] < m) & kb_mask[None, :, None],
        other=0.0,
    )
    x_odd = tl.load(
        x_row_ptrs + x_k_high * stride_xk,
        mask=(offs_m[:, None, None] < m) & kb_mask[None, :, None],
        other=0.0,
    )
    return (
        tl.reshape(tl.join(x_even, x_odd), (BLOCK_M, BLOCK_K_BLOCKS * 32)),
        cur_kb,
        kb_mask,
    )


@triton.jit
def load_x_chunk(
    x_ptr,
    stride_xm,
    stride_xk,
    offs_m,
    m,
    k_start,
    CHUNK: tl.constexpr,
):
    offs_k = k_start + tl.arange(0, CHUNK)
    return tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=offs_m[:, None] < m,
        other=0.0,
    )


@triton.jit
def load_scale_min_k4(scales_ptr, mask, j: tl.constexpr):
    if j < 4:
        d = tl.load(scales_ptr + j, mask=mask, other=0)
        m = tl.load(scales_ptr + j + 4, mask=mask, other=0)
        return d & 0x3F, m & 0x3F

    hi = tl.load(scales_ptr + j + 4, mask=mask, other=0)
    lo_d = tl.load(scales_ptr + j - 4, mask=mask, other=0)
    lo_m = tl.load(scales_ptr + j + 0, mask=mask, other=0)
    d = (hi & 0x0F) | ((lo_d >> 6) << 4)
    m = (hi >> 4) | ((lo_m >> 6) << 4)
    return d, m


@triton.jit
def load_q3_k_scale(scales_ptr, mask, is_idx: tl.constexpr):
    if is_idx < 4:
        lo = tl.load(scales_ptr + is_idx, mask=mask, other=0)
        hi = tl.load(scales_ptr + is_idx + 8, mask=mask, other=0)
        return (lo & 0x0F) | (((hi >> 0) & 0x03) << 4)
    if is_idx < 8:
        lo = tl.load(scales_ptr + is_idx, mask=mask, other=0)
        hi = tl.load(scales_ptr + is_idx + 4, mask=mask, other=0)
        return (lo & 0x0F) | (((hi >> 2) & 0x03) << 4)
    if is_idx < 12:
        lo = tl.load(scales_ptr + is_idx - 8, mask=mask, other=0)
        hi = tl.load(scales_ptr + is_idx, mask=mask, other=0)
        return ((lo >> 4) & 0x0F) | (((hi >> 4) & 0x03) << 4)

    lo = tl.load(scales_ptr + is_idx - 8, mask=mask, other=0)
    hi = tl.load(scales_ptr + is_idx - 4, mask=mask, other=0)
    return ((lo >> 4) & 0x0F) | (((hi >> 6) & 0x03) << 4)


def _validate_args(
    W: torch.Tensor,
    X: torch.Tensor,
    row: int,
    quant_type: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    if quant_type not in TRITON_SUPPORTED_TYPES:
        raise ValueError(f"Unsupported Triton quant type: {quant_type}")
    if not W.is_cuda or not X.is_cuda:
        raise ValueError("Triton kernels require CUDA tensors")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if X.dtype not in TRITON_SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Triton kernels support torch.float16, torch.bfloat16, and torch.float32 "
            f"activations, got {X.dtype}"
        )
    if X.dim() not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.dim()}D")
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )

    block_bytes = BLOCK_BYTES_BY_TYPE[quant_type]
    if W.shape[1] % block_bytes != 0:
        raise ValueError(
            f"Invalid row width {W.shape[1]} for quant type {quant_type}: "
            f"must be divisible by {block_bytes}"
        )

    num_k_blocks = W.shape[1] // block_bytes
    hidden_size = num_k_blocks * BLOCK_QK_BY_TYPE[quant_type]
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match "
            f"quantized weight width {hidden_size}"
        )

    return (
        W.contiguous(),
        X.reshape(-1, hidden_size).contiguous(),
        X.shape,
        num_k_blocks,
    )


def run_triton_kernel(
    kernel,
    W: torch.Tensor,
    X: torch.Tensor,
    row: int,
    quant_type: int,
    extra_args: tuple = (),
) -> torch.Tensor:
    W, X_2d, X_shape, num_k_blocks = _validate_args(W, X, row, quant_type)
    Y_2d = torch.empty((X_2d.shape[0], row), device=X.device, dtype=X.dtype)

    grid = (
        triton.cdiv(X_2d.shape[0], TRITON_BLOCK_M),
        triton.cdiv(row, TRITON_BLOCK_N),
    )

    kernel[grid](
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
        *extra_args,
        BLOCK_M=TRITON_BLOCK_M,
        BLOCK_N=TRITON_BLOCK_N,
        BLOCK_K_BLOCKS=TRITON_BLOCK_K_BLOCKS,
        num_warps=TRITON_NUM_WARPS,
        num_stages=TRITON_NUM_STAGES,
    )

    if len(X_shape) == 2:
        return Y_2d
    return Y_2d.view(*X_shape[:-1], row)
