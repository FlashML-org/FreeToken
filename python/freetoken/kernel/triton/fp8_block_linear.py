"""DeepSeek-V3-style 128x128 block-FP8 dense linear, shared across models.

The checkpoint stores each quantized weight as fp8-e4m3 ``weight`` plus a per-128x128
block ``weight_scale_inv`` (bf16, shape ``[ceil(N/128), ceil(K/128)]``) such that
``weight_bf16[i,j] = weight_fp8[i,j] * weight_scale_inv[i//128, j//128]``. Activations
are quantized dynamically per token, per 128-element K group (``activation_scheme:
dynamic``). The matmul accumulates ``A_fp8 @ W_fp8`` in fp32 and applies the act + weight
block scales per 128-K block -- the exact numerics of sglang's ``w8a8_block_fp8_matmul``
/ vLLM's ``w8a8_triton_block_scaled_mm`` (this is a self-contained FreeToken port; the
GEMM/quant skeleton mirrors ``kernel/triton/dsv4/fp8_linear.py`` but uses plain fp32 block
scales rather than that path's ue8m0 power-of-two codes).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from freetoken.layers import BaseOP

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_act_dtype,
    e4m3_kernel_view,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    round_e4m3,
)

FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0  # e4m3 finite max
_BLOCK = 128
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


# ======================================================================================
# Dynamic per-token, per-128-group activation FP8 quantization (fp32 scale).
# ======================================================================================
@triton.jit
def _act_quant_kernel(
    x_ptr, y_ptr, s_ptr, M, K,
    stride_xm, stride_xk, stride_ym, stride_yk, stride_sm, stride_sk,
    BLOCK_M: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    s = amax / 448.0  # [BLOCK_M] fp32 per-token-group scale (e4m3 finite max = 448)
    y = tl.clamp(x / s[:, None], -448.0, 448.0)
    if e4m3_native_cx():
        y = y.to(tl.float8e4nv)
    else:
        y = round_e4m3(y)  # e4m3-grid values into the wrapper's bf16 buffer
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk, y, mask=m_mask[:, None])
    tl.store(s_ptr + offs_m * stride_sm + pid_k * stride_sk, s, mask=m_mask)


def per_token_group_quant_fp8(x: torch.Tensor, block: int = _BLOCK):
    """Quantize ``x`` ``[..., K]`` to fp8 with one fp32 scale per token per 128-K group.
    Returns ``(x_fp8 [M, K], x_scale [M, K//block] fp32)``."""
    *lead, K = x.shape
    assert K % block == 0, (K, block)
    x2d = x.reshape(-1, K)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    M = x2d.shape[0]
    y = torch.empty((M, K), dtype=e4m3_act_dtype(), device=x.device)
    s = torch.empty((M, K // block), dtype=torch.float32, device=x.device)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), K // block)
    _act_quant_kernel[grid](
        x2d, y, s, M, K,
        x2d.stride(0), x2d.stride(1), y.stride(0), y.stride(1), s.stride(0), s.stride(1),
        BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return y, s


# ======================================================================================
# Block-scaled FP8 x FP8 GEMM (fp32 act scale + fp32 weight block scale).
# ======================================================================================
@triton.jit
def _block_fp8_gemm_kernel(
    a_ptr, w_ptr, sa_ptr, sb_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk,
    stride_sam, stride_sak, stride_sbn, stride_sbk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    """``BLOCK_N == BLOCK_K == 128`` so each tile maps to exactly one weight scale block
    (``sb`` is one value per (pid_n, k)). ``sa`` is one value per (row, k)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for k in range(num_k):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        if e4m3_native_cx():
            p = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        else:
            # bf16 dot on the same e4m3 grid: operands exact in bf16, fp32 acc
            p = tl.dot(a, tl.trans(e4m3_u8_to_f32(w).to(tl.bfloat16)), out_dtype=tl.float32)
        sa = tl.load(sa_ptr + offs_m * stride_sam + k * stride_sak, mask=m_mask, other=0.0).to(tl.float32)
        sb = tl.load(sb_ptr + pid_n * stride_sbn + k * stride_sbk).to(tl.float32)
        acc += p * sa[:, None] * sb
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None])


def block_fp8_matmul(
    a_fp8: torch.Tensor, a_scale: torch.Tensor,
    weight: torch.Tensor, weight_scale: torch.Tensor, out_dtype: torch.dtype,
) -> torch.Tensor:
    """``y = (a_fp8 * a_scale) @ (weight * weight_scale)^T`` over 128-K blocks.

    ``a_fp8`` ``[M, K]`` fp8, ``a_scale`` ``[M, K//128]`` fp32; ``weight`` ``[N, K]`` fp8,
    ``weight_scale`` ``[N//128, K//128]`` (bf16 or fp32, cast in-kernel)."""
    M, K = a_fp8.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert N % _BLOCK == 0 and K % _BLOCK == 0, (N, K)
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    out = torch.empty((M, N), dtype=compute, device=a_fp8.device)
    w = e4m3_kernel_view(weight)
    # Larger M-tile for compute-bound prefill (better fp8 tensor-core utilization).
    block_m = 64 if M >= 64 else 32
    nwarps = 8 if M >= 64 else 4
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, _BLOCK))
    _block_fp8_gemm_kernel[grid](
        a_fp8, w, a_scale, weight_scale, out,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1), w.stride(0), w.stride(1),
        a_scale.stride(0), a_scale.stride(1), weight_scale.stride(0), weight_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=_BLOCK, BLOCK_K=_BLOCK, compute_type=_TL_DTYPE[compute],
        num_warps=nwarps, num_stages=3,
    )
    return out


# ======================================================================================
# Decode (M==1) split-K GEMV. The block GEMM tiles M by 32, wasting 31/32 rows at M=1;
# this GEMV is the bs=1 decode path. Skeleton from kernel/triton/dsv4/fp8_linear.py's
# _fp8_act_gemv_splitk, with the ue8m0 scale decode replaced by plain bf16/fp32 block scales.
# ======================================================================================
@triton.jit
def _block_fp8_gemv_splitk_kernel(
    a_ptr, sa_ptr, w_ptr, sb_ptr, part_ptr, N, K, n_kb, kb_per,
    stride_ak, stride_wn, stride_wk, stride_sbn, stride_sbk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, USE_A_SCALE: tl.constexpr,
):
    """Each pid_k split covers ``kb_per`` 128-K-blocks (ceil-tiled, masked to ``n_kb``), so
    K only has to be a multiple of 128 -- no power-of-two ``K//128`` / split divisibility
    assumption (the loads are k-masked, the partials reduced over the full grid)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    sn = offs_n // 128
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * 128 + tl.arange(0, 128)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            scb = tl.load(sb_ptr + sn * stride_sbn + kb * stride_sbk, mask=n_mask, other=0.0).to(tl.float32)
            if USE_A_SCALE:
                scb *= tl.load(sa_ptr + kb).to(tl.float32)  # W8A8: also the per-token-group act scale
            acc += tl.sum(w * a[None, :], axis=1) * scb
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(part_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
                          stride_pk, stride_pn, BLOCK: tl.constexpr, OUT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(OUT), mask=mask)


def _block_fp8_gemv(a, weight, weight_scale, out_dtype, a_scale=None):
    """M==1 split-K GEMV. ``a`` [K]; ``weight`` [N,K] fp8, scale [N//128,K//128]. W8A16 when
    ``a_scale is None`` (a is bf16, weight-only dequant -- skips the act-quant kernel), else
    W8A8 with the per-128-group act scale ``a_scale`` [K//128]."""
    N, K = weight.shape
    assert K % 128 == 0, K
    n_kb = K // 128
    BLOCK_N = 16
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(1536 // n_tiles, n_kb))
    # Power-of-two split keeps the reduction bit-identical for the common pow2 K//128 (so the
    # decode trajectory is stable); the kb_per ceil-tile + k_mask below still handle any K.
    split_k = 1 << (split_k.bit_length() - 1)
    kb_per = triton.cdiv(n_kb, split_k)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a.device)
    w = e4m3_kernel_view(weight)
    _block_fp8_gemv_splitk_kernel[(n_tiles, split_k)](
        a, a_scale if a_scale is not None else a, w, weight_scale, part, N, K, n_kb, kb_per,
        a.stride(0), w.stride(0), w.stride(1),
        weight_scale.stride(0), weight_scale.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, USE_A_SCALE=a_scale is not None, num_warps=1,
    )
    out = torch.empty(N, dtype=out_dtype, device=a.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16], num_warps=2,
    )
    return out


def block_fp8_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ weight^T`` with block-fp8 weight. Decode (M=1) uses a W8A16 split-K GEMV
    (bf16 activation, weight-only dequant -- no act-quant kernel); prefill (M>1) uses the
    W8A8 block GEMM."""
    *lead, K = x.shape
    N = weight.shape[0]
    if x.numel() // K == 1:  # decode bs=1: W8A16 GEMV (bf16 act, no per-token-group quant)
        out = _block_fp8_gemv(x.reshape(K), weight, weight_scale, x.dtype).reshape(*lead, N)
    else:
        a_fp8, a_scale = per_token_group_quant_fp8(x, _BLOCK)
        out = block_fp8_matmul(a_fp8, a_scale, weight, weight_scale, x.dtype).reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor, block: int = _BLOCK) -> torch.Tensor:
    """Dequantize a block-fp8 weight ``[N, K]`` (+ scale ``[N//block, K//block]``) to bf16.

    ``out[i,j] = weight_fp8[i,j] * scale[i//block, j//block]`` (used by the M1 expert
    bank loader, which runs the routed experts in bf16)."""
    N, K = weight.shape
    sn, sk = scale.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


# ======================================================================================
# BaseOP linear layers (TP=1, replicated).
# ======================================================================================
class Fp8BlockLinear(BaseOP):
    """Replicated block-fp8 linear: fp8-e4m3 ``weight`` ``[out, in]`` + bf16
    ``weight_scale_inv`` ``[out//128, in//128]``. Activations are quantized dynamically;
    the GEMM dequantizes per block, so activations keep bf16 precision."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        assert in_features % _BLOCK == 0 and out_features % _BLOCK == 0, (in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features, dtype=FP8)
        self.weight_scale_inv = torch.empty(
            out_features // _BLOCK, in_features // _BLOCK, dtype=torch.bfloat16
        )
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return block_fp8_linear(x, self.weight, self.weight_scale_inv, self.bias)


class Fp8BlockColMerged(Fp8BlockLinear):
    """Block-fp8 column-merged linear (drop-in for ``LinearColParallelMerged``).

    Holds one fp8 weight that is the concatenation of several projections along the output
    dim; the caller splits the bf16 output by the (TP-local) part sizes. Each part's
    out dim must be a multiple of 128 so the concatenated ``weight_scale_inv`` blocks align.

    With ``local_output_sizes`` (TP>1) each part is sharded along the output dim
    independently (GQA KV-head replication included) and ``weight_scale_inv`` is sharded
    identically along its row axis."""

    def __init__(self, in_features: int, output_sizes: list[int], has_bias: bool = False,
                 local_output_sizes: list[int] | None = None):
        for o in output_sizes:
            assert o % _BLOCK == 0, (output_sizes, "each merged output size must be /128 for fp8")
        self.output_sizes = list(output_sizes)
        if local_output_sizes is not None:
            assert len(local_output_sizes) == len(output_sizes), (
                local_output_sizes, output_sizes
            )
            for o in local_output_sizes:
                assert o % _BLOCK == 0, (local_output_sizes, "local sizes must be /128 for fp8")
            out_features = sum(local_output_sizes)
        else:
            out_features = sum(output_sizes)
        super().__init__(in_features, out_features, has_bias)


class Fp8BlockRowParallel(Fp8BlockLinear):
    """Row-parallel block-fp8 linear (drop-in for ``LinearRowParallel``): fp8 ``weight``
    ``[out, in/tp]`` + ``weight_scale_inv`` ``[out//128, in/tp//128]``. Each rank reduces
    over its input shard only (activations are quantized per 128-K group *within* the
    shard, so the dequant stays exact per rank); the all-reduce completes the sum and the
    bias (if any) is added once, after it."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        from freetoken.distributed import DistributedCommunicator, get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        self._tp_size = tp_info.size
        self._comm = DistributedCommunicator() if tp_info.size > 1 else None
        super().__init__(div_even(in_features, tp_info.size), out_features, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = block_fp8_linear(x, self.weight, self.weight_scale_inv, None)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


__all__ = [
    "FP8",
    "Fp8BlockLinear",
    "Fp8BlockColMerged",
    "Fp8BlockRowParallel",
    "block_fp8_linear",
    "block_fp8_matmul",
    "per_token_group_quant_fp8",
    "dequant_block_fp8",
]
