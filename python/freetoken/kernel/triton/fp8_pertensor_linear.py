"""Per-tensor (per-output-row) FP8 W8A16 dense linear, shared across models.

The nvidia/modelopt ``MIXED_PRECISION`` checkpoints (Qwen3.5/3.6, ...) quantize the attention
and GatedDeltaNet projections to **per-tensor FP8** (``weight`` fp8-e4m3 + a single scalar
``weight_scale``; ``weight_bf16 = weight_fp8 * weight_scale``) while the routed experts are
NVFP4. FreeToken used to *dequantize these dense weights to bf16 at load* and run them through
cuBLAS bf16 GEMV -- which is the dominant decode cost (~54% of the step) because it reads
2x the bytes of the resident FP8 weight. This module keeps the weight FP8 and reads it
directly in a W8A16 kernel (bf16 activation, fp8 weight), halving that traffic.

Fused projections (q/k/v -> ``qkv_proj``; GDN qkv|z -> ``in_proj_qkvz``) concatenate parts
that each carry their *own* per-tensor scalar, so the scalar is broadcast to a per-output-row
vector ``weight_scale`` ``[N]`` (piecewise-constant after fusion) and applied per output row.
This is exactly equal to applying each part's scalar to its own rows.

Numerics: the kernel converts fp8->f32 and accumulates in fp32 (standard W8A16; strictly more
accurate than the previous ``weight.to(bf16) * scale`` materialization, which it replaces).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from freetoken.layers import BaseOP
from freetoken.layers.base import _concat_prefix

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_kernel_view,
    e4m3_native,
    e4m3_native_cx,
    e4m3_u8_to_f32,
)

FP8 = torch.float8_e4m3fn
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Escape hatch: FREETOKEN_DEBUG_FP8_REF=1 swaps the triton kernels for a pure-torch dequant matmul
# (numeric reference / A-B debugging). Evaluated once; the kernels are the default.
_USE_REF = os.environ.get("FREETOKEN_DEBUG_FP8_REF") == "1"


# ======================================================================================
# Decode (M==1) split-K GEMV: raw fp8 x bf16 reduction in fp32, per-row scale at reduce.
# ======================================================================================
@triton.jit
def _gemv_splitk_kernel(
    a_ptr, w_ptr, part_ptr, N, K, n_kb, kb_per,
    stride_ak, stride_wn, stride_wk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Each (pid_n, pid_k) computes the partial sum over ``kb_per`` BLOCK_K chunks for a
    BLOCK_N slice of outputs. ``kb_per`` ceil-tiles K so K only needs to be a multiple of
    BLOCK_K is *not* required (loads are k-masked). Per-row scale is applied in the reduce."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
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
            acc += tl.sum(w * a[None, :], axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(
    part_ptr, scale_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
    stride_pk, stride_pn, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (acc * scale).to(OUT), mask=mask)


def _gemv(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M==1 split-K GEMV. ``a`` [K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32."""
    N, K = weight.shape
    BLOCK_K = 128
    n_kb = triton.cdiv(K, BLOCK_K)
    BLOCK_N = 16
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(1536 // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # pow2 -> stable reduction order
    kb_per = triton.cdiv(n_kb, split_k)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a.device)
    _gemv_splitk_kernel[(n_tiles, split_k)](
        a, weight, part, N, K, n_kb, kb_per,
        a.stride(0), weight.stride(0), weight.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=1,
    )
    out = torch.empty(N, dtype=out_dtype, device=a.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, weight_scale, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16],
        num_warps=2,
    )
    return out


# ======================================================================================
# Prefill (M>1) W8A16 GEMM: fp8 weight read from HBM, upcast to bf16 in-register for the
# tensor-core dot (fp8 e4m3 -> bf16 is lossless), per-row scale applied after accumulation.
# ======================================================================================
@triton.jit
def _gemm_kernel(
    a_ptr, w_ptr, scale_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        w_mask = n_mask[:, None] & (offs_k[None, :] < k_rem)
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(a.dtype)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=w_mask, other=0)).to(a.dtype)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None] & n_mask[None, :])


def _gemm(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M>1 W8A16 GEMM. ``a`` [M, K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32."""
    M, K = a.shape
    N = weight.shape[0]
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    out = torch.empty((M, N), dtype=compute, device=a.device)
    BLOCK_M = 64 if M >= 64 else 32
    BLOCK_N, BLOCK_K = 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        a, weight, weight_scale, out, M, N, K,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, compute_type=_TL_DTYPE[compute],
        num_warps=8 if M >= 64 else 4, num_stages=3,
    )
    return out


# ======================================================================================
# Batched decode (M > 1) W8A8: quantize the activation with the checkpoint's static
# per-tensor ``input_scale`` and hand both fp8 operands to cuBLASLt via torch._scaled_mm.
#
# ``_gemm`` above is a *prefill* kernel: grid is (cdiv(M, BLOCK_M), cdiv(N, 128)) with no
# split-K, so at decode batch sizes the M axis contributes a single block row and the whole
# GEMM runs on cdiv(N, 128) CTAs -- 8 of them for N=1024, on a 188-SM part. Its runtime then
# tracks K and ignores both N and M: latency-bound, not bandwidth-bound. Measured on
# Qwen3.5-27B's projections (RTX PRO 6000, 1461 GB/s copy roof): 594 GB/s flat across
# M=2..16, against 1334 GB/s for the M=1 GEMV over the same weights. Routing M>=2 to
# cuBLASLt restores ~1300 GB/s -- 10.15 ms -> 5.56 ms of per-decode-step GEMM.
#
# Requires sm_89+ (torch._scaled_mm's floor; Ampere has no FP8 tensor cores) *and* a
# checkpoint carrying ``input_scale``. Without either, ``_gemm`` still runs: glm_moe_dsa
# quantizes to fp8 at load with genuine per-row scales and has no activation scale at all.
# ======================================================================================
_E4M3_MAX = 448.0  # largest finite e4m3 magnitude


@triton.jit
def _static_quant_kernel(x_ptr, out_ptr, scale_ptr, n_elements, BLOCK: tl.constexpr):
    """bf16 activation -> fp8-e4m3 under one broadcast per-tensor scale."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def _static_quant(a: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """One launch, no host sync: the scale is read from device memory inside the kernel, so
    this stays CUDA-graph safe (fixed shapes, no dependence on tensor *values* on the host)."""
    n = a.numel()
    out = torch.empty_like(a, dtype=FP8)
    BLOCK = 1024
    _static_quant_kernel[(triton.cdiv(n, BLOCK),)](
        a, out, input_scale, n, BLOCK=BLOCK, num_warps=4,
    )
    return out


def _scaled_mm(
    a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    input_scale: torch.Tensor, uniform_scale: bool, out_dtype: torch.dtype,
) -> torch.Tensor:
    """``a @ (weight_fp8 * weight_scale)^T`` as a W8A8 cuBLASLt GEMM.

    ``weight`` stays in the checkpoint-native row-major ``[N, K]``: ``.t()`` is exactly the
    column-major ``[K, N]`` operand cuBLASLt wants, so the transpose is a stride change and
    never a copy (unlike Marlin, or our own NVFP4 path's K-major repack).

    A genuine per-tensor weight (one scalar repeated down ``weight_scale``) takes the
    tensor-wise path. A fused projection, whose ``weight_scale`` is piecewise-constant
    because each part carries its own scalar, takes the row-wise path -- that keeps every
    part's scale exact, where vLLM/SGLang instead requantize the parts onto a shared maximum
    and eat the precision loss. Row-wise costs ~4% here (5.56 ms vs 5.39 ms per step)."""
    qa = _static_quant(a, input_scale)
    wt = weight.t()  # [N, K] row-major -> [K, N] column-major, stride-only
    if uniform_scale:
        return torch._scaled_mm(
            qa, wt, scale_a=input_scale.reshape(()), scale_b=weight_scale[0].reshape(()),
            out_dtype=out_dtype,
        )
    return torch._scaled_mm(
        qa, wt,
        scale_a=input_scale.reshape(1, 1).expand(a.shape[0], 1).contiguous(),
        scale_b=weight_scale.reshape(1, -1),
        out_dtype=out_dtype,
    )


def fp8_pertensor_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    uniform_scale: bool = False,
) -> torch.Tensor:
    """``y = x @ (weight_fp8 * weight_scale)^T``. ``weight`` [N, K] fp8-e4m3, ``weight_scale``
    [N] fp32 (per output row).

    Whether the activation is quantized is a property of the *deployment*, never of the batch:
    with ``input_scale`` on sm_89+ every M runs W8A8, otherwise every M runs W8A16 (split-K
    GEMV at M=1, GEMM above it). Picking per-M would make a request's numerics depend on how
    many unrelated requests happened to be in flight, so a reply would not reproduce at bs=1 --
    and W8A16 at M=1 is worth only ~0.5% (5.53 ms vs 5.56 ms of per-step GEMM) anyway. vLLM and
    SGLang likewise run one scheme across all M on any GPU with FP8 tensor cores."""
    *lead, K = x.shape
    N = weight.shape[0]
    if _USE_REF:  # numeric-reference fallback (debug / A-B)
        w = weight.to(x.dtype) * weight_scale.to(x.dtype)[:, None]
        out = (x.reshape(-1, K) @ w.t()).reshape(*lead, N)
    elif input_scale is not None and e4m3_native():
        out = _scaled_mm(
            x.reshape(-1, K), weight, weight_scale, input_scale, uniform_scale, x.dtype,
        ).reshape(*lead, N)
    elif x.numel() // K == 1:
        out = _gemv(x.reshape(K), e4m3_kernel_view(weight), weight_scale, x.dtype).reshape(*lead, N)
    else:
        out = _gemm(
            x.reshape(-1, K), e4m3_kernel_view(weight), weight_scale, x.dtype,
        ).reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


# ======================================================================================
# BaseOP linear layers (TP=1, replicated). Buffers: fp8 ``weight`` + fp32 ``weight_scale``.
# ======================================================================================
class Fp8PerTensorLinear(BaseOP):
    """Replicated per-tensor-FP8 linear: fp8-e4m3 ``weight`` ``[out, in]`` + per-row fp32
    ``weight_scale`` ``[out]`` (a genuine per-tensor weight stores the same scalar in every
    row; a fused projection stores each part's scalar across its own rows).

    ``input_scale`` (modelopt's calibrated per-tensor activation scale) is optional: when the
    checkpoint ships it, batched decode runs W8A8 through cuBLASLt; when it does not (models
    that quantize to fp8 at load, e.g. glm_moe_dsa), every batch size stays W8A16."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features, dtype=FP8)
        self.weight_scale = torch.empty(out_features, dtype=torch.float32)
        self.bias = torch.empty(out_features) if has_bias else None
        # Set from the state dict when present. Left as None (not a tensor) so BaseOP's
        # reflective state_dict/load_state_dict skip it entirely on checkpoints without one.
        self.input_scale: torch.Tensor | None = None
        self._uniform_scale = False

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        # Taken out before BaseOP's reflective pass (so it is not an "unexpected key") and
        # put back after (so that pass does not try to pop it a second time on a reload).
        input_scale = state_dict.pop(_concat_prefix(prefix, "input_scale"), None)
        self.input_scale = None
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        self.input_scale = input_scale
        # Tensor-wise scaling needs one scalar for the whole weight; a fused projection is
        # only piecewise-constant, so decide once here rather than syncing on every forward.
        scale = self.weight_scale
        self._uniform_scale = bool((scale == scale[0]).all().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_pertensor_linear(
            x, self.weight, self.weight_scale, self.bias,
            self.input_scale, self._uniform_scale,
        )


class Fp8PerTensorColMerged(Fp8PerTensorLinear):
    """Column-merged per-tensor-FP8 linear (drop-in for ``LinearColParallelMerged``):
    one fp8 weight concatenating several projections along the output dim; the caller splits
    the bf16 output by the (TP-local) part sizes.

    With ``local_output_sizes`` (TP>1) each part is sharded along the output dim
    independently -- including GQA KV-head replication when ``num_kv_heads < tp_size`` --
    and the per-row ``weight_scale`` is sharded identically (it is piecewise-constant, one
    scalar per part, so row sharding keeps every part's scale exact)."""

    def __init__(self, in_features: int, output_sizes: list[int], has_bias: bool = False,
                 local_output_sizes: list[int] | None = None):
        self.output_sizes = list(output_sizes)
        if local_output_sizes is not None:
            assert len(local_output_sizes) == len(output_sizes), (
                local_output_sizes, output_sizes
            )
            out_features = sum(local_output_sizes)
        else:
            out_features = sum(output_sizes)
        super().__init__(in_features, out_features, has_bias)

class Fp8PerTensorRowParallel(Fp8PerTensorLinear):
    """Row-parallel per-tensor-FP8 linear (drop-in for ``LinearRowParallel``): fp8 ``weight``
    ``[out, in/tp]`` + per-row ``weight_scale`` ``[out]`` (the output dim is not sharded).
    Each rank reduces over its input shard only, so the result is a partial sum that the
    all-reduce completes. The bias (if any) is added once, after the all-reduce.

    ``input_scale`` is unchanged: every rank quantizes its own activation shard with the
    same per-tensor scale, so the W8A8 path stays exact per rank."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        from freetoken.distributed import DistributedCommunicator, get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        self._tp_size = tp_info.size
        self._comm = DistributedCommunicator() if tp_info.size > 1 else None
        super().__init__(div_even(in_features, tp_info.size), out_features, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = fp8_pertensor_linear(
            x, self.weight, self.weight_scale, None,
            self.input_scale, self._uniform_scale,
        )
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


__all__ = [
    "FP8",
    "Fp8PerTensorLinear",
    "Fp8PerTensorColMerged",
    "Fp8PerTensorRowParallel",
    "fp8_pertensor_linear",
]
