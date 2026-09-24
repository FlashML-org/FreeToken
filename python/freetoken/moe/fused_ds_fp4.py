"""Routed FP4 expert compute on the offload-cache banks (DeepSeek-V4).

Composes the DeepSeek-FP4 fused-MoE kernels (``dsv4_fused_moe``) into the full
routed-expert path: ``gate_up -> swiglu(limit) -> down``, reading expert weights
directly from the resident banks (no bf16 materialization). ``slots`` maps each
(token, route) to a bank row: a cache slot for the decode GEMV path, a
streamed full-layer position (== expert id) for the grouped prefill path.
"""

from __future__ import annotations

import torch
import triton

from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip
from freetoken.kernel.triton.dsv4.fused_moe import (
    _decode_dsfp4_moe_kernel,
    _e2m1_lut,
    _prefill_dsfp4_moe_kernel,
    fused_swiglu,
)
from freetoken.moe.fused import moe_align_block_size

_TL_DTYPE = None


def _compute_type(dtype: torch.dtype):
    import triton.language as tl

    return {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[dtype]


def _grouped_decode(
    a: torch.Tensor,            # [A_rows, K] compute dtype
    packed_cache: torch.Tensor,  # [S, N, K//2] uint8
    scale_cache: torch.Tensor,   # [S, N, K//32] e8m0
    slots: torch.Tensor,         # [T, top_k] int32 -> cache slot
    topk_weights: torch.Tensor | None,
    *,
    a_row_is_route: bool,
    mul_routed_weight: bool,
) -> torch.Tensor:
    """Grouped per-route GEMM ``c[t,r,:] = a[row] @ dequant(W[slot])^T``.

    Returns ``[T, top_k, N]``. Used for both gate_up (a_row=token) and down
    (a_row=route). FP4 weights are dequantized inline from the slot cache. A slot
    of ``-1`` is an inactive route: its output is zero and its storage is never read.
    """
    T, top_k = slots.shape
    N = packed_cache.shape[1]
    K = packed_cache.shape[2] * 2
    total_routes = T * top_k
    dtype = a.dtype
    out = torch.empty((T, top_k, N), dtype=dtype, device=a.device)
    if topk_weights is None:
        topk_weights = out.new_empty((1, 1), dtype=torch.float32)
    scale_u8 = scale_cache.view(torch.uint8)

    # Keep the existing DSFP4 launch geometry; model support does not retune it
    # for one GPU. Each K tile contains whole packed FP4 scale groups.
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_KB = 128
    _NW = 1
    assert (K // 2) % BLOCK_SIZE_KB == 0, (K, BLOCK_SIZE_KB)
    grid = (total_routes, triton.cdiv(N, BLOCK_SIZE_N))
    _decode_dsfp4_moe_kernel[grid](
        a, packed_cache, scale_u8, out, topk_weights, slots,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_cache.stride(0), packed_cache.stride(1), packed_cache.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        topk_weights.stride(0) if topk_weights.ndim == 2 else 0,
        topk_weights.stride(1) if topk_weights.ndim == 2 else 0,
        slots.stride(0), slots.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_KB=BLOCK_SIZE_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_compute_type(dtype),
        num_warps=_NW,
    )
    return out


def routed_experts_fp4(
    x: torch.Tensor,             # [T, H] compute dtype
    slots: torch.Tensor,         # [T, top_k] int32 -> cache slot
    topk_weights: torch.Tensor,  # [T, top_k] fp32 (incl. route_scale + renorm)
    gate_up_packed: torch.Tensor,  # [S, 2I, H//2] uint8
    gate_up_scale: torch.Tensor,   # [S, 2I, H//32] e8m0
    down_packed: torch.Tensor,     # [S, H, I//2] uint8
    down_scale: torch.Tensor,      # [S, H, I//32] e8m0
    swiglu_limit: float,
    act_block: int = 128,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Full routed-expert output (summed over the top-k routes), excludes shared expert. Each route's
    down output is rounded to the compute dtype (the reference's per-expert bf16 output) and the sum
    over routes accumulates in fp32; ``out_dtype=torch.float32`` returns that sum unrounded -- the
    reference keeps it in fp32 through the shared-expert merge -- else it is rounded once.

    Precision matches the reference ``Expert.forward`` over ``fp4_gemm(act_quant(x, act_block),
    W_fp4)``: the gate_up and down activations are FP8-round-tripped (block ``act_block`` -- the
    checkpoint's fp8 block, 128 on V4 and 32 on V4.1 -- ue8m0) before each GEMM, and the routing
    weight multiplies the fp32 SwiGLU intermediate BEFORE its bf16 cast and fp8 quant (the down
    output is summed unweighted). Since an fp8 value x pow2 scale is exact in bf16, the
    round-tripped activation entering the bf16 decode kernel is bit-identical to the reference's
    dequantized FP8 activation."""
    T, top_k = slots.shape
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2

    x = act_quant_fp8_roundtrip(x, act_block)  # gate_up activation -> FP8 round-trip (no clone)
    gate_up = _grouped_decode(
        x, gate_up_packed, gate_up_scale, slots, None,
        a_row_is_route=False, mul_routed_weight=False,
    )  # [T, top_k, 2I]
    # [T, top_k, I]: routing weight applied in fp32, then the down activation's FP8 round-trip, one pass
    act = fused_swiglu(gate_up, swiglu_limit, topk_weights, act_block=act_block).reshape(T * top_k, I)
    down = _grouped_decode(
        act, down_packed, down_scale, slots, None,
        a_row_is_route=True, mul_routed_weight=False,
    )  # [T, top_k, H]
    return down.sum(dim=1, dtype=out_dtype or x.dtype)  # [T, H]: fp32 accumulation over the bf16 route outputs


_GROUPED_MIN_ROUTES = 768  # existing short-prefill crossover; batch-invariant calls bypass it


def _grouped_prefill(
    a: torch.Tensor,             # [A_rows, K] compute dtype (FP8 round-tripped)
    packed_cache: torch.Tensor,  # [S, N, K//2] uint8
    scale_cache: torch.Tensor,   # [S, N, K//32] e8m0
    c: torch.Tensor,             # [T, top_k, N] output, flat-indexed over routes
    tw_flat: torch.Tensor,       # [T*top_k] fp32
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: dict,
) -> None:
    N = packed_cache.shape[1]
    K = packed_cache.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale_u8 = scale_cache.view(torch.uint8)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_dsfp4_moe_kernel[grid](
        a, packed_cache, scale_u8, c, tw_flat,
        sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid,
        a.stride(0), a.stride(1),
        packed_cache.stride(0), packed_cache.stride(1), packed_cache.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        c.stride(-2), c.stride(-1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_compute_type(a.dtype),
        num_warps=cfg.get("num_warps", 4),
        num_stages=cfg.get("num_stages", 3),
    )


def routed_experts_fp4_prefill(
    x: torch.Tensor,             # [T, H] compute dtype
    slots: torch.Tensor,         # [T, top_k] int32 -> bank row in [0, num_rows)
    topk_weights: torch.Tensor,  # [T, top_k] fp32 (incl. route_scale + renorm)
    gate_up_packed: torch.Tensor,  # [S, 2I, H//2] uint8
    gate_up_scale: torch.Tensor,   # [S, 2I, H//32] e8m0
    down_packed: torch.Tensor,     # [S, H, I//2] uint8
    down_scale: torch.Tensor,      # [S, H, I//32] e8m0
    swiglu_limit: float,
    num_rows: int,
    act_block: int = 128,
    out_dtype: torch.dtype | None = None,
    *,
    batch_invariant: bool = False,
) -> torch.Tensor:
    """Grouped prefill with optional batch-invariant dispatch for prefix recomputation.

    Ordinary short prefills use GEMV below the established 768-route crossover.
    Batch-invariant callers use one GEMM configuration for every prefill size.
    """
    T, top_k = slots.shape
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2
    routes = T * top_k
    if not batch_invariant and routes < _GROUPED_MIN_ROUTES:
        return routed_experts_fp4(
            x, slots, topk_weights, gate_up_packed, gate_up_scale, down_packed, down_scale,
            swiglu_limit, act_block=act_block, out_dtype=out_dtype,
        )
    # One static config for every density (no autotune): the kernel is
    # dequant-floor-bound, so per-expert padding at BLOCK_M=64 costs the same
    # as tighter tiles while keeping the wgmma-wide M tile on sm_90.
    cfg = dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
        num_warps=8, num_stages=1,
    )
    sorted_ids, expert_ids, ntpp = moe_align_block_size(slots, cfg["BLOCK_SIZE_M"], num_rows)
    tw = topk_weights.reshape(-1).contiguous()

    x = act_quant_fp8_roundtrip(x, act_block)  # gate_up activation -> FP8 round-trip (no clone)
    gate_up = torch.empty((T, top_k, two_I), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        x, gate_up_packed, gate_up_scale, gate_up, tw,
        sorted_ids, expert_ids, ntpp, routes, top_k, False, cfg,
    )
    # [T, top_k, I]: routing weight applied in fp32, then the down activation's FP8 round-trip, one pass
    act = fused_swiglu(gate_up, swiglu_limit, tw, act_block=act_block).reshape(routes, I)
    down = torch.empty((T, top_k, H), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        act, down_packed, down_scale, down, tw,
        sorted_ids, expert_ids, ntpp, routes, 1, False, cfg,
    )
    return down.sum(dim=1, dtype=out_dtype or x.dtype)  # [T, H]


__all__ = ["routed_experts_fp4", "routed_experts_fp4_prefill", "_grouped_decode"]
