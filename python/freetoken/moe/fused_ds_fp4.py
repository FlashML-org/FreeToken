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

from freetoken.kernel.triton.dsv4.fp8_linear import (
    act_quant_fp8_inplace,
    act_quant_fp8_roundtrip,
)
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
    (a_row=route). FP4 weights are dequantized inline from the slot cache.
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

    # Decode is a per-route GEMV, bound by the inline FP4 dequant (LUT gather + block
    # scale), not weight HBM bandwidth -- so it tops out ~29% of HBM peak. BN=16/
    # BKB=128/1 warp maximizes memory-level parallelism (many single-warp CTAs, no
    # cross-warp reduction) -> ~950 GB/s on H100 (was ~830 at BN=8/BKB=256/2). K_BYTES
    # must be a multiple of BLOCK_SIZE_KB (gate_up 2048, down 1024 -- both /128).
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
) -> torch.Tensor:
    """Full routed-expert output (summed over the top-k routes), excludes shared expert.

    Precision matches the reference ``fp4_gemm(act_quant(x, 128), W_fp4)``: the gate_up
    and down activations are FP8-round-tripped (block 128, ue8m0) before each GEMM. Since
    an fp8 value x pow2 scale is exact in bf16, the round-tripped activation entering the
    bf16 decode kernel is bit-identical to the reference's dequantized FP8 activation
    (validated max diff = 0 vs the tilelang ``fp4_gemm`` reference)."""
    T, top_k = slots.shape
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2

    x = act_quant_fp8_roundtrip(x, 128)  # gate_up activation -> FP8 round-trip (no clone)
    gate_up = _grouped_decode(
        x, gate_up_packed, gate_up_scale, slots, None,
        a_row_is_route=False, mul_routed_weight=False,
    )  # [T, top_k, 2I]
    act = fused_swiglu(gate_up, swiglu_limit)  # [T, top_k, I]

    act = act.reshape(T * top_k, I)
    act_quant_fp8_inplace(act, 128)  # down activation -> FP8 round-trip
    down = _grouped_decode(
        act, down_packed, down_scale, slots, topk_weights,
        a_row_is_route=True, mul_routed_weight=True,
    )  # [T, top_k, H]
    return down.sum(dim=1)  # [T, H]


# Above this the grouped GEMM beats the per-route GEMV despite its padding;
# below, the GEMV's exact routes*N work wins (short streaming chunks). The
# grouped kernel sits on its dequant floor (~6.4-6.9ms/GEMM-pair at DSV4
# geometry) for any chunk size, so the crossover is where the GEMV's
# routes-proportional cost reaches that floor (H100 sweep).
_GROUPED_MIN_ROUTES = 768


def _prefill_cfg(routes: int, num_rows: int) -> dict:
    """Tile config for the grouped prefill GEMM.

    sm_90 keeps the shipped constants: one static config for every density, the
    kernel being dequant-floor-bound means per-expert padding at BLOCK_M=64 costs
    the same as tighter tiles, and 64 is the wgmma-wide M tile.

    sm_89 has no wgmma, so nothing pins M to 64 -- and without that floor the
    moe_align padding BLOCK_M=64 pays becomes the dominant term at low route
    density. Measured on main, on an RTX 6000 Ada (142 SM, 96 MiB L2, 960 GB/s)
    at DSV4 TP=2 geometry, 288 configs per shape, every one checked against the
    sm_90 config's output (0 numeric rejections):

        routes  rows/expert   shipped     sm_89 best               gain
          1536      6    gate_up   4.233 ms  BM16  BN64  BK128 w4  2.376 ms  +43.9%
          1536      6    down      2.298 ms  BM16  BN128 BK128 w4  1.121 ms  +51.2%
          6144     24    gate_up   5.305 ms  BM32  BN128 BK128 w8  3.375 ms  +36.4%
          6144     24    down      2.665 ms  BM32  BN128 BK128 w4  1.444 ms  +45.8%
         24576     96    gate_up  11.098 ms  BM128 BN128 BK128 w8  6.587 ms  +40.6%
         24576     96    down      5.648 ms  BM128 BN128 BK64  w8  3.420 ms  +39.5%

    The optimum tracks routes-per-expert (the padding term), so M is picked from
    it rather than fixed. num_stages barely moves the result on sm_89 (stages=1
    wins or ties at four of the six shapes), so it stays 1 -- the win is tile
    shape, not pipelining.
    """
    if torch.cuda.get_device_capability()[0] >= 9:
        return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=8, num_warps=8, num_stages=1)
    per_expert = routes / max(1, num_rows)
    if per_expert <= 12:
        bm, bn = 16, 64
    elif per_expert <= 48:
        bm, bn = 32, 128
    else:
        bm, bn = 128, 128
    return dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
                GROUP_SIZE_M=8, num_warps=8, num_stages=1)


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
) -> torch.Tensor:
    """Grouped-GEMM counterpart of :func:`routed_experts_fp4` for dense prefill
    chunks: one moe_align sort shared by both GEMMs, each expert's weights
    dequantized once per N-tile instead of once per route. Same FP8
    round-tripped activations; differs from the GEMV only in fp32 accumulation
    order (tl.dot tree vs sequential K-walk)."""
    T, top_k = slots.shape
    if T * top_k < _GROUPED_MIN_ROUTES:
        return routed_experts_fp4(
            x, slots, topk_weights,
            gate_up_packed, gate_up_scale, down_packed, down_scale, swiglu_limit,
        )
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2
    routes = T * top_k
    cfg = _prefill_cfg(routes, num_rows)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(slots, cfg["BLOCK_SIZE_M"], num_rows)
    tw = topk_weights.reshape(-1).contiguous()

    x = act_quant_fp8_roundtrip(x, 128)  # gate_up activation -> FP8 round-trip (no clone)
    gate_up = torch.empty((T, top_k, two_I), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        x, gate_up_packed, gate_up_scale, gate_up, tw,
        sorted_ids, expert_ids, ntpp, routes, top_k, False, cfg,
    )
    act = fused_swiglu(gate_up, swiglu_limit)  # [T, top_k, I]

    act = act.reshape(routes, I)
    act_quant_fp8_inplace(act, 128)  # down activation -> FP8 round-trip
    down = torch.empty((T, top_k, H), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        act, down_packed, down_scale, down, tw,
        sorted_ids, expert_ids, ntpp, routes, 1, True, cfg,
    )
    return down.sum(dim=1)  # [T, H]


__all__ = ["routed_experts_fp4", "routed_experts_fp4_prefill", "_grouped_decode"]
