"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Dict, Tuple

import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.backend import is_rocm
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import gated_act_and_mul
from freetoken.moe.fused import moe_align_block_size
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KB = 128
_DECODE_WARPS = 4

# Marlin-style decode: every K-loop load must keep the word tile's register layout, so BLOCK_N * BLOCK_KW <= 128 * num_warps and BLOCK_KW <= 32 at 4 warps.
# Long K (gate_up) uses the narrow N tile for more programs at M=1; short K (down) uses the wide one.
_DECODE_MARLIN_BLOCK_N = 16
_DECODE_MARLIN_BLOCK_KW = 16
_DECODE_MARLIN_UNROLL = 2
_DECODE_MARLIN_WARPS = 4
_DECODE_MARLIN_DEEPK_BLOCK_N = 8
_DECODE_MARLIN_DEEPK_BLOCK_KW = 32
_DECODE_MARLIN_DEEPK_UNROLL = 2
_DECODE_MARLIN_DEEPK_THRESHOLD = 2048


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction
    (:func:`_decode_nvfp4_marlin_kernel`). ``packed`` is the uint8 ``[S, N, K//2]`` bank;
    it is reinterpreted as int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4)."""
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    deep_k = K >= _DECODE_MARLIN_DEEPK_THRESHOLD
    block_n = _DECODE_MARLIN_DEEPK_BLOCK_N if deep_k else _DECODE_MARLIN_BLOCK_N
    block_kw = _DECODE_MARLIN_DEEPK_BLOCK_KW if deep_k else _DECODE_MARLIN_BLOCK_KW
    unroll = _DECODE_MARLIN_DEEPK_UNROLL if deep_k else _DECODE_MARLIN_UNROLL
    # int64 activation loads need contiguous 16-bit rows that start 16-byte aligned
    a_wide = (
        a.dtype in (torch.bfloat16, torch.float16)
        and a.stride(1) == 1 and a.stride(0) % 8 == 0 and a.data_ptr() % 16 == 0
    )
    grid = (total_routes, triton.cdiv(N, block_n))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        UNROLL=unroll,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        A_WIDE=a_wide,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_MARLIN_WARPS,
    )


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


# rho = bf16 tensor-core FLOP / fp32 ALU FLOP per SM per clock, with fp32 accumulate (the only accumulator for bf16 MMA).
# Each M block decodes its weight tile once on the ALUs, and the MMA also runs on the padding up to BLOCK_SIZE_M.
# High rho makes decode the cost, so big blocks win; low rho makes padded MMA the cost, so small blocks win.
# Spec, FLOP/clk/SM: GeForce RTX 30/40/50 512 / 256 = 2, as GeForce runs fp32 accumulate at half rate.
# RTX A, RTX Ada Generation, RTX PRO Blackwell, A10, A40, L4, L40S 1024 / 256 = 4; A100 2048 / 128 = 16; H100 4096 / 256 = 16; H20 148 / 44 TFLOP/s = 3.4.
# Measured (cuBLAS bf16 / Triton fp32 FMA): RTX PRO 6000 4.5, RTX 5090 2.3, RTX 4090 1.9, RTX 3080 Ti 1.6.
_PRO_GPU_RE = re.compile(
    r"RTX\s*A\d{3,4}"                        # RTX A400 .. A6000
    r"|RTX\s*\d{3,4}E?\s*(SFF\s*)?-?\s*Ada"  # RTX 500 .. 6000 Ada Generation
    r"|RTX\s*PRO\s*\d{3,4}"                  # RTX PRO 500 .. 6000 Blackwell
    r"|RTX\s*\d{4}D\b"                       # RTX 6000D
    r"|Quadro|TITAN"
    r"|^(NVIDIA\s+)?(GRID\s+)?(A2|A10[GM]?|A16|A40|L2|L20|L4|L40S?)\b",
    re.I,
)
# also matches names without "GeForce", e.g. "NVIDIA RTX 4090"
_GEFORCE_GPU_RE = re.compile(r"GeForce|RTX\s*[2-6]0[5-9]0(?!\d)", re.I)


def _tensor_rho(name: str, major: int, minor: int) -> float:
    if major not in (8, 12) or (major, minor) == (8, 0):
        return 3.4 if re.search(r"\bH20\b", name) else 16.0
    if _PRO_GPU_RE.search(name):
        return 4.0
    if _GEFORCE_GPU_RE.search(name):
        return 2.0
    logger.warning("unknown GPU %r: NVFP4 MoE prefill assumes full-rate fp32-accumulate tensor cores", name)
    return 4.0


# (min rho, tokens per expert where BLOCK_SIZE_M becomes 32, 64, 128), from sweeps on H100, RTX PRO 6000 and RTX 5090 / 4090 / 3080 Ti
_PREFILL_BLOCK_M_STEPS = ((8.0, (12, 24, 64)), (3.0, (16, 36, 150)), (0.0, (16, 150, float("inf"))))
# BLOCK_SIZE_M -> (BLOCK_SIZE_N, BLOCK_SIZE_KB, num_stages)
_PREFILL_TILES_H100 = {16: (64, 64, 3), 32: (64, 32, 3), 64: (64, 32, 3), 128: (64, 32, 3)}
_PREFILL_TILES_RTX_PRO = {16: (64, 64, 4), 32: (128, 64, 3), 64: (128, 64, 2), 128: (64, 32, 2)}
_PREFILL_TILES_GEFORCE = {16: (64, 64, 4), 32: (128, 64, 2), 64: (128, 32, 2)}


@functools.lru_cache(maxsize=None)
def _prefill_tuning(device_index: int) -> Tuple[Tuple[float, float, float], Dict[int, Tuple[int, int, int]]]:
    if is_rocm():
        # TODO: tune for AMD GPUs; they keep the H100 tuning
        return _PREFILL_BLOCK_M_STEPS[0][1], _PREFILL_TILES_H100
    props = torch.cuda.get_device_properties(device_index)
    rho = _tensor_rho(props.name, props.major, props.minor)
    steps = next(s for min_rho, s in _PREFILL_BLOCK_M_STEPS if rho >= min_rho)
    # the RTX tables come from cc 8.6 / 8.9 / 12.0 sweeps (mma.sync, 99 KB shared memory); other GPUs keep the H100 one
    if (props.major, props.minor) not in ((8, 6), (8, 9), (12, 0)):
        return steps, _PREFILL_TILES_H100
    return steps, _PREFILL_TILES_RTX_PRO if rho >= 3.0 else _PREFILL_TILES_GEFORCE


def _prefill_config(M: int, top_k: int, num_experts: int, device_index: int) -> Dict[str, int]:
    # moe_align_block_size pads each expert's token count to a multiple of BLOCK_SIZE_M, so triton.autotune cannot pick it.
    (to32, to64, to128), tiles = _prefill_tuning(device_index)
    per_expert = M * top_k / num_experts
    block_m = 128 if per_expert >= to128 else 64 if per_expert >= to64 else 32 if per_expert >= to32 else 16
    block_n, block_kb, stages = tiles[block_m]
    return dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_KB=block_kb,
                GROUP_SIZE_M=8, num_warps=4, num_stages=stages)


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config(M, top_k, num_experts, hidden_states.device.index)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]
