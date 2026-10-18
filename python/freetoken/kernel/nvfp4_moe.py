"""Kernel dispatch for the inline-dequant NVFP4 MoE GEMMs: which kernel runs, with which tiles and switches, and its launch.

The kernels are templates (:mod:`freetoken.kernel.triton.nvfp4_fused_moe`, :mod:`freetoken.kernel.gluon.nvfp4.moe.sm120`);
:mod:`freetoken.moe.fused_nvfp4` strings the GEMMs into the MoE layer.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Dict, Tuple

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from freetoken.kernel.backend import is_rocm
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (decode_gemm), kept only for A/B.
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


def decode_gemm(
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


def decode_gemm_marlin(
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
# steps while the Gluon kernel runs BLOCK_SIZE_M >= 64, from an RTX PRO 6000 sweep; classes without an entry keep the Triton steps
_PREFILL_BLOCK_M_STEPS_GLUON = ((3.0, (16, 36, 96)),)


# Gluon prefill GEMM tiles: BLOCK_SIZE_N -> (BLOCK_SIZE_K, setmaxnreg of the loader / decode warps)
_GLUON_PREFILL_TILES = {128: (64, 96, 96), 256: (32, 88, 72)}


def _gluon_prefill_config(N: int, block_m: int, mul_routed_weight: bool) -> Dict[str, Any]:
    # BLOCK_SIZE_M 64 tiles are short (few tokens per expert, often short K): persistent CTAs overlap one tile's epilogue and ring fill with the next tile, and a third x / decoded-weight stage absorbs DRAM latency
    persistent = block_m == 64
    # a 256-row tile halves the token loads per weight byte, but a partial last tile still decodes all 256 rows
    # the persistent epilogue keeps one 64-bit address per element in registers, which spills with the routed-weight loads
    block_n = 256 if N % 256 == 0 and not (persistent and mul_routed_weight) else 128
    block_k, load_regs, decode_regs = _GLUON_PREFILL_TILES[block_n]
    return dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k, GROUP_SIZE_M=8,
                STAGES=3 if persistent else 2, W_STAGES=4, PERSISTENT=persistent, DECODE_WARPS=4, MMA_WARPS_N=4,
                LOAD_REGS=load_regs, DECODE_REGS=decode_regs, num_warps=8)


@functools.lru_cache(maxsize=None)
def _gluon_prefill(device_index: int):
    """The Gluon module whose prefill GEMM replaces the Triton kernel on this GPU, or None."""
    from freetoken.kernel import gluon

    if torch.cuda.get_device_capability(device_index) != (12, 0) or not gluon.available():
        return None
    from freetoken.kernel.gluon.nvfp4.moe import sm120

    return sm120


@functools.lru_cache(maxsize=None)
def _prefill_tuning(device_index: int) -> Tuple[Tuple[float, float, float], Dict[int, Tuple[int, int, int]]]:
    if is_rocm():
        # TODO: tune for AMD GPUs; they keep the H100 tuning
        return _PREFILL_BLOCK_M_STEPS[0][1], _PREFILL_TILES_H100
    props = torch.cuda.get_device_properties(device_index)
    rho = _tensor_rho(props.name, props.major, props.minor)
    steps = next(s for min_rho, s in _PREFILL_BLOCK_M_STEPS if rho >= min_rho)
    if _gluon_prefill(device_index) is not None:
        steps = next((s for min_rho, s in _PREFILL_BLOCK_M_STEPS_GLUON if rho >= min_rho), steps)
    # the RTX tables come from cc 8.6 / 8.9 / 12.0 sweeps (mma.sync, 99 KB shared memory); other GPUs keep the H100 one
    if (props.major, props.minor) not in ((8, 6), (8, 9), (12, 0)):
        return steps, _PREFILL_TILES_H100
    return steps, _PREFILL_TILES_RTX_PRO if rho >= 3.0 else _PREFILL_TILES_GEFORCE


@functools.lru_cache(maxsize=None)
def _tma_device(device_index: int) -> bool:
    return not is_rocm() and torch.cuda.get_device_capability(device_index) == (12, 0)


def _prefill_use_tma(packed: torch.Tensor, scale: torch.Tensor, K: int, device_index: int) -> bool:
    # a TMA box row is >= 16 bytes and every global stride a multiple of 16 bytes, so K // 16 scale bytes need K % 256 == 0;
    # the TMA scale path reads the scales as fp8, so the forced e4m3 emulation stays on pointer loads
    return (_tma_device(device_index) and e4m3_native() and K % 256 == 0
            and packed.is_contiguous() and scale.is_contiguous()
            and packed.data_ptr() % 16 == 0 and scale.data_ptr() % 16 == 0)


@functools.lru_cache(maxsize=None)
def _device_cc(device_index: int) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _prefill_fast_decode(block_m: int, per_expert: float, device_index: int) -> int:
    """FAST_DECODE of the prefill kernel: 0 keeps the fp16 pair decode (ROCm: the fast decode is PTX inline asm), 1 words, 2 bytes."""
    if is_rocm():
        return 0
    if _device_cc(device_index) == (8, 9):
        # RTX 4090 A/B: up to 32 tokens per expert the pair decode streams the weights 2-18% faster, from 51 words win 11-14%
        return 0 if per_expert < 48 else 1
    if _device_cc(device_index) == (9, 0):
        # H100 A/B: bytes beat words at BLOCK_SIZE_M 64 / 128 by 1-17% per layer (24-31% on Gemma-4-26B-A4B's K = 704 down GEMM)
        return 2
    # the word decode halves the loop but stages the decoded tile through shared memory; small tiles keep the byte form
    return 2 if block_m <= 32 else 1


def prefill_config(M: int, top_k: int, num_experts: int, device_index: int) -> Dict[str, int]:
    # moe_align_block_size pads each expert's token count to a multiple of BLOCK_SIZE_M, so triton.autotune cannot pick it.
    (to32, to64, to128), tiles = _prefill_tuning(device_index)
    per_expert = M * top_k / num_experts
    block_m = 128 if per_expert >= to128 else 64 if per_expert >= to64 else 32 if per_expert >= to32 else 16
    block_n, block_kb, stages = tiles[block_m]
    return dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_KB=block_kb,
                GROUP_SIZE_M=8, num_warps=4, num_stages=stages,
                FAST_DECODE=_prefill_fast_decode(block_m, per_expert, device_index))


def prefill_gemm(
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
    gluon = _gluon_prefill(a.device.index)
    if gluon is not None and cfg["BLOCK_SIZE_M"] >= 64 and gluon.supports(a, packed, scale, c):
        gluon.prefill_gemm(a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids, num_tokens_post_padded,
                           num_valid_tokens, kernel_top_k, mul_routed_weight,
                           _gluon_prefill_config(packed.shape[1], cfg["BLOCK_SIZE_M"], mul_routed_weight))
        return
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    cfg = dict(cfg)
    fast = cfg.pop("FAST_DECODE", 0)
    if K % 8:
        fast = 0
    if fast == 1:
        packed = packed.view(torch.int32)
    use_tma = _prefill_use_tma(packed, scale, K, a.device.index)
    # the 16-byte scale box costs registers that the BLOCK_SIZE_M >= 64 tiles do not have
    tma_scales = cfg["BLOCK_SIZE_M"] <= 32
    w_desc = s_desc = None
    if use_tma:
        rows = packed.shape[0] * N
        box_k = cfg["BLOCK_SIZE_KB"] // 4 if fast == 1 else cfg["BLOCK_SIZE_KB"]
        w_desc = TensorDescriptor.from_tensor(packed.view(rows, packed.shape[2]), [cfg["BLOCK_SIZE_N"], box_k])
        if tma_scales:
            s_desc = TensorDescriptor.from_tensor(scale.view(rows, K // 16), [cfg["BLOCK_SIZE_N"], 16])
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        w_desc, s_desc,
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
        USE_TMA=use_tma,
        TMA_SCALES=tma_scales,
        FAST_DECODE=fast,
        **cfg,
    )
