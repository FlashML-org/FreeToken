"""Kernel dispatch for the NVFP4 MoE GEMMs: which kernel runs, with which tiles and switches, and its launch.

The kernels are templates (:mod:`freetoken.kernel.triton.nvfp4_fused_moe`, :mod:`freetoken.kernel.gluon.nvfp4.moe.sm120`, :mod:`freetoken.kernel.gluon.nvfp4.moe.sm120_w4a4`);
:mod:`freetoken.moe.fused_nvfp4` strings the GEMMs into the MoE layer.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, replace
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


# BLOCK_SIZE_M -> (BLOCK_SIZE_N, BLOCK_SIZE_KB, num_stages)
_PREFILL_TILES_H100 = {16: (64, 64, 3), 32: (64, 32, 3), 64: (64, 32, 3), 128: (64, 32, 3)}
_PREFILL_TILES_RTX_PRO = {16: (64, 64, 4), 32: (128, 64, 3), 64: (128, 64, 2), 128: (64, 32, 2)}
_PREFILL_TILES_GEFORCE = {16: (64, 64, 4), 32: (128, 64, 2), 64: (128, 32, 2)}


@dataclass(frozen=True)
class _PrefillTuning:
    """Triton prefill GEMM tuning of one GPU class."""

    steps: Tuple[float, float, float]  # tokens per expert where BLOCK_SIZE_M becomes 32, 64, 128
    tiles: Dict[int, Tuple[int, int, int]]
    pair_decode_below: float = 0.0  # tokens per expert under which FAST_DECODE stays 0
    small_tile_decode: int = 2  # FAST_DECODE of BLOCK_SIZE_M <= 32
    large_tile_decode: int = 1  # FAST_DECODE of BLOCK_SIZE_M >= 64
    scale_words: bool = False  # SCALE_WORDS on BLOCK_SIZE_M 64 word-decode tiles
    long_k_warps: int = 4  # num_warps of SCALE_WORDS GEMMs with K >= _LONG_K


_INF = float("inf")
_LONG_K = 2048
_TUNING_H100 = _PrefillTuning((12, 24, 64), _PREFILL_TILES_H100)
_TUNING_RTX_PRO = _PrefillTuning((16, 36, 150), _PREFILL_TILES_RTX_PRO)
_TUNING_GEFORCE = _PrefillTuning((16, 150, _INF), _PREFILL_TILES_GEFORCE)
# RTX 4090 A/B: up to 32 tokens per expert the pair decode streams the weights 2-18% faster, from 51 words win 11-14%, also on 32-row tiles
_SM89_DECODE = dict(pair_decode_below=48, small_tile_decode=1)
# H100 A/B: bytes beat words at BLOCK_SIZE_M 64 / 128 by 1-17% per layer (24-31% on Gemma-4-26B-A4B's K = 704 down GEMM)
_SM90_DECODE = dict(small_tile_decode=2, large_tile_decode=2)
# RTX 4090 / 3080 Ti layer A/B: scale words + 8 warps on the long-K GEMM give 1.03-1.09x / 1.05-1.19x; with them sm_86 gains from 64-row tiles at 100 tokens per expert, sm_89 does not
_GEFORCE_SCALE_WORDS = dict(scale_words=True, long_k_warps=8)
# (rho class, compute capability) -> tuning, from sweeps on H100, RTX PRO 6000 and RTX 5090 / 4090 / 3080 Ti; (class, None) covers the rest of the class.
# The RTX tiles come from cc 8.6 / 8.9 / 12.0 sweeps (mma.sync, 99 KB shared memory); other GPUs keep the H100 tiles.
_PREFILL_TUNING = {
    ("wide", (9, 0)): replace(_TUNING_H100, **_SM90_DECODE),
    ("wide", None): _TUNING_H100,
    ("pro", (8, 6)): _TUNING_RTX_PRO,
    ("pro", (8, 9)): replace(_TUNING_RTX_PRO, **_SM89_DECODE),
    ("pro", (9, 0)): replace(_TUNING_RTX_PRO, tiles=_PREFILL_TILES_H100, **_SM90_DECODE),
    ("pro", (12, 0)): _TUNING_RTX_PRO,
    ("pro", None): replace(_TUNING_RTX_PRO, tiles=_PREFILL_TILES_H100),
    ("geforce", (8, 6)): replace(_TUNING_GEFORCE, steps=(16, 100, _INF), **_GEFORCE_SCALE_WORDS),
    ("geforce", (8, 9)): replace(_TUNING_GEFORCE, **_SM89_DECODE, **_GEFORCE_SCALE_WORDS),
    ("geforce", (12, 0)): _TUNING_GEFORCE,
    ("geforce", None): replace(_TUNING_GEFORCE, tiles=_PREFILL_TILES_H100),
}
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
def _device_rho(device_index: int) -> float:
    props = torch.cuda.get_device_properties(device_index)
    return _tensor_rho(props.name, props.major, props.minor)


@functools.lru_cache(maxsize=None)
def _prefill_tuning(device_index: int) -> _PrefillTuning:
    if is_rocm():
        # TODO: tune for AMD GPUs; they keep the H100 tuning
        return _TUNING_H100
    rho = _device_rho(device_index)
    rho_class = "wide" if rho >= 8.0 else "pro" if rho >= 3.0 else "geforce"
    cc = torch.cuda.get_device_capability(device_index)
    return _PREFILL_TUNING.get((rho_class, cc), _PREFILL_TUNING[(rho_class, None)])


@functools.lru_cache(maxsize=None)
def _prefill_block_m_steps(device_index: int) -> Tuple[float, float, float]:
    steps = _prefill_tuning(device_index).steps
    if _gluon_prefill(device_index) is not None:
        rho = _device_rho(device_index)
        steps = next((s for min_rho, s in _PREFILL_BLOCK_M_STEPS_GLUON if rho >= min_rho), steps)
    return steps


@functools.lru_cache(maxsize=None)
def _tma_device(device_index: int) -> bool:
    return not is_rocm() and torch.cuda.get_device_capability(device_index) == (12, 0)


def _prefill_use_tma(packed: torch.Tensor, scale: torch.Tensor, K: int, device_index: int) -> bool:
    # a TMA box row is >= 16 bytes and every global stride a multiple of 16 bytes, so K // 16 scale bytes need K % 256 == 0;
    # the TMA scale path reads the scales as fp8, so the forced e4m3 emulation stays on pointer loads
    return (_tma_device(device_index) and e4m3_native() and K % 256 == 0
            and packed.is_contiguous() and scale.is_contiguous()
            and packed.data_ptr() % 16 == 0 and scale.data_ptr() % 16 == 0)


def _prefill_use_scale_words(scale: torch.Tensor, K: int, cfg: Dict[str, Any], device_index: int) -> bool:
    # K // 16 scale bytes per row split into whole int32 words, and a K step reads at least one word
    return (_prefill_tuning(device_index).scale_words and cfg["BLOCK_SIZE_M"] == 64 and cfg["BLOCK_SIZE_KB"] >= 32
            and K % 64 == 0 and scale.is_contiguous() and scale.data_ptr() % 4 == 0)


def _prefill_fast_decode(block_m: int, per_expert: float, device_index: int) -> int:
    """FAST_DECODE of the prefill kernel: 0 keeps the fp16 pair decode (ROCm: the fast decode is PTX inline asm), 1 words, 2 bytes."""
    if is_rocm():
        return 0
    tuning = _prefill_tuning(device_index)
    if per_expert < tuning.pair_decode_below:
        return 0
    # the word decode halves the loop but stages the decoded tile through shared memory; small tiles keep the byte form
    return tuning.small_tile_decode if block_m <= 32 else tuning.large_tile_decode


def prefill_config(M: int, top_k: int, num_experts: int, device_index: int) -> Dict[str, int]:
    # moe_align_block_size pads each expert's token count to a multiple of BLOCK_SIZE_M, so triton.autotune cannot pick it.
    to32, to64, to128 = _prefill_block_m_steps(device_index)
    per_expert = M * top_k / num_experts
    block_m = 128 if per_expert >= to128 else 64 if per_expert >= to64 else 32 if per_expert >= to32 else 16
    block_n, block_kb, stages = _prefill_tuning(device_index).tiles[block_m]
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
    scale_words = not use_tma and fast == 1 and _prefill_use_scale_words(scale, K, cfg, a.device.index)
    if scale_words:
        scale = scale.view(torch.int32)
        if K >= _LONG_K:
            cfg["num_warps"] = _prefill_tuning(a.device.index).long_k_warps
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
        SCALE_WORDS=scale_words,
        **cfg,
    )


@functools.lru_cache(maxsize=None)
def _w4a4_prefill(device_index: int):
    """The Gluon module of the W4A4 prefill GEMMs on this GPU, or None."""
    from freetoken.kernel import gluon

    if torch.cuda.get_device_capability(device_index) != (12, 0) or not gluon.available():
        return None
    from freetoken.kernel.gluon.nvfp4.moe import sm120_w4a4

    return sm120_w4a4


def w4a4_available(device_index: int) -> bool:
    return _w4a4_prefill(device_index) is not None


# Below this many tokens per expert both prefills stream the weights at the same speed (RTX PRO 6000 and RTX 5090 sweeps over the
# NVFP4 MoE shapes); W4A16 keeps the exact activations there
_W4A4_MIN_TOKENS_PER_EXPERT = 4
# (min rho, tokens per expert where BLOCK_SIZE_M becomes 128), from the same sweeps
_W4A4_BLOCK_M_STEPS = ((3.0, 128), (0.0, 64))


@functools.lru_cache(maxsize=None)
def _w4a4_device(device_index: int) -> Tuple[float, int]:
    props = torch.cuda.get_device_properties(device_index)
    rho = _tensor_rho(props.name, props.major, props.minor)
    return next(t for min_rho, t in _W4A4_BLOCK_M_STEPS if rho >= min_rho), props.L2_cache_size


def w4a4_prefill_config(M: int, top_k: int, num_experts: int, H: int, I: int, activation: str, dtype: torch.dtype,
                        device_index: int) -> Dict[str, Any] | None:
    """Tiles of the W4A4 prefill GEMMs, or None where the W4A16 prefill runs instead."""
    w4a4 = _w4a4_prefill(device_index)
    if w4a4 is None or dtype != torch.bfloat16 or activation not in w4a4.ACTIVATIONS or not w4a4.supports(H, I):
        return None
    to128, l2_bytes = _w4a4_device(device_index)
    per_expert = M * top_k / num_experts
    if per_expert < _W4A4_MIN_TOKENS_PER_EXPERT:
        return None
    block_m = 128 if per_expert >= to128 else 64
    common = dict(BLOCK_SIZE_M=block_m, GROUP_SIZE_M=8, MMA_WARPS_N=4, LOAD_WARPS=4, num_warps=8)
    # the gated tile holds BN / 2 gate rows and their up rows, so I must split into whole half tiles
    gate_up = dict(common, BLOCK_SIZE_N=256 if I % 128 == 0 else 128, BLOCK_SIZE_K=128 if H % 128 == 0 else 64,
                   N_GROUP=0, STAGES=3, LOAD_REGS=88)
    if I < 1024:
        # short K: the persistent kernel overlaps a tile's atomics with the next mainloop; N strips keep the [M, H] output
        # in L2 once it outgrows half of it
        down = dict(common, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, STAGES=4, N_GROUP=2 if M * H * 2 > l2_bytes // 2 else 0,
                    LOAD_REGS=56, EPI_WARPS=4, EPI_REGS=72, PERSISTENT=True)
    else:
        down = dict(common, BLOCK_SIZE_N=256 if H % 256 == 0 else 128, BLOCK_SIZE_K=128 if I % 128 == 0 else 64,
                    N_GROUP=0, STAGES=3, LOAD_REGS=88, PERSISTENT=False)
    return dict(BLOCK_SIZE_M=block_m, gate_up=gate_up, down=down)


def w4a4_gate_up_gemm(xq, xs, g_in, packed, scale, glob, q, qs, g_out, sorted_ids, expert_ids, num_tokens_post_padded,
                      num_valid_tokens, top_k, activation, act_alpha, act_limit, cfg):
    _w4a4_prefill(xq.device.index).gate_up_gemm(
        xq, xs, g_in, packed, scale, glob, q, qs, g_out, sorted_ids, expert_ids, num_tokens_post_padded,
        num_valid_tokens, top_k, activation, act_alpha, act_limit, cfg)


def w4a4_down_gemm(xq, xs, g_in, packed, scale, glob, out, topk_weights_flat, sorted_ids, expert_ids,
                   num_tokens_post_padded, num_valid_tokens, top_k, cfg):
    _w4a4_prefill(xq.device.index).down_gemm(
        xq, xs, g_in, packed, scale, glob, out, topk_weights_flat, sorted_ids, expert_ids, num_tokens_post_padded,
        num_valid_tokens, top_k, cfg)
