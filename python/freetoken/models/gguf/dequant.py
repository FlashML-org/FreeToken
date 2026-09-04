"""GGML block-quant dequantization and type metadata.

This module serves two purposes:

1. **Type metadata for the packed GPU path** (the hot path): GGUF weights stay packed
   and are dequantized inside the borrowed ggml CUDA kernels (see ``freetoken.kernel.gguf``).
   The ``BLOCK_SHAPE`` table and :func:`row_bytes` are shared by ``GGUFLinear``,
   ``GGUFEmbedding``, and expert-bank loaders for weight allocation and unpacking.

2. **Pure-torch reference dequantizers** (CPU/test path): The :func:`dequantize` function
   and helper ``dequant_*`` routines materialize F32/F16 tensors at load (norms, scales,
   router) and cross-check CUDA kernels in tests. These implement only Q4_0 and Q6_K;
   the missing types are handled by the CUDA kernels in production.

``BLOCK_SHAPE`` covers all 21 types (F32, F16, BF16, STD_K, IQ); ``dequantize()`` and
``_DEQUANT`` cover Q4_0 and Q6_K only.

Each ``dequant_*`` takes raw little-endian bytes as a ``uint8`` tensor whose final axis
spans whole blocks, and returns values in *storage order* (ggml's fastest axis first);
the caller reshapes to torch shape (``dims[::-1]``). The math mirrors ``ggml-quants.c``.
"""

from __future__ import annotations

import torch

# ggml_type enum values. Mirrors the ggml.h enum in llama.cpp.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_IQ2_XXS = 16
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
GGML_IQ1_S = 19
GGML_IQ4_NL = 20
GGML_IQ3_S = 21
GGML_IQ2_S = 22
GGML_IQ4_XS = 23
GGML_IQ1_M = 29
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type. Derived from block structs in
# python/freetoken/kernel/csrc/gguf/ggml-common.h (lines 18-192).
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q4_1: (32, 20),
    GGML_Q5_0: (32, 22),
    GGML_Q5_1: (32, 24),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_IQ2_XXS: (256, 66),
    GGML_IQ2_XS: (256, 74),
    GGML_IQ3_XXS: (256, 98),
    GGML_IQ1_S: (256, 50),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ3_S: (256, 110),
    GGML_IQ2_S: (256, 82),
    GGML_IQ4_XS: (256, 136),
    GGML_IQ1_M: (256, 56),
    GGML_BF16: (1, 2),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_Q4_0: "Q4_0",
    GGML_Q4_1: "Q4_1",
    GGML_Q5_0: "Q5_0",
    GGML_Q5_1: "Q5_1",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ2_XXS: "IQ2_XXS",
    GGML_IQ2_XS: "IQ2_XS",
    GGML_IQ3_XXS: "IQ3_XXS",
    GGML_IQ1_S: "IQ1_S",
    GGML_IQ4_NL: "IQ4_NL",
    GGML_IQ3_S: "IQ3_S",
    GGML_IQ2_S: "IQ2_S",
    GGML_IQ4_XS: "IQ4_XS",
    GGML_IQ1_M: "IQ1_M",
    GGML_BF16: "BF16",
}

# CUDA kernel dispatch: which types each C function handles.
# Mirrors switch (type) in ggml_get_to_cuda (dequantize.cuh:541)
DEQUANT_TYPES = frozenset({
    GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
    GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ1_S, GGML_IQ4_NL,
    GGML_IQ3_S, GGML_IQ2_S, GGML_IQ4_XS, GGML_IQ1_M,
})

# Mirrors switch (type) in ggml_mul_mat_vec_a8 (gguf_kernel.cu:116)
MMVQ_TYPES = frozenset({
    GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
    GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ1_S, GGML_IQ4_NL,
    GGML_IQ3_S, GGML_IQ2_S, GGML_IQ4_XS, GGML_IQ1_M,
})

# Mirrors switch (type) in ggml_mul_mat_a8 (gguf_kernel.cu:219)
# I-quants do not have an MMQ (large-batch matmul) kernel.
MMQ_TYPES = frozenset({
    GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
    GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
})

# Mirrors switch (type) in ggml_moe_a8_vec (gguf_kernel.cu:577)
MOE_VEC_TYPES = frozenset({
    GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
    GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ1_S, GGML_IQ4_NL,
    GGML_IQ3_S, GGML_IQ2_S, GGML_IQ4_XS, GGML_IQ1_M,
})

# Mirrors switch (type) in ggml_moe_a8 (gguf_kernel.cu:369), whose coverage ggml_moe_get_block_size (gguf_kernel.cu:835) mirrors
# I-quants do not have an MMQ (grouped MoE large-batch) kernel.
MOE_MMQ_TYPES = frozenset({
    GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
    GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
})

# Unquantized types: no dequantization needed, handled by separate path in layers/gguf.py.
GGML_UNQUANTIZED = frozenset({GGML_F32, GGML_F16, GGML_BF16})


def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


def _f16_scales(raw: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret bytes ``[lo:hi]`` (2 per block) of each block row as fp16 -> fp32 [N,1]."""
    return raw[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def dequant_q4_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; ``w = d*(q-8)``.

    Byte ``j`` of the 16 holds element ``j`` in its low nibble and ``j+16`` in its high
    nibble, so storage order within the block is ``[lo0..lo15, hi0..hi15]``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return ((q - 8.0) * d).reshape(-1).to(out_dtype)


def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 256-elem super-block = 128B low nibbles + 64B high 2-bits + 16 int8
    sub-scales + fp16 ``d``. Direct vectorization of ggml's two-half loop."""
    raw = raw.reshape(-1, 210)
    n = raw.shape[0]
    ql = raw[:, 0:128]  # [n,128]
    qh = raw[:, 128:192]  # [n,64]
    sc = raw[:, 192:208].view(torch.int8).to(torch.float32)  # [n,16]
    d = _f16_scales(raw, 208, 210)  # [n,1]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # l in 0..15 -> is=0; l in 16..31 -> is=1 (per ggml: is = l/16).
    is_idx = (torch.arange(32, device=raw.device) // 16)  # [32] in {0,1}
    for h in range(2):  # two 128-elem halves of the super-block
        qlh = ql[:, h * 64:(h + 1) * 64]  # [n,64]
        qhh = qh[:, h * 32:(h + 1) * 32]  # [n,32]
        sch = sc[:, h * 8:(h + 1) * 8]  # [n,8]
        a = qlh[:, 0:32].to(torch.int32)  # ql[l]
        b = qlh[:, 32:64].to(torch.int32)  # ql[l+32]
        hb = qhh.to(torch.int32)  # qh[l]
        q1 = ((a & 0x0F) | (((hb >> 0) & 3) << 4)) - 32
        q2 = ((b & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
        q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
        q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
        s1 = sch.index_select(1, is_idx + 0).to(torch.float32)
        s2 = sch.index_select(1, is_idx + 2).to(torch.float32)
        s3 = sch.index_select(1, is_idx + 4).to(torch.float32)
        s4 = sch.index_select(1, is_idx + 6).to(torch.float32)
        base = h * 128
        y[:, base + 0:base + 32] = d * s1 * q1.to(torch.float32)
        y[:, base + 32:base + 64] = d * s2 * q2.to(torch.float32)
        y[:, base + 64:base + 96] = d * s3 * q3.to(torch.float32)
        y[:, base + 96:base + 128] = d * s4 * q4.to(torch.float32)
    return y.reshape(-1).to(out_dtype)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q6_K: dequant_q6_k,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) in pure torch (Q4_0, Q6_K, F32/F16/BF16 only).

    This is the CPU reference path for loading norms and scales. The packed GPU path
    (GGUFLinear, GGUFEmbedding, expert banks) dequantizes all 21 types via CUDA kernels;
    see ``BLOCK_SHAPE`` for the full type list.
    """
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"pure-torch dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} "
            f"not implemented (only Q4_0 and Q6_K supported in CPU path; "
            f"other types use CUDA kernels via GGUFLinear)"
        )
    return fn(raw, out_dtype)


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_Q4_0",
    "GGML_Q4_1",
    "GGML_Q5_0",
    "GGML_Q5_1",
    "GGML_Q8_0",
    "GGML_Q2_K",
    "GGML_Q3_K",
    "GGML_Q4_K",
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_IQ2_XXS",
    "GGML_IQ2_XS",
    "GGML_IQ3_XXS",
    "GGML_IQ1_S",
    "GGML_IQ4_NL",
    "GGML_IQ3_S",
    "GGML_IQ2_S",
    "GGML_IQ4_XS",
    "GGML_IQ1_M",
    "GGML_BF16",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "DEQUANT_TYPES",
    "MMVQ_TYPES",
    "MMQ_TYPES",
    "MOE_VEC_TYPES",
    "MOE_MMQ_TYPES",
    "GGML_UNQUANTIZED",
    "row_bytes",
    "dequant_q4_0",
    "dequant_q6_k",
    "dequantize",
]
