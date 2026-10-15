"""GGML block-quant dequantization in pure torch (the formats this repo's GGUF
checkpoints use: Q4_0, Q6_K, plus trivial F32/F16/BF16).

This is the *reference / CPU* path, NOT the engine's hot path: GGUF weights stay
packed and are dequantized inside the borrowed ggml CUDA kernels (see
``freetoken.kernel.gguf``). These routines are used only to (a) materialize the few
dense F32/F16 tensors at load (norms, scales, router) via :func:`dequantize`, and
(b) cross-check the CUDA kernels in tests. The ``BLOCK_SHAPE`` table and
:func:`row_bytes` are the type metadata the packed (kernel) path also relies on.

Each ``dequant_*`` takes the raw little-endian bytes as a ``uint8`` tensor whose
final axis spans whole blocks, and returns the values in *storage order* (ggml's
fastest axis first); the caller reshapes to the torch shape (``dims[::-1]``). The
math mirrors ``ggml-quants.c``.
"""

from __future__ import annotations

import math

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q6_K = 14
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q4_K: (256, 144),
    GGML_Q6_K: (256, 210),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q8_0: "Q8_0",
    GGML_Q4_K: "Q4_K",
    GGML_Q6_K: "Q6_K",
}


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


def padded_row_bytes(numel: int, ggml_type: int) -> tuple[int, int]:
    """``(row_bytes, padded_numel)``: the row padded with whole blocks to a 16-byte multiple.

    The offload cache moves expert rows with 16-byte-granular kernels, but a ggml K-quant row
    is only as aligned as its blocks allow (Q6_K: 210 bytes per 256 elements, so a 1536-wide
    row is 1260 bytes; Q4_K: 144 bytes per 256, already a multiple of 16). Padding with whole
    super-blocks keeps the row 16-byte aligned; an all-zero weight block dequantizes to exactly
    zero and the matching zero-padded activation contributes exactly zero, so the dot products
    are unchanged.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    if numel % block:
        raise ValueError(
            f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
        )
    blocks = numel // block
    step = 16 // math.gcd(type_size, 16)  # blocks per 16-byte row segment
    blocks = -(-blocks // step) * step
    return blocks * type_size, blocks * block


def row_numel(row_bytes: int, ggml_type: int) -> int:
    """Element width of a packed row of ``row_bytes`` bytes (the inverse of :func:`padded_row_bytes`)."""
    block, type_size = BLOCK_SHAPE[ggml_type]
    return row_bytes // type_size * block


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


def dequant_q4_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_K: 256-elem super-block = fp16 ``d`` + fp16 ``dmin`` + 12B packed 6-bit scales/mins
    + 128B nibbles; ``w = d*sc*q - dmin*m``. Vectorization of ggml's ``get_scale_min_k4``
    block loop."""

    raw = raw.reshape(-1, 144)
    n = raw.shape[0]
    d = _f16_scales(raw, 0, 2)  # [n,1]
    dmin = _f16_scales(raw, 2, 4)  # [n,1]
    packed_scales = raw[:, 4:16].to(torch.int32)  # [n,12] raw bytes
    low = packed_scales & 0x3F  # [n,12]: bytes 0-3 scales, 4-7 mins, 8-11 the high halves
    qs = raw[:, 16:144].to(torch.int32)  # [n,128]

    # 8 sub-blocks of 32 elements: scales 0-3 / mins 0-3 in the low 6 bits of bytes 0-7,
    # scales 4-7 / mins 4-7 in the low 4 bits of bytes 8-11 plus the top 2 bits of 0-7
    # (ggml get_scale_min_k4).
    sc = torch.empty((n, 8), dtype=torch.int32, device=raw.device)
    mn = torch.empty((n, 8), dtype=torch.int32, device=raw.device)
    sc[:, :4] = low[:, :4]
    mn[:, :4] = low[:, 4:8]
    hi = packed_scales[:, 8:12]  # [n,4]
    sc[:, 4:] = (hi & 0x0F) | ((packed_scales[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (hi >> 4) | ((packed_scales[:, 4:8] >> 6) << 4)

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    for half in range(4):  # 64 elements per iteration: 32 low nibbles then 32 high nibbles
        q = qs[:, half * 32 : (half + 1) * 32]
        d1 = d * sc[:, 2 * half].unsqueeze(1).to(torch.float32)
        m1 = dmin * mn[:, 2 * half].unsqueeze(1).to(torch.float32)
        d2 = d * sc[:, 2 * half + 1].unsqueeze(1).to(torch.float32)
        m2 = dmin * mn[:, 2 * half + 1].unsqueeze(1).to(torch.float32)
        base = half * 64
        y[:, base : base + 32] = d1 * (q & 0x0F).to(torch.float32) - m1
        y[:, base + 32 : base + 64] = d2 * (q >> 4).to(torch.float32) - m2
    return y.reshape(-1).to(out_dtype)


def dequant_q8_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q8_0: per 32-elem block = fp16 scale ``d`` + 32 int8 values; ``w = d*q``."""
    raw = raw.reshape(-1, 34)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    q = raw[:, 2:34].contiguous().view(torch.int8).to(torch.float32)  # [N,32]
    return (q * d).reshape(-1).to(out_dtype)


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
    GGML_Q8_0: dequant_q8_0,
    GGML_Q4_K: dequant_q4_k,
    GGML_Q6_K: dequant_q6_k,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) of any supported ggml type to flat ``out_dtype``."""
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} not implemented"
        )
    return fn(raw, out_dtype)


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_BF16",
    "GGML_Q4_0",
    "GGML_Q8_0",
    "GGML_Q4_K",
    "GGML_Q6_K",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "row_bytes",
    "padded_row_bytes",
    "row_numel",
    "dequant_q4_0",
    "dequant_q8_0",
    "dequant_q4_k",
    "dequant_q6_k",
    "dequantize",
]
