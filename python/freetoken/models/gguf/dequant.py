"""GGML block-quant dequantization in pure torch (the formats this repo's GGUF
checkpoints use: Q4_0, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, plus trivial F32/F16/BF16).

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

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_IQ4_NL = 20
GGML_IQ4_XS = 23
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ4_XS: (256, 136),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q8_0: "Q8_0",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ4_NL: "IQ4_NL",
    GGML_IQ4_XS: "IQ4_XS",
}

# ggml's 16-entry int8 codebook for the non-linear 4-bit quants (IQ4_NL/IQ4_XS).
IQ4NL_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


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


_IQ4NL_LUT: dict[torch.device, torch.Tensor] = {}


def _iq4nl_lut(device: torch.device) -> torch.Tensor:
    """The IQ4 codebook as an fp32 tensor on ``device`` (created once per device)."""
    lut = _IQ4NL_LUT.get(device)
    if lut is None:
        lut = torch.tensor(IQ4NL_KVALUES, dtype=torch.float32, device=device)
        _IQ4NL_LUT[device] = lut
    return lut


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


def dequant_q8_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q8_0: per 32-elem block = fp16 scale ``d`` + 32 int8 values; ``w = d*q``."""
    raw = raw.reshape(-1, 34)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    q = raw[:, 2:34].view(torch.int8).to(torch.float32)  # [N,32]
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


def dequant_iq4_nl(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """IQ4_NL: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; each nibble
    indexes a 16-entry int8 codebook, so ``w = d * kvalues_iq4nl[q]``.

    Same byte/nibble packing as Q4_0 (element ``j`` low, ``j+16`` high), only the
    lookup replaces the linear ``q - 8``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    values = _iq4nl_lut(raw.device)
    q = torch.cat([values[(qs & 0x0F).long()], values[(qs >> 4).long()]], dim=1)  # [N,32]
    return (q * d).reshape(-1).to(out_dtype)


def dequant_iq4_xs(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """IQ4_XS: 256-elem super-block = fp16 ``d`` + 16-bit high scale bits + 4B low
    scale nibbles + 128B packed nibbles. The 6-bit ``ls`` per 32-elem sub-block gives
    ``dl = d*(ls-32)``, then the same codebook lookup as IQ4_NL.
    """
    raw = raw.reshape(-1, 136)
    n = raw.shape[0]
    d = _f16_scales(raw, 0, 2)  # [n,1]
    scales_h = raw[:, 2:4].contiguous().view(torch.uint16).to(torch.int64)  # [n,1]
    scales_l = raw[:, 4:8].to(torch.int64)  # [n,4]
    qs = raw[:, 8:136]  # [n,128]
    ib = torch.arange(8, device=raw.device)  # [8], one per 32-elem sub-block
    low = (scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0x0F  # [n,8]
    high = (scales_h >> (2 * ib)) & 3  # [n,8]
    dl = d * ((low | (high << 4)) - 32).to(torch.float32)  # [n,8]
    q = qs.reshape(n, 8, 16)  # [n,8 sub-blocks,16 bytes]
    values = _iq4nl_lut(raw.device)
    y = torch.cat([values[(q & 0x0F).long()], values[(q >> 4).long()]], dim=2)  # [n,8,32]
    return (y * dl.unsqueeze(2)).reshape(-1).to(out_dtype)


def dequant_q5_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_K: 256-elem super-block = fp16 ``d`` + fp16 ``dmin`` + 12B packed 6-bit
    sub-block scales/mins + 32B 5th bits + 128B low nibbles.

    ``w = d*sc*q - dmin*m`` with the 5-bit ``q`` per element; 8 sub-blocks of 32, each
    with a 6-bit scale and min (ggml's ``get_scale_min_k4`` packing).
    """
    raw = raw.reshape(-1, 176)
    n = raw.shape[0]
    d = _f16_scales(raw, 0, 2)  # [n,1]
    dmin = _f16_scales(raw, 2, 4)  # [n,1]
    scales = raw[:, 4:16].to(torch.int64)  # [n,12]
    qh = raw[:, 16:48].to(torch.int64)  # [n,32]
    qs = raw[:, 48:176].to(torch.int64)  # [n,128]

    sc = torch.empty((n, 8), dtype=torch.int64, device=raw.device)
    mn = torch.empty((n, 8), dtype=torch.int64, device=raw.device)
    sc[:, 0:4] = scales[:, 0:4] & 0x3F
    sc[:, 4:8] = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
    mn[:, 0:4] = scales[:, 4:8] & 0x3F
    mn[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)
    d1 = d * sc.to(torch.float32)  # [n,8]
    m1 = dmin * mn.to(torch.float32)  # [n,8]

    ql = qs.reshape(n, 4, 32)  # 4 groups of 64 elements
    q = torch.stack([ql & 0x0F, (ql >> 4) & 0x0F], dim=2).reshape(n, 8, 32)  # [n,8,32]
    ib = torch.arange(8, device=raw.device).view(1, 8, 1)
    q = q + (((qh.view(n, 1, 32) >> ib) & 1) << 4)  # 5th bit per sub-block
    y = d1.unsqueeze(2) * q - m1.unsqueeze(2)
    return y.reshape(-1).to(out_dtype)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q8_0: dequant_q8_0,
    GGML_Q5_K: dequant_q5_k,
    GGML_Q6_K: dequant_q6_k,
    GGML_IQ4_NL: dequant_iq4_nl,
    GGML_IQ4_XS: dequant_iq4_xs,
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
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_IQ4_NL",
    "GGML_IQ4_XS",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "IQ4NL_KVALUES",
    "row_bytes",
    "dequant_q4_0",
    "dequant_q8_0",
    "dequant_q5_k",
    "dequant_q6_k",
    "dequant_iq4_nl",
    "dequant_iq4_xs",
    "dequantize",
]
