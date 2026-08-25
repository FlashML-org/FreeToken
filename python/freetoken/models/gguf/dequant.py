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

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q5_K = 13
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
    GGML_Q5_K: "Q5_K",
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


def quantize_q8_0(w: torch.Tensor) -> torch.Tensor:
    """Quantize dense rows to packed Q8_0 blocks (``half d`` + 32 int8).

    ``w``'s last dim must be a multiple of 32 (the Q8_0 block); returns the packed
    ``[..., n/32*34]`` uint8 layout the ggml Q8_0 kernels read. Used to re-quantize
    K-quant expert banks to a uniform 8-bit type (Q8_0 >= Q5_K/Q6_K precision, so no
    quality loss) when the offload cache needs a single per-bank format.
    """
    n = w.shape[-1]
    assert n % 32 == 0, f"Q8_0 quantize needs last dim % 32 == 0, got {n}"
    wq = w.float().view(*w.shape[:-1], n // 32, 32)
    d = wq.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / 127.0
    q = torch.round(wq / d).to(torch.int8)
    dh = d.to(torch.float16).view(torch.uint8)  # [..., n//32, 2]
    packed = torch.cat([dh, q.view(torch.uint8)], dim=-1)  # [..., n//32, 34]
    return packed.reshape(*w.shape[:-1], (n // 32) * 34).contiguous()


def dequant_q5_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_K: 256-elem super-block = half2 dm (dall, dmin), 12B 6-bit scale/min, 32B
    qh high-bits, 128B qs low nibbles. Mirrors ggml's dequantize_block_q5_K."""
    raw = raw.reshape(-1, 176)
    n = raw.shape[0]
    dm = raw[:, 0:4].view(torch.float16).to(torch.float32)  # [n,2]
    dall, dmin = dm[:, 0], dm[:, 1]
    scales = raw[:, 4:16]
    qh = raw[:, 16:48].to(torch.int32)
    qs = raw[:, 48:176].to(torch.int32)

    def _sm(j):
        if j < 4:
            d = scales[:, j] & 63
            m = scales[:, j + 4] & 63
        else:
            d = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
        return d.to(torch.float32), m.to(torch.float32)

    y = torch.zeros(n, 256, dtype=torch.float32, device=raw.device)
    for il in range(4):
        s0, m0 = _sm(2 * il)
        s1, m1 = _sm(2 * il + 1)
        d0, M0 = dall * s0, dmin * m0
        d1, M1 = dall * s1, dmin * m1
        bit0 = 1 << (2 * il)
        bit1 = bit0 << 1
        ql = qs[:, 32 * il:32 * il + 32]
        ql0, ql1 = ql[:, 0::2], ql[:, 1::2]
        h0, h1 = qh[:, 0::2], qh[:, 1::2]
        v0 = (ql0 & 0xF) + ((h0 & bit0) != 0).to(torch.float32) * 16
        v1 = (ql1 & 0xF) + ((h1 & bit0) != 0).to(torch.float32) * 16
        even = torch.stack([v0, v1], dim=-1).reshape(n, 32)  # [v0[0],v1[0],v0[1],...]
        y[:, 64 * il:64 * il + 32] = even * d0.unsqueeze(1) - M0.unsqueeze(1)
        w0 = (ql0 >> 4) + ((h0 & bit1) != 0).to(torch.float32) * 16
        w1 = (ql1 >> 4) + ((h1 & bit1) != 0).to(torch.float32) * 16
        odd = torch.stack([w0, w1], dim=-1).reshape(n, 32)
        y[:, 64 * il + 32:64 * il + 64] = odd * d1.unsqueeze(1) - M1.unsqueeze(1)
    return y.reshape(-1).to(out_dtype)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q5_K: dequant_q5_k,
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
    "dequant_q4_0",
    "dequant_q5_k",
    "dequant_q6_k",
    "quantize_q8_0",
    "dequantize",
]
