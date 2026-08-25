"""NVFP4 -> MXFP4 (gpt-oss/FreeToken ``mxfp4_triton``) weight converter.

ModelOpt NVFP4 (the format stored in FreeToken's native ``nvfp4`` banks) packs e2m1
codes with a *fp8-e4m3* per-16 block scale and a per-output-row fp16 *global* scale.
MXFP4 (FreeToken's ``mxfp4_triton`` banks) packs e2m1 codes with an *e8m0* per-32
block scale and a per-block bias, and is the native format of the gpt-oss family --
the AMD-supported quant matrix alongside BF16/GGUF.

This module converts a checkpoint's NVFP4 expert weights to MXFP4 *on load* (once,
cached per model), so a checkpoint that only ships NVFP4 can still run on AMD via the
portable MXFP4 path (the converter runs on the host, not on the GPU). The two formats
share the e2m1 code-packing (2 codes per byte, low nibble first), so only the scale
granularity (16 -> 32) and scale format (e4m3 -> e8m0) change.

Layouts handled here (per projection/expert, ``N`` = output rows, ``K`` = input cols):

* input (NVFP4, native ModelOpt rows): ``packed [N, K//2]`` uint8, ``scale [N, K//16]``
  fp8-e4m3, ``global [N]`` fp16.
* output (MXFP4, ``mxfp4_triton`` bank layout): ``blocks_t [N, K//2]`` uint8,
  ``scales_t [N, K//32]`` uint8 e8m0. (FreeToken's MXFP4 stores the projection
  transposed, N innermost, matching gpt-oss -- the converter emits that shape.)

All numeric work is done in numpy so the core is unit-testable without a GPU/torch
runtime; the public entry converts torch tensors to/from numpy on the host.
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is a hard dep
    _np = None

__all__ = [
    "convert_nvfp4_to_mxfp4",
    "dequantize_nvfp4_block",
    "fp4_e2m1_table",
    "e8m0_scale_and_codes",
]

# ---------------------------------------------------------------------------
# e2m1 (fp4) code -> value table, and e8m0 (block scale) encode.
#
# e2m1: 1 sign + 2 exponent + 1 mantissa. With exponent bias 1 the finite set is:
#   code  value   code  value
#     0    0.0      8   -0.0
#     1    0.5      9   -0.5
#     2    1.0     10   -1.0
#     3    1.5     11   -1.5
#     4    2.0     12   -2.0
#     5    3.0     13   -3.0
#     6    4.0     14   -4.0
#     7    6.0     15   -6.0
# ---------------------------------------------------------------------------
_FP4_CODES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_FP4_TABLE = _np.asarray(_FP4_CODES, dtype=_np.float32)
# Magnitudes sorted ascending for the nearest-code search.
_FP4_SORT = _np.asarray(sorted(abs(v) for v in _FP4_CODES[1:8]), dtype=_np.float32)
_FP4_SORT_SIGN = _np.asarray([1.0 if i < 4 else -1.0 for i in range(len(_FP4_SORT))],
                             dtype=_np.float32)


def fp4_e2m1_table() -> Sequence[float]:
    """The 16 e2m1 values keyed by 4-bit code (index == code)."""
    return list(_FP4_CODES)


def _nearest_e2m1_codes(values: _np.ndarray) -> _np.ndarray:
    """Nearest e2m1 *code* for each fp32 ``values`` (signed, including 0/NaN)."""
    a = _np.abs(values)
    diff = _np.abs(a[..., None] - _FP4_SORT)  # [..., 7]
    idx = diff.argmin(axis=-1)
    mag = _FP4_SORT[idx]
    neg = _np.signbit(values)
    code = (idx + 1).astype(_np.uint8)  # _FP4_SORT[i] == table[i+1]; positive codes 1..7
    out = _np.where(neg, code | 0x8, code)
    # Magnitudes below the smallest representable value (0.5) round to +0.
    return _np.where(mag < 0.25, 0, out)


def e8m0_scale_and_codes(values: _np.ndarray, block: int = 32) -> tuple[_np.ndarray, _np.ndarray]:
    """Return ``(scale_codes, fp4_codes)`` for ``values`` shaped ``[..., block]``: an
    e8m0 ``uint8`` scale per block (the smallest power-of-2 scale covering the block
    max-abs, in the MX ``2**(v-127)`` encoding) and the requantized 4-bit codes.

    ``scale_codes`` has shape ``values.shape[:-1]``; ``fp4_codes`` matches ``values``.
    """
    v = values.reshape(-1, block)
    amax = _np.max(_np.abs(v), axis=-1)
    # e2m1 max positive magnitude is 6.0; choose the smallest power-of-2 scale so the
    # block max-abs maps near the top of the e2m1 range (best precision):
    #   s = 2^ceil(log2(max_abs / 6.0)), stored as e8m0 code v with 2**(v-127) == s.
    amax_safe = _np.maximum(amax, 1e-38)
    exp = _np.ceil(_np.log2(amax_safe / 6.0)).astype(_np.float32)
    exp = _np.where(amax == 0.0, 0.0, exp)
    scale_codes = (127.0 + exp).astype(_np.uint8)  # v-127 == exp
    scale_v = (2.0 ** exp).astype(_np.float32)
    # Requantize values in the block by its scale, then nearest-e2m1-code.
    q = v / scale_v[:, None]
    codes = _fp4_quantize(q).astype(_np.uint8)
    return scale_codes.reshape(values.shape[:-1]), codes.reshape(values.shape)


def _fp4_quantize(values: _np.ndarray) -> _np.ndarray:
    a = _np.abs(values)
    diff = _np.abs(a[..., None] - _FP4_SORT)
    idx = diff.argmin(axis=-1)
    mag = _FP4_SORT[idx]
    out = _np.where(mag < 0.25, 0, (idx + 1).astype(_np.uint8))
    out = _np.where(_np.signbit(values), out | 8, out)
    return out


# ---------------------------------------------------------------------------
# NVFP4 -> MXFP4
# ---------------------------------------------------------------------------


def dequantize_nvfp4_block(
    packed: _np.ndarray,
    scale: _np.ndarray,
    global_scale: _np.ndarray,
    *,
    block: int = 16,
) -> _np.ndarray:
    """Dequantize one native NVFP4 projection back to fp32.

    ``packed [N, K//2]`` uint8 (e2m1 pairs), ``scale [N, K//16]`` fp32 (already
    converted from fp8-e4m3), ``global_scale [N]`` fp32. Returns ``[N, K]`` fp32.
    """
    N, K2 = packed.shape
    K = K2 * 2
    lo = (packed & 0x0F).astype(_np.uint8)
    hi = (packed >> 4).astype(_np.uint8)
    codes = _np.stack([lo, hi], axis=-1).reshape(N, K)  # [N, K]
    vals = _FP4_TABLE[codes.astype(_np.int64)]  # [N, K]
    # Per-16 block scale broadcast over K.
    bs = _np.repeat(scale, block, axis=-1)  # [N, K]
    return (vals * bs).astype(_np.float32) * global_scale[:, None].astype(_np.float32)


def _pack_codes(codes: _np.ndarray) -> _np.ndarray:
    """Pack ``[N, K]`` uint8 4-bit codes -> ``[N, K//2]`` uint8 (low nibble first)."""
    N, K = codes.shape
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    return (even | (odd << 4)).astype(_np.uint8)


def convert_nvfp4_to_mxfp4(
    packed,
    scale,
    global_scale,
    *,
    axis: int = -1,
    block: int = 32,
):
    """Convert one projection's native NVFP4 expert weights to the MXFP4 layout.

    Args:
        packed: ``[..., K//2]`` uint8 e2m1 pairs (low nibble = first code).
        scale: ``[..., K//16]`` fp8-e4m3 block scale (fp32/fp16 input accepted).
        global_scale: ``[...]`` per-output-row fp16 global scale.
        axis: the K (contraction) axis along which blocks are grouped.

    Returns ``(mxfp4_packed [..., K//2] uint8, mxfp4_scales [..., K//32] uint8 e8m0)``
    matching the ``mxfp4_triton`` per-expert bank shape (K innermost).
    """
    np = _np
    packed = np.asarray(packed)
    scale = np.asarray(scale, dtype=np.float32)
    global_scale = np.asarray(global_scale, dtype=np.float32)

    if axis not in (-1, packed.ndim - 1):
        raise NotImplementedError("converter requires the K axis to be innermost")

    # Dequantize NVFP4 to fp32, move K to the last axis.
    K2 = packed.shape[-1]
    K = K2 * 2
    codes = np.stack([packed & 0x0F, (packed >> 4)], axis=-1).reshape(
        *packed.shape[:-1], K
    )
    vals = _FP4_TABLE[codes.astype(np.int64)]
    bs = np.repeat(scale, 16, axis=-1)
    f32 = (vals * bs).astype(np.float32) * global_scale[..., None].astype(np.float32)

    # Requantize to per-`block` e8m0 + e2m1.
    flat = f32.reshape(-1, K)
    # pad to a multiple of block for the reshape (K is a multiple of 32 in practice)
    n_blocks = K // block
    flat_b = flat[:, : n_blocks * block].reshape(-1, block)
    scale_codes, mxfp4_codes = e8m0_scale_and_codes(flat_b, block=block)
    out_codes = mxfp4_codes.reshape(flat.shape[0], n_blocks * block)

    mxfp4_packed = _pack_codes(out_codes)  # [..., K//2]
    mxfp4_scales = scale_codes.reshape(*packed.shape[:-1], n_blocks)
    return mxfp4_packed, mxfp4_scales


# ---------------------------------------------------------------------------
# torch-tensor entrypoint
# ---------------------------------------------------------------------------


def _to_numpy(t: object) -> _np.ndarray:
    if _np is not None and isinstance(t, _np.ndarray):
        return t
    try:
        return t.detach().cpu().numpy()
    except Exception as exc:  # pragma: no cover
        raise TypeError(
            f"convert_nvfp4_to_mxfp4 expects torch tensors or numpy arrays, got {type(t)!r}"
        ) from exc


def convert_torch_nvfp4_to_mxfp4(
    packed,
    scale,
    global_scale,
    *,
    block: int = 32,
):
    """Torch-tensor variant returning ``(mxfp4_packed, mxfp4_scales)`` torch tensors on
    the same device as ``packed`` (host converter: input tensors are pulled to CPU and
    the results copied back). Used at load time, cached per model."""
    import torch

    device = packed.device
    dtype = packed.dtype
    p, s = convert_nvfp4_to_mxfp4(
        _to_numpy(packed), _to_numpy(scale), _to_numpy(global_scale), block=block
    )
    return torch.from_numpy(p).to(device=device, dtype=dtype), torch.from_numpy(s).to(device=device)
