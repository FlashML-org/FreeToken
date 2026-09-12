"""8-bit storage for the DeepSeek-V4 window / compressed KV pools.

DSV4's attention KV is one MLA latent row per token -- K and V are the same slab --
so unlike a separate-K-and-V pool there is a single buffer to quantize and a single
scale array beside it.

Storage is fp8-e4m3 held as ``uint8``, with one fp16 scale per :data:`BLOCK` elements
along ``head_dim``::

    1 byte + 2/32 = 1.0625 bytes per element,  against bf16's 2

The scale varies along ``head_dim``, which is the reduction dimension of ``q @ kv``, so
the attention kernel cannot fold it in after the dot: it dequantizes the gathered tile
before the dot. Storage is what this buys, not tensor-core throughput.

Why the precision cost is small here. The model already rounds this KV onto the e4m3
grid before it reaches the pool -- ``act_quant_fp8_inplace(kv[..., :-rope_head_dim], 64)``
runs immediately before every ``store_window`` and before the compressor's scatter -- so
for those dims this is closer to a storage change than a precision change. Re-blocking
from 64 to 32 does move values (a 32-block's own max-abs sets a different scale than the
64-block's did), but only within the grid the values already live on. The rope tail is
the exception: those dims are genuine bf16 and this rounds them for the first time.

Needs native fp8 (sm_89+). Below that a float8 pointer is illegal inside a triton
kernel, and DSV4 checkpoints are fp8 to begin with, so the flag is rejected at config
time rather than carrying an emulated decode nobody would run.
"""

from __future__ import annotations

import torch

# Elements per scale, along head_dim.
BLOCK = 32
SCALE_DTYPE = torch.float16
STORAGE_DTYPE = torch.float8_e4m3fn
# Max-abs of a block maps to this magnitude, e4m3's largest finite value.
MAX_MAG = 448.0

BYTES_PER_ELEMENT = 1.0 + SCALE_DTYPE.itemsize / BLOCK  # 1.0625


def scale_width(head_dim: int) -> int:
    """Scales per row. ``head_dim`` must be a multiple of :data:`BLOCK`."""
    if head_dim % BLOCK:
        raise ValueError(f"head_dim {head_dim} is not a multiple of the quant block {BLOCK}")
    return head_dim // BLOCK


def quantize_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for the store kernel: ``[rows, D]`` -> ``(fp8, fp16 scales)``.

    The scale is rounded to its stored precision *before* dividing, so the value written
    here and the value the attention kernel reads back are scaled by the identical number.
    """
    rows, d = x.shape
    nb = scale_width(d)
    xb = x.float().reshape(rows, nb, BLOCK)
    amax = xb.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / MAX_MAG, torch.ones_like(amax))
    scale = scale.to(SCALE_DTYPE).float()
    q = (xb / scale[..., None]).clamp(-MAX_MAG, MAX_MAG)
    return q.to(STORAGE_DTYPE).reshape(rows, d), scale.to(SCALE_DTYPE)


def dequantize_ref(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_ref`, in fp32."""
    rows, d = q.shape
    nb = scale_width(d)
    x = q.float().reshape(rows, nb, BLOCK)
    return (x * scale.float()[..., None]).reshape(rows, d)


__all__ = [
    "BLOCK", "SCALE_DTYPE", "STORAGE_DTYPE", "MAX_MAG", "BYTES_PER_ELEMENT",
    "scale_width", "quantize_ref", "dequantize_ref",
]
