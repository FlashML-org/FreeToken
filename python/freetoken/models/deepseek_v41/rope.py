"""Rotary embeddings for DeepSeek-V4.1 (verbatim reference math, ``inference/model.py``).

Interleaved complex rotation on the last ``rope_head_dim`` channels, with YaRN when
``original_seq_len > 0``. Two regimes share two cached tables: window-only layers at the base
theta without YaRN, compressing layers at ``compress_rope_theta`` under YaRN. A compressed latent
stands for the first token of its group, so group ``j`` rotates at position ``j * ratio``; the
attention output is rotated back (``inverse=True``) because K == V share one rotated latent.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int, base: float, factor: float, beta_fast: int, beta_slow: int) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


@lru_cache(2)  # the model's two rope regimes; a 1M table is ~256 MB, so never one per layer
def get_freqs_cis(dim: int, seqlen: int, original_seq_len: int, base: float, factor: float, beta_fast: int, beta_slow: int, device: torch.device) -> torch.Tensor:
    return precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow).to(device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place rotation of ``x[..., :]`` (``[T, rd]`` or ``[T, H, rd]``) by per-token ``freqs_cis [T, rd//2]``."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.unsqueeze(1)
    x.copy_(torch.view_as_real(xc * freqs_cis).flatten(-2))
    return x


def apply_rotary_emb_decode(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Batched decode rotation with PER-ROW freqs: ``x [B, 1, ..., rd]``, ``freqs_cis [B, rd//2]``
    (the fused dsv4 triton kernel; bit-identical to ``apply_rotary_emb``)."""
    from freetoken.kernel.triton.dsv4.rope import rope_decode_inplace

    return rope_decode_inplace(x, freqs_cis, inverse)


__all__ = ["precompute_freqs_cis", "get_freqs_cis", "apply_rotary_emb", "apply_rotary_emb_decode"]
