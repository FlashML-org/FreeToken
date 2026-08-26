"""Pure-torch sampling fallback (CPU / no-triton, no-flashinfer path).

FreeToken's ``engine/sample.py`` pulls its sampling kernels from either
``flashinfer.sampling`` or ``freetoken.kernel.triton.sampling`` — both require a
CUDA GPU. This module is a functionally-equivalent pure-torch implementation of
the four ops ``sample_impl`` actually uses, so the cpu-device branch can draw
tokens without any GPU kernel package.

Contract mirrors the triton/flashinfer entry points:
  - softmax(logits, temperature=None, enable_pdl=None) -> probs
  - sampling_from_probs(probs, **kw) -> token ids (greedy/argmax draw)
  - top_k_sampling_from_probs(probs, top_k, **kw) -> token ids
  - top_p_sampling_from_probs(probs, top_p, **kw) -> token ids

The extra kwargs (indices, deterministic, generator, check_nan, seed, offset,
return_valid) are accepted for signature compatibility and ignored apart from
``return_valid`` (we return the 2-tuple when requested).
"""

from __future__ import annotations

import torch


def softmax(logits: torch.Tensor, temperature=None, enable_pdl=None) -> torch.Tensor:
    logits = logits.float()
    if temperature is None:
        temperature = 1.0
    if isinstance(temperature, torch.Tensor):
        # per-row temperature; unsqueeze to broadcast over vocab
        inv = (1.0 / temperature.to(torch.float32).reshape(-1, 1)).to(logits.dtype)
        return torch.softmax(logits * inv, dim=-1)
    return torch.softmax(logits / float(temperature), dim=-1)


def _draw(probs: torch.Tensor, generator=None) -> torch.Tensor:
    # inverse-CDF draw via multinomial (categorical) sampling per row.
    probs = probs.clamp_min_(0.0)
    row_sums = probs.sum(dim=-1, keepdim=True)
    safe = probs / row_sums.clamp_min_(1e-12)
    out = torch.multinomial(safe, num_samples=1, generator=generator).reshape(-1)
    return out.to(torch.int32)


def sampling_from_probs(probs, indices=None, deterministic=True, generator=None,
                       check_nan=False, seed=None, offset=None, return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    out = _draw(src, generator=generator)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_sampling_from_probs(probs, top_k, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    if isinstance(top_k, torch.Tensor):
        top_k = top_k.to(src.device).reshape(-1)
    B, V = src.shape
    if isinstance(top_k, torch.Tensor):
        k = top_k.clamp(min=1)
        renorm = torch.empty_like(src)
        for b in range(B):
            kb = int(k[b].item())
            kb = min(kb, V)
            thr = torch.topk(src[b], kb).values[-1]
            mask = src[b] >= thr
            renorm[b] = src[b] * mask
    else:
        k = max(1, int(top_k))
        k = min(k, V)
        thr = torch.topk(src, k, dim=-1).values[..., -1:]  # [B, 1]
        renorm = src * (src >= thr)
    out = _draw(renorm, generator=generator)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    if isinstance(top_p, torch.Tensor):
        top_p = top_p.to(src.device).reshape(-1)
    B, V = src.shape
    renorm = torch.empty_like(src)
    for b in range(B):
        p = float(top_p[b].item()) if isinstance(top_p, torch.Tensor) else float(top_p)
        sorted_probs, sorted_idx = torch.sort(src[b], descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # keep the smallest set whose cumulative mass <= p (nucleus)
        keep_mask = cumulative <= p
        # always keep at least the top-1 token
        keep_mask[0] = True
        allowed = torch.zeros_like(src[b])
        allowed.scatter_(0, sorted_idx[keep_mask], sorted_probs[keep_mask])
        renorm[b] = allowed
    out = _draw(renorm, generator=generator)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None,
                                    filter_apply_order="top_k_first", deterministic=True,
                                    generator=None, check_nan=False, seed=None, offset=None,
                                    return_valid=False):
    # Combined nucleus + top-k filter, then draw. Apply top-k first (truncate the
    # vocab to the k most likely), then nucleus over that truncated distribution.
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    if isinstance(top_k, torch.Tensor):
        top_k = top_k.to(src.device).reshape(-1)
    if isinstance(top_p, torch.Tensor):
        top_p = top_p.to(src.device).reshape(-1)
    B, V = src.shape
    renorm = torch.empty_like(src)
    for b in range(B):
        k = int(top_k[b].item()) if isinstance(top_k, torch.Tensor) else int(top_k)
        k = max(1, min(k, V))
        p = float(top_p[b].item()) if isinstance(top_p, torch.Tensor) else float(top_p)
        # top-k truncation
        kth = torch.topk(src[b], k).values[-1]
        k_masked = src[b] * (src[b] >= kth)
        # nucleus over the truncated dist
        sorted_probs, sorted_idx = torch.sort(k_masked, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative <= p
        keep_mask[0] = True
        allowed = torch.zeros_like(k_masked)
        allowed.scatter_(0, sorted_idx[keep_mask], sorted_probs[keep_mask])
        renorm[b] = allowed
    out = _draw(renorm, generator=generator)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


__all__ = [
    "softmax",
    "sampling_from_probs",
    "top_k_sampling_from_probs",
    "top_p_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
]
