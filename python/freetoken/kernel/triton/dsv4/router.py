"""Fused MoE gate selection (DeepSeek-V4 / V4.1 ``Gate``): score activation, selection bias, top-k,
weight normalization and route scale in one program per token.

    s       = act(scores)                       act: softmax | sigmoid | sqrt(softplus)
    ids     = topk(s + bias)                    selection is biased, the weights are not
    w       = s[ids] [/ (sum + 1e-20)] * route_scale

One launch replaces the eight or so elementwise / top-k / gather launches of the torch expression --
at decode each of those moves a few hundred floats and costs its launch floor. Ties in ``s + bias``
go to the lowest expert id (torch.topk leaves them unspecified).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_SCORE = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}


@triton.jit
def _router_select_kernel(
    scores_ptr, bias_ptr, alternate_bias_ptr, alternate_mask_ptr, w_ptr, id_ptr, E, route_scale,
    stride_s, stride_w, stride_id,
    SCORE: tl.constexpr, TOPK: tl.constexpr, BLOCK_K: tl.constexpr, NORMALIZE: tl.constexpr, BLOCK_E: tl.constexpr, ALTERNATE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    s = tl.load(scores_ptr + t * stride_s + offs, mask=mask, other=float("-inf")).to(tl.float32)
    if SCORE == 0:
        s = tl.exp(s - tl.max(s))
        s = s / tl.sum(tl.where(mask, s, 0.0))
    elif SCORE == 1:
        s = tl.sigmoid(s)
    else:
        s = tl.sqrt(tl.where(s > 20.0, s, libdevice.log1p(tl.exp(s))))  # torch softplus (threshold 20)
    if ALTERNATE:
        bias_ptr = tl.where(tl.load(alternate_mask_ptr + t), alternate_bias_ptr, bias_ptr)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    key = tl.where(mask, s + bias, float("-inf"))
    slots = tl.arange(0, BLOCK_K)
    picked_w = tl.zeros((BLOCK_K,), dtype=tl.float32)
    picked_id = tl.zeros((BLOCK_K,), dtype=tl.int32)
    for k in tl.static_range(TOPK):
        best = tl.max(key)
        idx = tl.min(tl.where(key == best, offs, BLOCK_E))  # lowest id among ties
        w = tl.sum(tl.where(offs == idx, s, 0.0))
        picked_w = tl.where(slots == k, w, picked_w)
        picked_id = tl.where(slots == k, idx.to(tl.int32), picked_id)
        key = tl.where(offs == idx, float("-inf"), key)
    if NORMALIZE:
        picked_w = picked_w / (tl.sum(picked_w) + 1e-20)
    picked_w = picked_w * route_scale
    tl.store(w_ptr + t * stride_w + slots, picked_w, mask=slots < TOPK)
    tl.store(id_ptr + t * stride_id + slots, picked_id, mask=slots < TOPK)


def router_select(
    scores: torch.Tensor, bias: torch.Tensor, topk: int, *, score_func: str, normalize: bool, route_scale: float,
    alternate_bias: torch.Tensor | None = None, alternate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scores [T, E]`` fp32 (pre-activation gate logits), ``bias [E]`` -> ``(weights [T, topk] fp32,
    ids [T, topk] int32)`` in selection order (best first)."""
    T, E = scores.shape
    assert scores.dtype == torch.float32 and scores.stride(1) == 1, (scores.dtype, scores.stride())
    weights = torch.empty((T, topk), dtype=torch.float32, device=scores.device)
    ids = torch.empty((T, topk), dtype=torch.int32, device=scores.device)
    if T == 0:
        return weights, ids
    _router_select_kernel[(T,)](
        scores, bias, alternate_bias if alternate_bias is not None else bias,
        alternate_mask if alternate_mask is not None else scores, weights, ids, E, float(route_scale),
        scores.stride(0), weights.stride(0), ids.stride(0),
        SCORE=_SCORE[score_func], TOPK=topk, BLOCK_K=triton.next_power_of_2(topk), NORMALIZE=normalize and topk > 1,
        BLOCK_E=triton.next_power_of_2(E), ALTERNATE=alternate_mask is not None, num_warps=4,
    )
    return weights, ids


__all__ = ["router_select"]
