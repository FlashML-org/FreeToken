"""Dense fp32 references for the Inkling 2 kernels, written from transformers #51050 (head 6173a2f679,
``modeling_inkling2_loop.py``) with no FreeToken import, so they also run next to the HF classes."""

from __future__ import annotations

import math

import torch


def attention(q, k, v, rel, table, sinks, q_pos, k_pos, sm_scale, sliding_window):
    """One request: q [Tq, H, D], k / v [Tk, KVH, D], rel [Tq, H, R], table [R, E], sinks [Tq, H] -> [Tq, H, D] fp32.

    ``eager_attention_forward`` with ``Inkling2LoopRelativePositionBias``: the bias of distance ``d`` is
    ``rel @ table`` gathered at ``d`` for ``0 <= d < E``, the sink is one more logit with a zero value."""
    H, kvh = q.shape[1], k.shape[1]
    k = k.float().repeat_interleave(H // kvh, dim=1)
    v = v.float().repeat_interleave(H // kvh, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), k) * sm_scale
    dist = q_pos[:, None] - k_pos[None, :]
    E = table.shape[1]
    prof = (rel.float() @ table.float()).permute(1, 0, 2)  # [H, Tq, E]
    bias = prof.gather(2, dist.clamp(0, E - 1).expand(H, -1, -1))
    s = s + torch.where((dist >= 0) & (dist < E), bias, 0.0)
    valid = dist >= 0
    if sliding_window is not None:
        valid = valid & (dist < sliding_window)
    s = s.masked_fill(~valid, float("-inf"))
    s = torch.cat([s, sinks.float().t().unsqueeze(-1)], dim=-1)
    p = torch.softmax(s, dim=-1)[..., :-1]
    return torch.einsum("hqk,khd->qhd", p, v)


def short_conv(x, conv1d_weight=None, static_weight=None, dynamic=None):
    """Whole sequence from a zero window: x [T, C]. Either ``conv1d_weight`` [C, 1, K] (``Inkling2LoopShortConvolution``,
    out / sqrt(K + 1)) or ``static_weight`` [K, C] plus ``dynamic`` [T, heads, K] per-token taps
    (``Inkling2LoopDynamicShortConvolution``, out / sqrt(2K + 1)). fp32 out."""
    T, C = x.shape
    xf = x.float()
    if conv1d_weight is not None:
        K = conv1d_weight.shape[-1]
        out = torch.nn.functional.conv1d(xf.t()[None], conv1d_weight.float(), padding=K - 1, groups=C)[0, :, :T].t()
        return (out + xf) / math.sqrt(K + 1)
    K = static_weight.shape[0]
    heads = dynamic.shape[1]
    weights = static_weight.float().t().reshape(heads, -1, K)[None] + dynamic.float()[:, :, None, :]  # [T, heads, C/heads, K]
    ctx = torch.nn.functional.pad(xf.t(), (K - 1, 0))
    windows = ctx.unfold(-1, K, 1).flip(-1).permute(1, 0, 2).reshape(T, heads, -1, K)
    out = (windows * weights).sum(-1).reshape(T, C)
    return (out + xf) / math.sqrt(2 * K + 1)


def softcap_swiglu(gate, up, gate_cap, up_cap):
    """``Inkling2LoopSoftcappedSwiGLU``: fp32 math, the sigmoid sees the raw gate."""
    gate, up = gate.float(), up.float()
    return torch.tanh(gate / gate_cap) * gate_cap * torch.sigmoid(gate) * torch.tanh(up / up_cap) * up_cap


def route(x, weight, bias, top_k, scale):
    """``Inkling2LoopTopKRouter``: returns (weights [T, k], ids [T, k]) sorted by id for comparison."""
    scores = torch.sigmoid(torch.nn.functional.linear(x.float(), weight.float()))
    ids = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    w = scores.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * scale
    order = ids.argsort(-1)
    return w.gather(1, order), ids.gather(1, order)
