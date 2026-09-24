"""Pure-torch transcriptions of the reference tilelang kernels (inference/kernel.py), for the test oracle.

Same numerics as the tilelang code: fp32 math, power-of-two (ue8m0) or e4m3 scales rounded the way
the kernels round them, the e2m1 grid with the hardware's round-to-nearest-even, and the fp8/fp4
in-place round trips that write the dequantized value back in the input dtype. Quantized GEMMs
dequantize independently in torch and round each projection's output to bf16.
"""

from __future__ import annotations

import torch

_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """``fast_round_scale``: 2 ** ceil(log2(x)) for x > 0."""
    return torch.exp2(torch.ceil(torch.log2(x)))


def _round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the signed e2m1 grid (the ``float4_e2m1fn`` cast)."""
    a = x.abs()
    grid = _FP4_GRID.to(x.device)
    hi = torch.searchsorted(grid, a.reshape(-1).contiguous(), right=True).reshape(a.shape).clamp(1, 7)
    below, above = grid[hi - 1], grid[hi]
    mid = (below + above) / 2
    even_above = torch.isin(above, torch.tensor([0.0, 1.0, 2.0, 4.0], device=x.device))
    r = torch.where((a > mid) | ((a == mid) & even_above), above, below)
    r = torch.where(a >= 6.0, torch.full_like(r, 6.0), r)
    return torch.copysign(r, x)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None, scale_dtype=torch.float32, inplace: bool = False):
    """Block-wise fp8 quantization; ``inplace=True`` writes the quant+dequant round trip back into ``x``."""
    n = x.size(-1)
    assert n % block_size == 0
    g = x.float().unflatten(-1, (-1, block_size))
    amax = g.abs().amax(dim=-1).clamp_min(1e-4)
    s = _pow2_ceil(amax / 448.0) if scale_fmt is not None else amax / 448.0
    q = (g / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise fp4 with ue8m0 (pow2) or e4m3 scales; ``inplace=True`` writes the round trip back."""
    n = x.size(-1)
    assert n % block_size == 0
    g = x.float().unflatten(-1, (-1, block_size))
    amax = g.abs().amax(dim=-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2**-9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2**-126)
        s = _pow2_ceil(amax / 6.0)
    q = _round_fp4((g / s.unsqueeze(-1)).clamp(-6.0, 6.0))
    if inplace:
        x.copy_((q * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.to(scale_dtype)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """``C = (A * s_a) @ (B * s_b)^T`` over dequantized operands."""
    k = a.size(-1)
    a_deq = (a.float().unflatten(-1, (-1, block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    n = b.size(0)
    bs = b_s.float().repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    return (a_deq @ (b.float() * bs).t()).bfloat16()


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float8_e8m0fnu, act_block_size=32):
    packed = b.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=b.device)
    weight = grid[codes] * b_s.float().repeat_interleave(32, -1)
    x = (a.float().unflatten(-1, (-1, act_block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    return (x @ weight.t()).bfloat16()


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Per (batch, query): softmax over the gathered ``topk_idxs`` (-1 = absent) plus the sink null key.

    ``q [b, m, h, d]``, ``kv [b, n, d]``, ``topk_idxs [b, m, topk]`` -> ``o [b, m, h, d]`` in q's dtype.
    A row with no valid index yields zeros (the finite ``-1e30`` running max of the kernel)."""
    b, m, h, d = q.shape
    valid = topk_idxs >= 0
    gathered = kv[torch.arange(b, device=q.device)[:, None, None], topk_idxs.clamp_min(0).long()]  # [b, m, topk, d]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    mx = scores.amax(dim=-1).clamp_min(-1e30)
    p = torch.exp(scores - mx.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(attn_sink.float()[None, None, :] - mx)
    o = torch.einsum("bmht,bmtd->bmhd", p, gathered.float()) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    hc = hc_mult
    mixes, sc, base = mixes.float(), hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * sc[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * sc[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
