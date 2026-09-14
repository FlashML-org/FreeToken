"""mHC -- Manifold-Constrained Hyper-Connections (GLM-5.3-Flash; arXiv 2512.24880).

The residual stream is widened to ``hc_mult`` (n) parallel streams. Around every
sublayer the streams are mixed by three learned, token-dependent maps computed
from ONE fp32 GEMM over the flattened streams (``fn [2n+n^2, n*hidden]``),
RMS-normalized over the full ``n*hidden`` vector and split into:

* ``pre_mix  [n]``     sigmoid gates: the sublayer input is ``sum_i pre_i * res_i``
* ``post_mix [n]``     sigmoid*mult gates: how much sublayer output enters each stream
* ``comb_mix [n, n]``  softmax + Sinkhorn-projected (approximately doubly-stochastic)
                       stream-mixing matrix -- the "manifold constraint": mixing
                       neither amplifies nor loses residual mass.

``mhc_post`` then rebuilds the streams: ``out_j = sum_i comb_ij * res_i + post_j * x``.

Semantics match vLLM's reference (``model_executor/kernels/mhc/torch.py``,
PR #53906) bit-for-bit in fp32; the per-layer weights are ``hc_{attn,ffn}_fn`` /
``_scale`` / ``_base`` from the checkpoint. This torch implementation is the
correctness baseline; the fused triton kernel (``kernel/triton/mhc.py``,
dispatched by ``mhc_fused_post_pre`` below) replaces it on CUDA and is
validated against this file.
"""

from __future__ import annotations

import torch


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """[T, hidden] -> [T, n, hidden] by replication (model entry)."""
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def hc_contract(x: torch.Tensor) -> torch.Tensor:
    """[T, n, hidden] -> [T, hidden] by averaging (model exit)."""
    return x.mean(dim=1)


def mhc_pre(
    residual: torch.Tensor,  # [T, n, hidden] bf16
    fn: torch.Tensor,  # [2n + n^2, n*hidden] fp32
    hc_scale: torch.Tensor,  # [3] fp32
    hc_base: torch.Tensor,  # [2n + n^2] fp32
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (post_mix [T, n, 1] fp32, comb_mix [T, n, n] fp32,
    layer_input [T, hidden] bf16)."""
    n, hidden = residual.shape[-2], residual.shape[-1]
    t = residual.shape[0]

    x = residual.reshape(t, n * hidden).to(torch.float32)
    mixes = x @ fn.t()
    # RMS over the FULL flattened n*hidden vector (not per stream).
    mixes = mixes * torch.rsqrt(x.square().sum(-1, keepdim=True) / (n * hidden) + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + hc_eps
    post_mix = torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
    post_mix = post_mix * post_mult

    comb = mixes[:, 2 * n :].view(t, n, n) * hc_scale[2] + hc_base[2 * n :].view(1, n, n)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    # Sinkhorn-Knopp projection toward the doubly-stochastic manifold: alternate
    # column / row normalization, ``sinkhorn_repeat`` column steps in total.
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    layer_input = (
        (pre_mix.unsqueeze(-1) * residual.to(torch.float32)).sum(dim=1).to(residual.dtype)
    )
    return post_mix.view(t, n, 1), comb, layer_input


def mhc_post(
    x: torch.Tensor,  # [T, hidden] sublayer output
    residual: torch.Tensor,  # [T, n, hidden]
    post_mix: torch.Tensor,  # [T, n, 1] fp32
    comb_mix: torch.Tensor,  # [T, n, n] fp32
) -> torch.Tensor:
    """out_j = sum_i comb_ij * residual_i + post_j * x; returns [T, n, hidden]."""
    mixed = torch.einsum(
        "tij,tih->tjh", comb_mix.to(torch.float32), residual.to(torch.float32)
    )
    post = post_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed + post).to(residual.dtype)


def mhc_fused_post_pre_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposed reference: hc_post then hc_pre; the fused triton kernel
    replaces it on CUDA and is validated against this function."""
    residual_new = mhc_post(x, residual, post_mix, comb_mix)
    post_new, comb_new, layer_input = mhc_pre(
        residual_new, fn, hc_scale, hc_base, rms_eps, hc_eps, post_mult, sinkhorn_repeat
    )
    return residual_new, post_new, comb_new, layer_input


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the previous sublayer's hc_post, then this sublayer's hc_pre on the
    updated streams. The fused triton kernel serves every batch size on CUDA
    (deliberately no T threshold); the decomposed torch path serves CPU/tests."""
    if residual.is_cuda:
        from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton

        return mhc_fused_post_pre_triton(
            x, residual, post_mix, comb_mix, fn, hc_scale, hc_base,
            rms_eps, hc_eps, post_mult, sinkhorn_repeat,
        )
    return mhc_fused_post_pre_torch(
        x, residual, post_mix, comb_mix, fn, hc_scale, hc_base,
        rms_eps, hc_eps, post_mult, sinkhorn_repeat,
    )


__all__ = [
    "hc_expand",
    "hc_contract",
    "mhc_pre",
    "mhc_post",
    "mhc_fused_post_pre",
    "mhc_fused_post_pre_torch",
]
