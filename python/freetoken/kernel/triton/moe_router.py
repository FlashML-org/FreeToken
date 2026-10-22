# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (kernels/ops/moe/moe_fused_gate.py)
"""Fused top-k MoE routers: softmax (bias-free, ungrouped) and sigmoid with a selection-only bias and optional group-limited top-k.

The ``num_token_non_padded`` row mask reads the device tensor, so it survives CUDA-graph capture.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from freetoken.utils.arch import is_sm90_supported


@functools.cache
def _pdl_supported() -> bool:
    return is_sm90_supported()


@triton.jit
def _keep_top_groups(ranked, offs_n, N_GROUP: tl.constexpr, TOPK_GROUP: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_G: tl.constexpr):
    # DeepSeek-V3 noaux_tc: a group scores the sum of its top-2 entries; experts outside the TOPK_GROUP best groups drop to -inf.
    offs_g = tl.arange(0, BLOCK_G)
    group_of_n = offs_n // GROUP_SIZE
    group_score = tl.full([ranked.shape[0], BLOCK_G], -float("inf"), dtype=tl.float32)
    for g in tl.static_range(N_GROUP):
        vals = tl.where(group_of_n[None, :] == g, ranked, -float("inf"))
        top1 = tl.max(vals, axis=1)[:, None]
        top1_lane = tl.min(tl.where(vals == top1, offs_n[None, :], offs_n.shape[0]), axis=1)[:, None]
        top2 = tl.max(tl.where(offs_n[None, :] == top1_lane, -float("inf"), vals), axis=1)[:, None]
        group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

    keep = tl.zeros(ranked.shape, dtype=tl.int1)
    for _ in tl.static_range(TOPK_GROUP):
        best = tl.max(group_score, axis=1)[:, None]
        win_g = tl.min(tl.where(group_score == best, offs_g[None, :], BLOCK_G), axis=1)[:, None]
        keep = keep | (group_of_n[None, :] == win_g)
        group_score = tl.where(offs_g[None, :] == win_g, -float("inf"), group_score)
    return tl.where(keep, ranked, -float("inf"))


@triton.jit
def _router_triton_kernel(
    scores_ptr,
    bias_ptr,
    out_weights_ptr,
    out_indices_ptr,
    num_token_non_padded_ptr,
    M,
    routed_scale,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SIGMOID: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_TOKEN_LIMIT: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # offs_m * stride can overflow int32 for large token counts.
    row_ptr = scores_ptr + offs_m[:, None].to(tl.int64) * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    logits = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)

    if SIGMOID:
        # libdevice exp and round-to-nearest division match torch.sigmoid bit for bit.
        activated = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        ranked = tl.where(mask_n[None, :], activated + bias[None, :], -float("inf"))
    else:
        ranked = tl.where(mask_n[None, :], logits, -float("inf"))
        row_max = tl.max(ranked, axis=1)[:, None]
        exp_row = tl.where(mask_n[None, :], tl.exp(ranked - row_max), 0.0)
        activated = exp_row / tl.sum(exp_row, axis=1)[:, None]

    # Map NaN -> a finite floor
    ranked = tl.where(ranked == ranked, ranked, -1e30)

    if N_GROUP > 1:
        ranked = _keep_top_groups(ranked, offs_n, N_GROUP, TOPK_GROUP, N // N_GROUP, BLOCK_G)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    cur = ranked
    for k in tl.static_range(K):
        max_val = tl.max(cur, axis=1)[:, None]
        lane_id = tl.where(cur == max_val, offs_n[None, :], N + 1)  # lowest expert id wins ties
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)
        win_activated = tl.sum(
            tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1
        )[:, None]
        slot = offs_k[None, :] == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        cur = tl.where(offs_n[None, :] == win_lane, -float("inf"), cur)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        routed_sum = tl.sum(tl.where(mask_k[None, :], selected_vals, 0.0), axis=1)[:, None]
        if SIGMOID:
            selected_vals = tl.div_rn(selected_vals, routed_sum + 1e-20)
        else:
            selected_vals = selected_vals / tl.where(routed_sum > 0.0, routed_sum, 1.0)
    if SIGMOID:
        selected_vals = selected_vals * routed_scale

    if HAS_TOKEN_LIMIT:
        limit = tl.load(num_token_non_padded_ptr)
        selected_idx = tl.where(offs_m[:, None] < limit, selected_idx, -1)

    out_w_ptr = out_weights_ptr + offs_m[:, None].to(tl.int64) * stride_wm + offs_k[None, :] * stride_wk
    out_i_ptr = out_indices_ptr + offs_m[:, None].to(tl.int64) * stride_im + offs_k[None, :] * stride_ik
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


def _launch_router(
    gating_output: torch.Tensor,
    bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    routed_scale: float,
    n_group: int,
    topk_group: int,
    num_token_non_padded: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gating_output.ndim == 2, "gating_output must be 2D"
    M, N = gating_output.shape
    weights = torch.empty((M, topk), dtype=torch.float32, device=gating_output.device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=gating_output.device)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = triton.next_power_of_2(topk)
    # Single warp per program keeps the per-row top-k reductions on cheap warp
    # shuffles; pack a few rows per program only when N is small so tiny launches
    # stay occupancy-bound. Swept on H100/B200; larger tiles / more warps regress
    # (register pressure).
    BLOCK_M = max(1, min(4, 256 // BLOCK_N))
    # For wide rows the K sequential argmax passes dominate and benefit from more
    # warps despite the cross-warp reduction cost.
    num_warps = 1 if BLOCK_N <= 512 else 4

    _router_triton_kernel[(triton.cdiv(M, BLOCK_M),)](
        gating_output,
        bias,
        weights,
        indices,
        num_token_non_padded,
        M,
        routed_scale,
        gating_output.stride(0),
        gating_output.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        N=N,
        K=topk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SIGMOID=bias is not None,
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        BLOCK_G=triton.next_power_of_2(n_group),
        RENORMALIZE=renormalize,
        HAS_TOKEN_LIMIT=num_token_non_padded is not None,
        launch_pdl=_pdl_supported(),
        num_warps=num_warps,
    )
    return weights, indices


def fused_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax over all experts, top-k, then renormalize; ties keep the lowest expert id.

    ``num_token_non_padded`` is a device scalar; rows at or past it get expert id -1.
    """
    return _launch_router(gating_output, None, topk, renormalize, 1.0, 1, 1, num_token_non_padded)


def fused_topk_sigmoid(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scale: float = 1.0,
    n_group: int = 1,
    topk_group: int = 1,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid scores; ``scores + bias`` picks the top-k (within the ``topk_group`` best of ``n_group`` groups when ``n_group > 1``).

    The weights are the unbiased scores, renormalized to sum to 1 when ``renormalize``, then times ``routed_scale``.
    Ties keep the lowest expert id; rows at or past ``num_token_non_padded`` get expert id -1.
    """
    num_experts = gating_output.shape[1]
    assert bias.shape == (num_experts,) and bias.stride(0) == 1
    assert n_group == 1 or (num_experts % n_group == 0 and num_experts // n_group >= 2 and 1 <= topk_group <= n_group)
    return _launch_router(gating_output, bias, topk, renormalize, float(routed_scale), n_group, topk_group, num_token_non_padded)


__all__ = ["fused_topk_sigmoid", "fused_topk_softmax"]
