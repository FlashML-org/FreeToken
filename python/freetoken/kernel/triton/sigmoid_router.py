"""Selection-biased sigmoid routing; weights always use the unbiased scores.

Based on the selection loop in dsv4/router.py. Equal expert/group scores choose
the lowest index; torch.topk does not specify an ordering for ties.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_router(
    logits, bias, weights, ids, token_limit,
    stride_row, stride_col, stride_bias,
    E: tl.constexpr, K: tl.constexpr, GROUPS: tl.constexpr,
    KEEP_GROUPS: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr, NORMALIZE: tl.constexpr, EPS: tl.constexpr,
    SCALE: tl.constexpr, HAS_LIMIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, BLOCK_G)
    lanes = tl.arange(0, BLOCK_E)
    expert = groups[:, None] * (E // GROUPS) + lanes[None, :]
    valid = (groups[:, None] < GROUPS) & (lanes[None, :] < E // GROUPS)
    x = tl.load(logits + row * stride_row + expert * stride_col, valid, other=0).to(tl.float32)
    s = tl.sigmoid(x)
    b = tl.load(bias + expert * stride_bias, valid, other=0).to(tl.float32)
    key = tl.where(valid, s + b, -float("inf"))
    eligible = valid
    if GROUPS > 1:
        first = tl.max(key, 1)
        first_id = tl.min(tl.where(key == first[:, None], lanes[None, :], BLOCK_E), 1)
        second = tl.max(tl.where(lanes[None, :] == first_id[:, None], -float("inf"), key), 1)
        group_key = first + second
        group_selected = tl.full((BLOCK_G,), False, tl.int1)
        for _ in tl.static_range(KEEP_GROUPS):
            best = tl.max(tl.where(group_selected, -float("inf"), group_key), 0)
            gid = tl.min(tl.where((group_key == best) & ~group_selected & (groups < GROUPS), groups, BLOCK_G), 0)
            group_selected = group_selected | (groups == gid)
        eligible = valid & group_selected[:, None]
    slots = tl.arange(0, BLOCK_K)
    out_w = tl.zeros((BLOCK_K,), tl.float32)
    out_id = tl.zeros((BLOCK_K,), tl.int32)
    for k in tl.static_range(K):
        best = tl.max(tl.max(tl.where(eligible, key, -float("inf")), 1), 0)
        idx = tl.min(tl.min(tl.where(eligible & (key == best), expert, E), 1), 0)
        w = tl.sum(tl.sum(tl.where(valid & (expert == idx), s, 0.0), 1), 0)
        out_w = tl.where(slots == k, w, out_w)
        out_id = tl.where(slots == k, idx, out_id)
        eligible = eligible & (expert != idx)
    if NORMALIZE:
        out_w = out_w / (tl.sum(out_w, 0) + EPS)
    out_w = out_w * SCALE
    if HAS_LIMIT:
        active = row < tl.load(token_limit)
        out_id = tl.where(active, out_id, -1)
        out_w = tl.where(active, out_w, 0.0)
    tl.store(weights + row * K + slots, out_w, slots < K)
    tl.store(ids + row * K + slots, out_id, slots < K)


def fused_topk_sigmoid(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    renormalize: bool,
    n_group: int = 1,
    topk_group: int = 1,
    routed_scale: float = 1.0,
    eps: float = 1e-20,
    num_token_non_padded: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous fp32 weights and int32 ids for finite logits and bias.

    Group scores are the sum of the two largest biased scores in each group.
    A device token limit masks padded rows with ids -1 and weights zero.
    ``eps=0`` preserves MiniMax-M2's normalization, including zero-sum NaNs.
    """
    assert logits.ndim == 2 and logits.is_cuda
    tokens, experts = logits.shape
    assert bias.shape == (experts,) and bias.device == logits.device
    assert n_group >= 1 and experts % n_group == 0
    assert 1 <= topk <= experts
    assert 1 <= topk_group <= n_group
    assert n_group == 1 or experts // n_group >= 2
    assert topk <= topk_group * (experts // n_group)
    if num_token_non_padded is not None:
        assert num_token_non_padded.device == logits.device
        assert num_token_non_padded.numel() == 1
        assert num_token_non_padded.dtype in (torch.int32, torch.int64)
    weights = torch.empty((tokens, topk), device=logits.device, dtype=torch.float32)
    ids = torch.empty((tokens, topk), device=logits.device, dtype=torch.int32)
    if tokens:
        _sigmoid_router[(tokens,)](
            logits, bias, weights, ids, num_token_non_padded,
            logits.stride(0), logits.stride(1), bias.stride(0),
            E=experts, K=topk, GROUPS=n_group, KEEP_GROUPS=topk_group,
            BLOCK_G=triton.next_power_of_2(n_group),
            BLOCK_E=triton.next_power_of_2(experts // n_group),
            BLOCK_K=triton.next_power_of_2(topk), NORMALIZE=renormalize,
            EPS=eps, SCALE=routed_scale, HAS_LIMIT=num_token_non_padded is not None,
            num_warps=4,
        )
    return weights, ids
