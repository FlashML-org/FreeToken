"""Gather native V4.1 FP8-window/FP4-compressed KV directly into attention tiles."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.sparse_attn import (
    BLOCK_H, BLOCK_T, _sparse_attn_splitk_merge_kernel, split_count,
)
from freetoken.kernel.triton.kv_nvfp4 import _decode_e2m1
from .quant import _e4m3_to_f32, _ue8m0_to_f32


@triton.jit
def _load_mixed(win, cmp, slots, is_window, live, dim,
                win_stride, cmp_stride, D: tl.constexpr):
    base = tl.where(is_window, win, cmp)
    stride = tl.where(is_window, win_stride, cmp_stride)
    row = base[:, None] + slots[:, None].to(tl.int64) * stride[:, None]
    # One selected data load avoids staging two full KV tiles on consumer GPUs.
    data = tl.load(row + dim[None, :], live[:, None] & (is_window[:, None] | (dim[None, :] < D // 2)),
                   other=0).to(tl.int32)
    packed = tl.gather(data, tl.broadcast_to(dim[None, :] // 2, (slots.shape[0], D)), 1)
    raw = tl.where(is_window[:, None], data, packed)
    group = tl.arange(0, D // 16)
    scale_offset = tl.where(is_window[:, None], D + group[None, :] // 2, D // 2 + group[None, :])
    scale_code = tl.load(row + scale_offset, live[:, None], other=0)
    nibble = tl.where((dim[None, :] & 1) == 0, raw & 15, raw >> 4)
    fp8_value = _e4m3_to_f32(raw)
    value = tl.where(is_window[:, None], fp8_value, _decode_e2m1(nibble))
    fp4_scale = _e4m3_to_f32(scale_code)
    scale = tl.where(is_window[:, None], _ue8m0_to_f32(scale_code),
                     fp4_scale)
    # The previous pool held the roundtrip's BF16 result, not its FP32 product.
    restored = value.reshape(slots.shape[0], D // 16, 16) * scale[:, :, None]
    return restored.reshape(slots.shape[0], D).to(tl.bfloat16)


@triton.jit
def _attend(q, win, cmp, out, lse, sink, indices, counts,
            scale, H, TOPK, N_WINDOW,
            qb, qm, qh, qd, win_stride, cmp_stride,
            ob, om, oh, os, od, lb, lm, lh, ls, ib, im, it, cb, cm,
            D: tl.constexpr, BH: tl.constexpr, BT: tl.constexpr,
            HAS_COUNTS: tl.constexpr, SPLITS: tl.constexpr):
    query_split, batch, head_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query, split = query_split // SPLITS, query_split % SPLITS
    h = head_block * BH + tl.arange(0, BH)
    d = tl.arange(0, D)
    hm = h < H
    active = TOPK
    if HAS_COUNTS:
        active = N_WINDOW + tl.load(counts + batch * cb + query * cm)
    first = 0
    end = active
    if SPLITS > 1:
        width = tl.cdiv(tl.cdiv(active, SPLITS), BT) * BT
        first = split * width
        end = tl.minimum(first + width, active)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    if end > first:
        query_values = tl.load(q + batch * qb + query * qm + h[:, None] * qh + d[None, :] * qd,
                               hm[:, None], other=0).to(tl.float32)
        for start in range(first, end, BT):
            columns = start + tl.arange(0, BT)
            slots = tl.load(indices + batch * ib + query * im + columns * it,
                            columns < end, other=-1)
            live = slots >= 0
            kv = _load_mixed(win, cmp, slots, columns < N_WINDOW, live, d,
                             win_stride, cmp_stride, D)
            # Promote after choosing each dot's layout so shared KV tiles stay BF16.
            # Reusing one FP32 tile here exceeds the RTX 5090 shared-memory limit.
            scores = tl.dot(query_values, tl.trans(kv).to(tl.float32)) * scale
            scores = tl.where(live[None, :], scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.where(next_max == -float("inf"), 1.0, tl.exp(maximum - next_max))
            p = tl.where(live[None, :], tl.exp(scores - next_max[:, None]), 0.0)
            denominator = denominator * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p, kv.to(tl.float32))
            maximum = next_max
    if SPLITS == 1:
        sink_value = tl.load(sink + h, hm, other=0).to(tl.float32)
        denominator += tl.exp(sink_value - maximum)
        result = acc / denominator[:, None]
    else:
        result = tl.where(denominator[:, None] == 0, 0.0, acc / denominator[:, None])
        logsum = tl.where(denominator == 0, -float("inf"), maximum + tl.log(denominator))
        tl.store(lse + batch * lb + query * lm + h * lh + split * ls, logsum, hm)
    tl.store(out + batch * ob + query * om + h[:, None] * oh + split * os + d[None, :] * od,
             result.to(out.dtype.element_ty), hm[:, None])


def sparse_attn_paged(q, window_pool, cmp_pool, attn_sink, topk_idxs, n_window,
                      softmax_scale, cmp_counts=None, *, force_splits=None):
    """Attend to packed selected rows; the full cache never expands to BF16."""
    if q.ndim != 4 or not q.is_cuda:
        raise ValueError("Packed sparse attention needs CUDA queries shaped [batch, queries, heads, dim]")
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    if d % 32 or d & (d - 1) or not 0 <= n_window <= topk:
        raise ValueError("Packed V4.1 attention needs a power-of-two dimension divisible by 32 and a valid window width")
    for tier, (pool, width) in enumerate(((window_pool, d + d // 32), (cmp_pool, d // 2 + d // 16))):
        if pool.dtype != torch.uint8 or pool.ndim != 2 or not pool.is_contiguous():
            raise ValueError("Packed KV rows must be contiguous uint8 tensors")
        if pool.device != q.device:
            raise ValueError("Packed KV rows and queries must share a device")
        # Window-only layers alias the window pool; no compressed column is read.
        if pool.shape[1] != width and not (tier == 1 and n_window == topk):
            raise ValueError("Packed KV row width does not match the query dimension")
    if tuple(topk_idxs.shape) != (b, m, topk) or attn_sink.numel() != h:
        raise ValueError("Sparse indices or attention sink do not match the queries")
    if any(t.device != q.device for t in (topk_idxs, attn_sink)):
        raise ValueError("Sparse indices, attention sink, and queries must share a device")
    if cmp_counts is not None and cmp_counts.device != q.device:
        raise ValueError("Compressed counts and queries must share a device")
    q = q.contiguous()
    idx = topk_idxs.contiguous().to(torch.int32)
    sink = attn_sink.contiguous().to(torch.float32)
    out = torch.empty_like(q)
    if not b or not m:
        return out
    has_counts = cmp_counts is not None
    if has_counts:
        counts = cmp_counts.contiguous().to(torch.int32).view(b, m)
        cb, cm = counts.stride()
    else:
        counts, cb, cm = idx, 0, 0
    splits = split_count(b, m, h, topk, q.device) if force_splits is None else force_splits
    if not isinstance(splits, int) or splits < 0:
        raise ValueError("force_splits must be a nonnegative integer")
    splits = max(1, splits)
    if splits > 1:
        partial = torch.empty((b, m, h, splits, d), device=q.device, dtype=torch.float32)
        logsum = torch.empty((b, m, h, splits), device=q.device, dtype=torch.float32)
        lstrides = logsum.stride()
    else:
        partial, logsum = out.unsqueeze(3), out
        lstrides = (0, 0, 0, 0)
    _attend[(m * splits, b, triton.cdiv(h, BLOCK_H))](
        q, window_pool, cmp_pool, partial, logsum, sink, idx, counts,
        float(softmax_scale), h, topk, n_window,
        *q.stride(), window_pool.stride(0), cmp_pool.stride(0),
        *partial.stride(), *lstrides, *idx.stride(), cb, cm,
        D=d, BH=BLOCK_H, BT=BLOCK_T, HAS_COUNTS=has_counts, SPLITS=splits,
        num_warps=8, num_stages=1,
    )
    if splits > 1:
        _sparse_attn_splitk_merge_kernel[(m, b, h)](
            partial, logsum, out, sink, *partial.stride(), *logsum.stride(), *out.stride(),
            D=d, NUM_SPLITS=splits, num_warps=4,
        )
    return out
