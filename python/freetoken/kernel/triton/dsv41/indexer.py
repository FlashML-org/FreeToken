"""Paged V4.1 index scores without materializing per-head score tensors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.kv_nvfp4 import _decode_e2m1
from .quant import _ue8m0_to_f32


@triton.jit
def _scores(q, weights, pool, table, ids, valid, out,
            qs, qh, qd, ws, wh, ps, pd, ts, td, ids_s, ids_k, os, ok,
            K: tl.constexpr, RATIO: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            BH: tl.constexpr, BK: tl.constexpr, PACKED: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * BK + tl.arange(0, BK)
    h, d = tl.arange(0, BH), tl.arange(0, D)
    logical = tl.load(ids + row * ids_s + j * ids_k, j < K, other=-1)
    limit = tl.load(valid + row)
    live = (j < K) & (logical >= 0) & (logical < limit)
    loc = tl.load(table + row * ts + tl.maximum(logical, 0) * RATIO * td, live, other=-1)
    live = live & (loc >= 0)
    key_rows = pool + (tl.maximum(loc, 0) // RATIO)[:, None].to(tl.int64) * ps
    if PACKED:
        raw = tl.load(key_rows + d[None, :] // 2, live[:, None], other=0).to(tl.int32)
        code = tl.where((d[None, :] & 1) == 0, raw & 15, raw >> 4)
        scale = tl.load(key_rows + D // 2 + d[None, :] // 32, live[:, None], other=0)
        k = (_decode_e2m1(code) * _ue8m0_to_f32(scale)).to(tl.bfloat16).to(q.dtype.element_ty)
    else:
        k = tl.load(key_rows + d[None, :] * pd, live[:, None], other=0)
    query = tl.load(q + row * qs + h[:, None] * qh + d[None, :] * qd, h[:, None] < H, other=0)
    w = tl.load(weights + row * ws + h * wh, h < H, other=0)
    # The reference einsum and elementwise product each round to the query dtype.
    dot = tl.dot(query, tl.trans(k)).to(query.dtype).to(tl.float32)
    weighted = (tl.maximum(dot, 0.0) * w[:, None].to(tl.float32)).to(query.dtype).to(tl.float32)
    score = tl.sum(weighted, 0).to(query.dtype)
    tl.store(out + row * os + j * ok, tl.where(live, score, -float("inf")), j < K)


def index_scores(q, weights, pool, table, ids, ratio, valid):
    """Score logical key IDs [queries, keys] through each query's page-table row."""
    if ids.shape[0] != q.shape[0] or table.shape[0] != q.shape[0]:
        raise ValueError("Index query, ID, and page-table row counts differ")
    count, heads, dim = q.shape
    packed = pool.dtype == torch.uint8
    if packed and (dim % 32 or pool.ndim != 2 or pool.shape[1] != dim // 2 + dim // 32
                   or not pool.is_contiguous()):
        raise ValueError("Packed index rows need E2M1 bytes followed by one UE8M0 scale per 32 values")
    width = ids.shape[1]
    out = torch.empty((count, width), device=q.device, dtype=q.dtype)
    if not width:
        return out
    if not q.is_cuda:
        for row in range(count):
            live = (ids[row] >= 0) & (ids[row] < valid[row])
            positions = (ids[row].clamp_min(0) * ratio).clamp_max(table.shape[1] - 1)
            loc = table[row, positions].long()
            live = live & (loc >= 0)
            keys = pool[loc.clamp_min(0) // ratio]
            if packed:
                from .quant import unpack_fp4
                keys = unpack_fp4(keys, block_size=32, scale_format="e8m0", dtype=torch.bfloat16)
            dot = (q[row].float() @ keys.float().T).to(q.dtype)
            score = (dot.relu() * weights[row, :, None]).sum(0)
            out[row] = score.masked_fill(~live, -torch.inf)
        return out
    _scores[(count, triton.cdiv(width, 64))](
        q, weights, pool, table, ids, valid, out,
        *q.stride(), *weights.stride(), *pool.stride(), *table.stride(), *ids.stride(), *out.stride(),
        K=width, RATIO=ratio, H=heads, D=dim, BH=max(16, triton.next_power_of_2(heads)), BK=64,
        PACKED=packed, num_warps=4, num_stages=2,
    )
    return out


def _merge_topk(values, indices, next_values, next_indices, count):
    values = torch.cat((values, next_values), -1)
    indices = torch.cat((indices, next_indices), -1)
    # Equal quantized scores prefer earlier positions, independent of tile/chunk widths.
    by_position = indices.argsort(dim=-1, stable=True)
    by_score = values.gather(-1, by_position).argsort(dim=-1, descending=True, stable=True)
    selected = by_position.gather(-1, by_score[..., :count])
    return values.gather(-1, selected), indices.gather(-1, selected)


def select_indices(q, weights, pool, table, valid, width, ratio, topk,
                   *, candidates=None, candidate_topk=0, block_size=8,
                   query_tile=32, key_tile=4096):
    """Exact tiled top-k, optionally publishing or consuming candidate block IDs.

    Working score memory is bounded by query_tile * key_tile. The source retains
    candidate IDs instead of a query-by-context boolean mask.
    """
    if key_tile % block_size:
        raise ValueError("Index key tile must contain whole candidate blocks")
    result, block_result = [], []
    final_k = min(topk, width)
    final_c = min(candidate_topk, triton.cdiv(width, block_size))
    for first in range(0, q.shape[0], query_tile):
        last = min(first + query_tile, q.shape[0])
        query, gate, mapping, limits = q[first:last], weights[first:last], table[first:last], valid[first:last]
        empty = q.new_empty((last - first, 0))
        best, best_ids = empty, torch.empty_like(empty, dtype=torch.int64)
        cbest, cbest_ids = empty, torch.empty_like(empty, dtype=torch.int64)
        search_width = width if candidates is None else candidates.shape[-1] * block_size
        for offset in range(0, search_width, key_tile):
            end = min(offset + key_tile, search_width)
            if candidates is None:
                ids = torch.arange(offset, end, device=q.device).expand(last - first, -1)
            else:
                blocks = candidates[first:last, offset // block_size:triton.cdiv(end, block_size)]
                ids = blocks[:, :, None] * block_size + torch.arange(block_size, device=q.device)
                ids = torch.where(blocks[:, :, None] >= 0, ids, -1).flatten(1)[:, :end-offset]
            scores = index_scores(query, gate, pool, mapping, ids, ratio, limits)
            best, best_ids = _merge_topk(best, best_ids, scores, ids, final_k)
            if candidate_topk:
                padded = torch.nn.functional.pad(scores, (0, -scores.shape[-1] % block_size), value=-torch.inf)
                block_scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
                block_ids = torch.arange(offset // block_size, triton.cdiv(end, block_size), device=q.device)
                block_ids = block_ids.expand(last - first, -1)
                newest = (limits - 1) // block_size
                block_scores = block_scores.masked_fill((block_ids == newest[:, None]) & (limits[:, None] > 0), torch.inf)
                cbest, cbest_ids = _merge_topk(cbest, cbest_ids, block_scores, block_ids, final_c)
        ids = torch.where(torch.isfinite(best), best_ids, width)
        ids = ids.sort(-1).values
        result.append(torch.where(ids < width, ids, -1))
        if candidate_topk:
            block_result.append(torch.where(cbest > -torch.inf, cbest_ids, -1))
    selected = torch.cat(result, 0) if result else torch.empty((0, final_k), device=q.device, dtype=torch.int64)
    blocks = torch.cat(block_result, 0) if block_result else None
    return selected, blocks
