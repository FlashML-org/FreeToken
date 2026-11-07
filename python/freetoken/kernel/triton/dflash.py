"""Triton kernels of the DFlash 2 draft block: attention over the per-request KV ring, and the grouped 2-tap convolution."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ring_attn_split_kernel(
    q_ptr, k_ptr, v_ptr, rows_ptr, starts_ptr,
    o_ptr, m_ptr, l_ptr,
    layer, num_rows, ring_len, window, scale,
    NUM_KV_HEADS: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KEYS_PER_SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_splits = tl.num_programs(1)
    b = pid // NUM_KV_HEADS
    kvh = pid % NUM_KV_HEADS
    row = tl.load(rows_ptr + b)
    start = tl.load(starts_ptr + b)

    m_idx = tl.arange(0, BLOCK_M)
    g = m_idx // BLOCK
    j = m_idx % BLOCK
    m_ok = m_idx < GROUP * BLOCK
    d = tl.arange(0, HEAD_DIM)
    q_off = ((b * BLOCK + j) * (NUM_KV_HEADS * GROUP) + kvh * GROUP + g) * HEAD_DIM
    q = tl.load(q_ptr + q_off[:, None] + d[None, :], mask=m_ok[:, None], other=0.0)

    top = start + BLOCK - 1
    # a query at start+j sees positions > start+j-window (sliding window), and only ones that hold context
    lo = tl.maximum(start + j - window + 1, 0)
    base = (layer * num_rows + row) * ring_len
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    for n0 in range(0, KEYS_PER_SPLIT, BLOCK_N):
        slot = split * KEYS_PER_SPLIT + n0 + tl.arange(0, BLOCK_N)
        s_ok = slot < ring_len
        # % keeps the dividend's sign; slots past top give a negative dividend early in a sequence
        pos = top - ((top - slot) % ring_len + ring_len) % ring_len
        kv_off = ((base + slot) * NUM_KV_HEADS + kvh) * HEAD_DIM
        k = tl.load(k_ptr + kv_off[:, None] + d[None, :], mask=s_ok[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale
        keep = s_ok[None, :] & (pos[None, :] >= lo[:, None])
        qk = tl.where(keep, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(v_ptr + kv_off[:, None] + d[None, :], mask=s_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    part = (pid * num_splits + split) * BLOCK_M + m_idx
    tl.store(o_ptr + part[:, None] * HEAD_DIM + d[None, :], acc)
    tl.store(m_ptr + part, m_i)
    tl.store(l_ptr + part, l_i)


@triton.jit
def _ring_attn_merge_kernel(
    o_ptr, m_ptr, l_ptr, out_ptr, num_splits,
    NUM_KV_HEADS: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, MAX_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.program_id(1)
    b = pid // NUM_KV_HEADS
    kvh = pid % NUM_KV_HEADS
    sp = tl.arange(0, MAX_SPLITS)
    sp_ok = sp < num_splits
    d = tl.arange(0, HEAD_DIM)
    part = (pid * num_splits + sp) * BLOCK_M + m
    ms = tl.load(m_ptr + part, mask=sp_ok, other=-1.0e30)
    ls = tl.load(l_ptr + part, mask=sp_ok, other=0.0)
    w = tl.exp(ms - tl.max(ms, axis=0))
    o = tl.load(o_ptr + part[:, None] * HEAD_DIM + d[None, :], mask=sp_ok[:, None], other=0.0)
    acc = tl.sum(o * w[:, None], axis=0) / tl.sum(w * ls, axis=0)
    g = m // BLOCK
    j = m % BLOCK
    out_off = ((b * BLOCK + j) * (NUM_KV_HEADS * GROUP) + kvh * GROUP + g) * HEAD_DIM
    tl.store(out_ptr + out_off + d, acc.to(out_ptr.dtype.element_ty))


_KEYS_PER_SPLIT = 128
_BLOCK_N = 64


def _block_m(group: int, block: int) -> int:
    return max(16, triton.next_power_of_2(group * block))


def ring_attention_workspace(max_bs: int, block: int, num_heads: int, num_kv_heads: int, head_dim: int, ring_len: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fp32 split partials (out, max, sum) ``ring_attention`` needs for up to ``max_bs`` requests; on the meta device they only size the budget."""
    parts = max_bs * num_kv_heads * triton.cdiv(ring_len, _KEYS_PER_SPLIT) * _block_m(num_heads // num_kv_heads, block)
    f32 = dict(dtype=torch.float32, device=device)
    return torch.empty(parts, head_dim, **f32), torch.empty(parts, **f32), torch.empty(parts, **f32)


def ring_attention(q, ring_k, ring_v, layer: int, rows, starts, window: int, scale: float, workspace) -> torch.Tensor:
    """Non-causal block attention of ``[bs*block, heads*head_dim]`` queries over a ``[layers, rows, ring_len, kv_heads, head_dim]`` KV ring.

    The ring holds position ``p`` at slot ``p % ring_len``; the block occupies ``start .. start+block-1`` and the window behind it the rest.
    """
    ring_len, num_kv_heads, head_dim = ring_k.shape[2:]
    bs = starts.shape[0]
    block, group = q.shape[0] // bs, q.shape[1] // (num_kv_heads * head_dim)
    block_m, num_splits = _block_m(group, block), triton.cdiv(ring_len, _KEYS_PER_SPLIT)
    part_o, part_m, part_l = workspace
    out = torch.empty_like(q)
    _ring_attn_split_kernel[(bs * num_kv_heads, num_splits)](
        q, ring_k, ring_v, rows, starts, part_o, part_m, part_l,
        layer, ring_k.shape[1], ring_len, window, scale,
        NUM_KV_HEADS=num_kv_heads, GROUP=group, BLOCK=block, HEAD_DIM=head_dim,
        BLOCK_M=block_m, BLOCK_N=_BLOCK_N, KEYS_PER_SPLIT=_KEYS_PER_SPLIT, num_warps=4,
    )
    _ring_attn_merge_kernel[(bs * num_kv_heads, group * block)](
        part_o, part_m, part_l, out, num_splits,
        NUM_KV_HEADS=num_kv_heads, GROUP=group, BLOCK=block, HEAD_DIM=head_dim,
        BLOCK_M=block_m, MAX_SPLITS=triton.next_power_of_2(num_splits), num_warps=4,
    )
    return out


@triton.jit
def _grouped_conv_kernel(
    x_ptr, coef_ptr, base_ptr, out_ptr, hidden, coef_row_stride, block,
    TAPS: tl.constexpr, GROUP_SIZE: tl.constexpr, NUM_GROUPS: tl.constexpr, BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_ok = h < hidden
    grp = h // GROUP_SIZE
    first = n % block
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for t in tl.static_range(TAPS):
        coef = tl.load(base_ptr + t * hidden + h, mask=h_ok, other=0.0).to(tl.float32)
        coef += tl.load(coef_ptr + n * coef_row_stride + t * NUM_GROUPS + grp, mask=h_ok, other=0.0).to(tl.float32)
        live = first >= t
        x = tl.load(x_ptr + (n - t) * hidden + h, mask=h_ok & live, other=0.0).to(tl.float32)
        acc += coef * x
    tl.store(out_ptr + n * hidden + h, acc.to(out_ptr.dtype.element_ty), mask=h_ok)


def grouped_conv(x: torch.Tensor, coef: torch.Tensor, base: torch.Tensor, block: int, group_size: int) -> torch.Tensor:
    """``out[n] = sum_t (base[t] + coef[n, t, group]) * x[n - t]`` within each block of ``block`` rows; ``coef`` is ``[N, taps, groups]`` with unit inner strides."""
    n, hidden = x.shape
    taps, groups = coef.shape[1], coef.shape[2]
    assert coef.stride(2) == 1 and coef.stride(1) == groups and x.is_contiguous() and base.is_contiguous()
    out = torch.empty_like(x)
    block_h = 512
    _grouped_conv_kernel[(n, triton.cdiv(hidden, block_h))](
        x, coef, base, out, hidden, coef.stride(0), block,
        TAPS=taps, GROUP_SIZE=group_size, NUM_GROUPS=groups, BLOCK_H=block_h, num_warps=4,
    )
    return out


@triton.jit
def _norm_rope(x1, x2, w_ptr, cos, sin, eps, HALF: tl.constexpr):
    d = tl.arange(0, HALF)
    rstd = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / (2 * HALF) + eps)
    x1 = x1 * rstd * tl.load(w_ptr + d).to(tl.float32)
    x2 = x2 * rstd * tl.load(w_ptr + HALF + d).to(tl.float32)
    return x1 * cos - x2 * sin, x2 * cos + x1 * sin


@triton.jit
def _norm_rope_store_kernel(
    src_ptr, src_stride, q_out_ptr, q_norm_ptr, k_norm_ptr, cos_sin_ptr, pos_ptr, rows_ptr, ring_k_ptr, ring_v_ptr,
    layer0, num_rows, ring_len, eps,
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    n = tl.program_id(0)
    per_layer = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    layer_local = tl.program_id(1) // per_layer
    h = tl.program_id(1) % per_layer
    layer = layer0 + layer_local
    half: tl.constexpr = HEAD_DIM // 2
    d = tl.arange(0, half)
    base = src_ptr + n * src_stride + (layer_local * per_layer + h) * HEAD_DIM
    x1 = tl.load(base + d).to(tl.float32)
    x2 = tl.load(base + half + d).to(tl.float32)
    pos = tl.load(pos_ptr + n)
    slot = pos.to(tl.int64) % ring_len
    row = tl.load(rows_ptr + n)
    cos = tl.load(cos_sin_ptr + pos * HEAD_DIM + d)
    sin = tl.load(cos_sin_ptr + pos * HEAD_DIM + half + d)
    if h >= NUM_Q_HEADS + NUM_KV_HEADS:
        dst = ring_v_ptr + (((layer * num_rows + row) * ring_len + slot) * NUM_KV_HEADS + h - NUM_Q_HEADS - NUM_KV_HEADS) * HEAD_DIM
        tl.store(dst + d, x1.to(ring_v_ptr.dtype.element_ty))
        tl.store(dst + half + d, x2.to(ring_v_ptr.dtype.element_ty))
    elif h < NUM_Q_HEADS:
        o1, o2 = _norm_rope(x1, x2, q_norm_ptr, cos, sin, eps, half)
        dst = q_out_ptr + (n * NUM_Q_HEADS + h) * HEAD_DIM
        tl.store(dst + d, o1.to(q_out_ptr.dtype.element_ty))
        tl.store(dst + half + d, o2.to(q_out_ptr.dtype.element_ty))
    else:
        o1, o2 = _norm_rope(x1, x2, k_norm_ptr + layer * HEAD_DIM, cos, sin, eps, half)
        dst = ring_k_ptr + (((layer * num_rows + row) * ring_len + slot) * NUM_KV_HEADS + h - NUM_Q_HEADS) * HEAD_DIM
        tl.store(dst + d, o1.to(ring_k_ptr.dtype.element_ty))
        tl.store(dst + half + d, o2.to(ring_k_ptr.dtype.element_ty))


def norm_rope_store(
    src: torch.Tensor, *, num_layers: int, num_q_heads: int, q_out: torch.Tensor | None, q_norm: torch.Tensor | None,
    k_norms: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor,
    ring_k: torch.Tensor, ring_v: torch.Tensor, layer0: int, eps: float,
) -> None:
    """Per-head RMSNorm + NeoX RoPE of ``[n, num_layers*(q|k|v heads)*head_dim]`` rows: q into ``q_out``, k and v into the ring at ``positions % ring_len``.

    ``k_norms`` is ``[layers, head_dim]`` indexed by ``layer0 + layer``; ``cos_sin`` is ``[max_pos, head_dim]`` (cos half, sin half).
    """
    n = src.shape[0]
    num_kv_heads, head_dim = ring_k.shape[3], ring_k.shape[4]
    assert src.stride(1) == 1 and cos_sin.dtype == torch.float32 and cos_sin.shape[1] == head_dim
    _norm_rope_store_kernel[(n, num_layers * (num_q_heads + 2 * num_kv_heads))](
        src, src.stride(0), q_out if q_out is not None else src, q_norm if q_norm is not None else k_norms, k_norms, cos_sin,
        positions, rows, ring_k, ring_v, layer0, ring_k.shape[1], ring_k.shape[2], eps,
        NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim, num_warps=1,
    )


@triton.jit
def _selector_kernel(
    ids_ptr, unary_ptr, state_ptr, anchor_ptr, pre_ptr, suc_ptr, temp_ptr, greedy_ptr, u_ptr, tok_ptr, q_ptr,
    SLOTS: tl.constexpr, TOP_K: tl.constexpr, RANK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.arange(0, TOP_K)
    r = tl.arange(0, RANK)
    temperature = tl.load(temp_ptr + b)
    greedy = tl.load(greedy_ptr + b) != 0
    prev = tl.load(anchor_ptr + b)
    for slot in tl.static_range(SLOTS):
        base = (b * SLOTS + slot) * TOP_K
        cand = tl.load(ids_ptr + base + c)
        w = tl.load(pre_ptr + prev * RANK + r).to(tl.float32) * tl.load(state_ptr + (b * SLOTS + slot) * RANK + r).to(tl.float32)
        keys = tl.load(suc_ptr + cand[:, None] * RANK + r[None, :]).to(tl.float32)
        scores = tl.load(unary_ptr + base + c) + tl.sum(keys * w[None, :], axis=1)
        if greedy:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, c, TOP_K), axis=0)
            probs = tl.where(c == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            e = tl.exp(scaled - tl.max(scaled, axis=0))
            probs = e / tl.sum(e, axis=0)
            u = tl.load(u_ptr + b * SLOTS + slot)
            index = tl.minimum(tl.sum(tl.where(u >= tl.cumsum(probs, axis=0), 1, 0), axis=0), TOP_K - 1)
            # fp32 cdf rounding can land u on a candidate whose q underflowed to 0, which rejection would accept on any p > 0
            index = tl.max(tl.where((c <= index) & (probs > 0), c, 0), axis=0)
        tl.store(q_ptr + base + c, probs)
        prev = tl.sum(tl.where(c == index, cand, 0), axis=0)
        tl.store(tok_ptr + b * SLOTS + slot, prev)


def selector_walk(ids, unary, state, anchor, pre, suc, temperature, greedy, uniforms):
    """One path through the candidate lattice per request: slot e scores ``unary[e] + <pre[prev] * state[e], suc[cand]>`` for the chosen predecessor only.

    Argmax on ``greedy`` rows, else drawn from softmax(scores / temperature) with ``uniforms``; returns tokens ``[bs, slots]`` and q ``[bs, slots, top_k]``.
    """
    bs, slots, top_k = ids.shape
    tokens = torch.empty(bs, slots, dtype=torch.int64, device=ids.device)
    q = torch.empty(bs, slots, top_k, dtype=torch.float32, device=ids.device)
    _selector_kernel[(bs,)](
        ids, unary, state, anchor, pre, suc, temperature, greedy, uniforms, tokens, q,
        SLOTS=slots, TOP_K=top_k, RANK=pre.shape[1], num_warps=4,
    )
    return tokens, q
