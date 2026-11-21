"""Triton kernels of Inkling 2 Mini (``models/inkling2``).

* ``extend_attention`` / ``decode_attention``: causal GQA over the paged KV cache with two
  additions no shared backend has: a per-query-token, per-head bias over the backward
  distance ``d = q_pos - k_pos`` (read from a ``[T, H, E]`` table, zero for ``d >= E``) and a
  per-query-token, per-head sink logit that only enlarges the softmax denominator.
* ``short_conv`` / ``short_conv_update_state``: the depthwise causal convolutions with a
  residual (``(conv(x) + x) * scale``), static or with per-token taps, whose left context
  comes from a per-request state row on the first rows of a chunk.
* ``attn_res``: the attention residual -- a softmax over RMS-normalized source rows, scored
  by learned queries, mixing the normalized rows.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


@triton.jit
def _extend_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    bias_ptr,
    sink_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    row_lo,
    row_hi,
    qk_scale,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_ot,
    stride_oh,
    stride_bt,
    stride_bh,
    stride_st,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    REL_EXTENT: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head = tl.program_id(1)
    block_m = tl.program_id(2)
    kv_head = q_head // GROUP

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_len = tl.load(qo_indptr_ptr + seq_id + 1) - q_start
    first = block_m * BLOCK_M
    # rows of this launch's bias window only: [row_lo, row_hi) of the batch
    if first >= q_len or q_start + first >= row_hi or q_start + first + BLOCK_M <= row_lo:
        return
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    kv_len = tl.load(kv_indptr_ptr + seq_id + 1) - kv_start
    prefix_len = tl.load(prefix_lens_ptr + seq_id)

    offs_m = first + tl.arange(0, BLOCK_M)
    rows = q_start + offs_m
    mask_m = (offs_m < q_len) & (rows >= row_lo) & (rows < row_hi)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    q_pos = prefix_len + offs_m

    q = tl.load(q_ptr + rows[:, None] * stride_qt + q_head * stride_qh + offs_d[None, :], mask=mask_m[:, None], other=0.0)
    # the sink starts the running max with one unit of mass, so no row is ever empty
    m_i = tl.load(sink_ptr + rows * stride_st + q_head, mask=mask_m, other=0.0).to(tl.float32) * 1.4426950408889634
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    bias_rows = bias_ptr + (rows - row_lo)[:, None] * stride_bt + q_head * stride_bh

    hi = tl.minimum(kv_len, prefix_len + tl.minimum(q_len, first + BLOCK_M))
    if SLIDING_WINDOW > 0:
        lo = tl.maximum(prefix_len + first - SLIDING_WINDOW + 1, 0) // BLOCK_N * BLOCK_N
    else:
        lo = 0
    for start_n in tl.range(lo, hi, BLOCK_N):
        k_pos = start_n + offs_n
        mask_n = k_pos < kv_len
        dist = q_pos[:, None] - k_pos[None, :]
        mask = mask_m[:, None] & mask_n[None, :] & (dist >= 0)
        if SLIDING_WINDOW > 0:
            mask = mask & (dist < SLIDING_WINDOW)
        slots = tl.load(kv_indices_ptr + kv_start + k_pos, mask=mask_n, other=0).to(tl.int64)
        k = tl.load(k_ptr + slots[None, :] * stride_ks + kv_head * stride_kh + offs_d[:, None], mask=mask_n[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        bias = tl.load(bias_rows + dist, mask=mask & (dist < REL_EXTENT), other=0.0)
        s = s + bias.to(tl.float32) * 1.4426950408889634
        s = tl.where(mask, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        v = tl.load(v_ptr + slots[:, None] * stride_vs + kv_head * stride_vh + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(o_ptr + rows[:, None] * stride_ot + q_head * stride_oh + offs_d[None, :], out.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


def extend_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    bias: torch.Tensor,
    sinks: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: float,
    sliding_window: int | None,
    rel_extent: int,
    out: torch.Tensor,
    row_lo: int = 0,
    row_hi: int | None = None,
) -> torch.Tensor:
    """Prefill / extend attention of the query rows ``[row_lo, row_hi)`` (default all).

    ``q`` ``[T, Hq, D]``; the K/V caches are flat ``[slots, Hkv, D]`` and already hold this
    chunk's rows. ``bias`` ``[row_hi - row_lo, Hq, rel_extent]`` holds the window's rows;
    ``sinks`` ``[T, Hq]`` fp32. Query positions are ``prefix_len + row-in-request``.
    """
    T, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    row_hi = T if row_hi is None else row_hi
    assert bias.shape == (row_hi - row_lo, Hq, rel_extent) and bias.stride(-1) == 1, bias.shape
    assert sinks.shape == (T, Hq) and sinks.stride(-1) == 1
    block_m, block_n = 64, 64
    grid = (qo_indptr.numel() - 1, Hq, triton.cdiv(max_q_len, block_m))
    _extend_attention_kernel[grid](
        q, k_cache, v_cache, out, bias, sinks,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        row_lo, row_hi, sm_scale * _LOG2E,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1),
        out.stride(0), out.stride(1), bias.stride(0), bias.stride(1), sinks.stride(0),
        GROUP=Hq // Hkv, D=D, BLOCK_M=block_m, BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0, REL_EXTENT=rel_extent,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def _decode_attention_stage1(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    part_acc_ptr,
    part_ml_ptr,
    qk_scale,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_bt,
    stride_bh,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    REL_EXTENT: tl.constexpr,
):
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    kv_start = tl.load(kv_indptr_ptr + b)
    kv_len = tl.load(kv_indptr_ptr + b + 1) - kv_start
    q_pos = kv_len - 1  # one query row per request: the newest key
    if SLIDING_WINDOW > 0:
        lo = tl.maximum(q_pos - SLIDING_WINDOW + 1, 0)
    else:
        lo = 0
    per = tl.cdiv(tl.cdiv(kv_len - lo, NUM_SPLITS), BLOCK_N) * BLOCK_N
    s_lo = lo + split * per
    s_hi = tl.minimum(s_lo + per, kv_len)

    offs_g = tl.arange(0, GROUP_PAD)
    mask_g = offs_g < GROUP
    heads = kv_head * GROUP + offs_g
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(q_ptr + b * stride_qt + heads[:, None] * stride_qh + offs_d[None, :], mask=mask_g[:, None], other=0.0)
    bias_rows = bias_ptr + b * stride_bt + heads[:, None] * stride_bh

    m_i = tl.full((GROUP_PAD,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((GROUP_PAD,), dtype=tl.float32)
    acc = tl.zeros((GROUP_PAD, D), dtype=tl.float32)
    for start in tl.range(s_lo, s_hi, BLOCK_N):
        k_pos = start + offs_n
        mask_n = k_pos < s_hi
        slots = tl.load(kv_indices_ptr + kv_start + k_pos, mask=mask_n, other=0).to(tl.int64)
        k = tl.load(k_ptr + slots[None, :] * stride_ks + kv_head * stride_kh + offs_d[:, None], mask=mask_n[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        dist = q_pos - k_pos
        bias = tl.load(bias_rows + dist[None, :], mask=mask_g[:, None] & (mask_n & (dist < REL_EXTENT))[None, :], other=0.0)
        s = s + bias.to(tl.float32) * 1.4426950408889634
        s = tl.where(mask_n[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        v = tl.load(v_ptr + slots[:, None] * stride_vs + kv_head * stride_vh + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    part = (b * HQ + heads) * NUM_SPLITS + split
    tl.store(part_acc_ptr + part[:, None] * D + offs_d[None, :], acc, mask=mask_g[:, None])
    tl.store(part_ml_ptr + part * 2, m_i, mask=mask_g)
    tl.store(part_ml_ptr + part * 2 + 1, l_i, mask=mask_g)


@triton.jit
def _decode_attention_stage2(
    part_acc_ptr,
    part_ml_ptr,
    sink_ptr,
    o_ptr,
    stride_st,
    stride_ot,
    stride_oh,
    HQ: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, D)
    part = (b * HQ + h) * NUM_SPLITS + offs_s
    m_s = tl.load(part_ml_ptr + part * 2)
    l_s = tl.load(part_ml_ptr + part * 2 + 1)
    sink = tl.load(sink_ptr + b * stride_st + h).to(tl.float32) * 1.4426950408889634
    m = tl.maximum(sink, tl.max(m_s, axis=0))
    w = tl.math.exp2(m_s - m)  # an empty split carries m = -inf, so weight 0
    acc = tl.sum(tl.load(part_acc_ptr + part[:, None] * D + offs_d[None, :]) * w[:, None], axis=0)
    l = tl.math.exp2(sink - m) + tl.sum(l_s * w, axis=0)
    tl.store(o_ptr + b * stride_ot + h * stride_oh + offs_d, (acc / l).to(o_ptr.dtype.element_ty))


def decode_num_splits(batch: int, num_kv_heads: int, device_index: int) -> int:
    """KV splits per (request, KV head): enough programs to fill the GPU twice, at most 32."""
    want = triton.cdiv(2 * _num_sms(device_index), max(1, batch * num_kv_heads))
    return max(1, min(32, triton.next_power_of_2(want)))


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    bias: torch.Tensor,
    sinks: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None,
    rel_extent: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """One query row per request (the request's newest key): split-K over the KV range."""
    B, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    group = Hq // Hkv
    splits = decode_num_splits(B, Hkv, q.device.index)
    part_acc = torch.empty((B * Hq * splits, D), dtype=torch.float32, device=q.device)
    part_ml = torch.empty((B * Hq * splits, 2), dtype=torch.float32, device=q.device)
    _decode_attention_stage1[(B, Hkv, splits)](
        q, k_cache, v_cache, bias, kv_indptr, kv_indices, part_acc, part_ml, sm_scale * _LOG2E,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1),
        bias.stride(0), bias.stride(1),
        GROUP=group, GROUP_PAD=max(16, triton.next_power_of_2(group)), HQ=Hq, D=D, BLOCK_N=32,
        NUM_SPLITS=splits, SLIDING_WINDOW=sliding_window or 0, REL_EXTENT=rel_extent,
        num_warps=4, num_stages=2,
    )
    _decode_attention_stage2[(B, Hq)](
        part_acc, part_ml, sinks, out, sinks.stride(0), out.stride(0), out.stride(1),
        HQ=Hq, D=D, NUM_SPLITS=splits, num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# short convolutions
# ---------------------------------------------------------------------------


@triton.jit
def _short_conv_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    dyn_ptr,
    state_ptr,
    slots_ptr,
    q_to_req_ptr,
    cu_q_ptr,
    prefix_lens_ptr,
    T,
    C,
    scale,
    stride_xt,
    stride_ot,
    stride_dt,
    stride_ss,
    stride_sr,
    HEAD_DIM: tl.constexpr,
    K: tl.constexpr,
    DYNAMIC: tl.constexpr,
    DECODE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_t = offs_t < T
    mask_c = offs_c < C
    mask = mask_t[:, None] & mask_c[None, :]
    if DECODE:
        req = offs_t
        local = tl.zeros((BLOCK_T,), dtype=tl.int32)
        has_state = mask_t
    else:
        req = tl.load(q_to_req_ptr + offs_t, mask=mask_t, other=0)
        local = offs_t - tl.load(cu_q_ptr + req, mask=mask_t, other=0)
        has_state = mask_t & (tl.load(prefix_lens_ptr + req, mask=mask_t, other=0) > 0)
    slot = tl.load(slots_ptr + req, mask=mask_t, other=0).to(tl.int64)

    x = tl.load(x_ptr + offs_t[:, None] * stride_xt + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
    acc = x * (tl.load(w_ptr + offs_c * K, mask=mask_c, other=0.0)[None, :] + 1.0)
    if DYNAMIC:
        acc += x * tl.load(dyn_ptr + offs_t[:, None] * stride_dt + (offs_c // HEAD_DIM)[None, :] * K, mask=mask, other=0.0).to(tl.float32)
    for lag in tl.static_range(1, K):
        src = local - lag
        in_chunk = src >= 0
        prev = tl.load(x_ptr + (offs_t - lag)[:, None] * stride_xt + offs_c[None, :], mask=mask & in_chunk[:, None], other=0.0).to(tl.float32)
        # the state row holds the K - 1 inputs before the chunk, oldest first
        from_state = tl.load(
            state_ptr + slot[:, None] * stride_ss + (K - 1 + src)[:, None] * stride_sr + offs_c[None, :],
            mask=mask & (has_state & (src < 0))[:, None],
            other=0.0,
        ).to(tl.float32)
        prev = tl.where(in_chunk[:, None], prev, from_state)
        w = tl.load(w_ptr + offs_c * K + lag, mask=mask_c, other=0.0)[None, :]
        if DYNAMIC:
            w = w + tl.load(dyn_ptr + offs_t[:, None] * stride_dt + (offs_c // HEAD_DIM)[None, :] * K + lag, mask=mask, other=0.0).to(tl.float32)
        acc += prev * w
    tl.store(out_ptr + offs_t[:, None] * stride_ot + offs_c[None, :], (acc * scale).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _short_conv_state_kernel(
    x_ptr,
    state_ptr,
    slots_ptr,
    cu_q_ptr,
    prefix_lens_ptr,
    C,
    stride_xt,
    stride_ss,
    stride_sr,
    K: tl.constexpr,
    DECODE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    req = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    if DECODE:
        start = req
        length = 1
        has_state = True
    else:
        start = tl.load(cu_q_ptr + req)
        length = tl.load(cu_q_ptr + req + 1) - start
        has_state = tl.load(prefix_lens_ptr + req) > 0
    row = state_ptr + tl.load(slots_ptr + req).to(tl.int64) * stride_ss + offs_c
    # gather all three new rows before writing: the shift reads rows it then overwrites
    tl.static_assert(K == 4)
    r0 = _state_row(x_ptr, row, start, length - 3, stride_xt, stride_sr, offs_c, mask_c, has_state)
    r1 = _state_row(x_ptr, row, start, length - 2, stride_xt, stride_sr, offs_c, mask_c, has_state)
    r2 = _state_row(x_ptr, row, start, length - 1, stride_xt, stride_sr, offs_c, mask_c, has_state)
    tl.store(row, r0.to(state_ptr.dtype.element_ty), mask=mask_c)
    tl.store(row + stride_sr, r1.to(state_ptr.dtype.element_ty), mask=mask_c)
    tl.store(row + 2 * stride_sr, r2.to(state_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def _state_row(x_ptr, row, start, src, stride_xt, stride_sr, offs_c, mask_c, has_state):
    """Input ``src`` rows after the chunk start: from the chunk when ``src >= 0``, else from the old state."""
    from_x = tl.load(x_ptr + (start + src) * stride_xt + offs_c, mask=mask_c & (src >= 0), other=0.0).to(tl.float32)
    from_state = tl.load(row + (3 + src) * stride_sr, mask=mask_c & (src < 0) & has_state, other=0.0).to(tl.float32)
    return tl.where(src >= 0, from_x, from_state)


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    scale: float,
    *,
    decode: bool,
    q_to_req: torch.Tensor | None = None,
    cu_q: torch.Tensor | None = None,
    prefix_lens: torch.Tensor | None = None,
    dynamic: torch.Tensor | None = None,
    head_dim: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(sum_lag w[c, lag] * x[t - lag, c] + x[t, c]) * scale`` for each row, then roll the state.

    ``weight`` ``[C, K]`` fp32 indexed by backward lag (lag 0 = the current row). ``dynamic``
    ``[T, C // head_dim, K]`` adds per-row taps shared by the ``head_dim`` channels of a head.
    ``state`` ``[rows, K - 1, C]`` view (row stride may exceed C) holds each request's last
    ``K - 1`` inputs; ``slots[r]`` is request ``r``'s row. A prefill request with
    ``prefix_lens[r] == 0`` starts from zeros. Decode: one row per request, no metadata.
    """
    T, C = x.shape
    K = weight.shape[1]
    assert weight.shape[0] == C and state.shape[1:] == (K - 1, C) and state.stride(-1) == 1 and x.stride(-1) == 1
    if out is None:
        out = torch.empty_like(x)
    if dynamic is not None:
        assert dynamic.shape == (T, C // head_dim, K) and dynamic.stride(-1) == 1 and dynamic.stride(-2) == K
    dummy = slots
    block_t, block_c = 16, 256
    _short_conv_kernel[(triton.cdiv(T, block_t), triton.cdiv(C, block_c))](
        x, out, weight, dynamic if dynamic is not None else x, state, slots,
        dummy if decode else q_to_req, dummy if decode else cu_q, dummy if decode else prefix_lens,
        T, C, scale, x.stride(0), out.stride(0), dynamic.stride(0) if dynamic is not None else 0,
        state.stride(0), state.stride(1),
        HEAD_DIM=head_dim, K=K, DYNAMIC=dynamic is not None, DECODE=decode, BLOCK_T=block_t, BLOCK_C=block_c,
        num_warps=4,
    )
    num_reqs = T if decode else cu_q.numel() - 1
    _short_conv_state_kernel[(num_reqs, triton.cdiv(C, 1024))](
        x, state, slots, dummy if decode else cu_q, dummy if decode else prefix_lens,
        C, x.stride(0), state.stride(0), state.stride(1),
        K=K, DECODE=decode, BLOCK_C=1024, num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# attention residual
# ---------------------------------------------------------------------------


def attn_res(
    sources: list[torch.Tensor],
    queries: torch.Tensor,
    *,
    eps: float = 1e-6,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[NQ, T, H]``: per row, ``softmax_s(q . n_s / sqrt(H)) @ n_s`` with ``n_s = rmsnorm(sources[s])``.

    ``queries`` ``[NQ, H]`` fp32. With ``norm_weight`` ``[NQ, H]`` each mixed row also goes
    through a weighted RMSNorm (``norm_eps``) before it is stored in ``out_dtype``.
    """
    H = sources[0].shape[-1]
    x = torch.stack([s.float() for s in sources], dim=1)
    n = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    p = torch.einsum("tsh,qh->tqs", n, queries.float()).mul_(H**-0.5).softmax(-1)
    y = torch.einsum("tqs,tsh->qth", p, n)
    if norm_weight is not None:
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + norm_eps) * norm_weight.float()[:, None, :]
    return y.to(out_dtype)


__all__ = ["attn_res", "decode_attention", "decode_num_splits", "extend_attention", "short_conv"]
