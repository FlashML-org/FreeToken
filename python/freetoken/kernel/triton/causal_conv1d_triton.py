"""Triton causal_conv1d (provenance: causal_conv1d<-vllm/sglang mamba).

Shipped triton FALLBACK for freetoken.kernel.causal_conv1d (used when the vendored
CUDA kernel can't be built/loaded). The prefill uses a vectorized 2D-tile kernel
(a [BLOCK_M tokens x BLOCK_N feats] tile with per-tap shifted-accumulate), which
is coalesced / high-ILP and approaches HBM bandwidth, vs a per-token serial MAC
loop (~200 GB/s). Launch configs are fixed at the launch sites (one-time H100
sweep; upstream vLLM/sglang likewise hardcode) -- no triton.autotune in this
module.

Public API (unchanged, so kernel/causal_conv1d.py needs no edits):
  * causal_conv1d_fn(x, weight, bias, conv_states, query_start_loc, seq_lens_cpu,
                     cache_indices=, has_initial_state=, activation=, pad_slot_id=)
  * causal_conv1d_update(x, conv_state, weight, bias=, activation=,
                         cache_seqlens=, conv_state_indices=, pad_slot_id=)
These are thin adapters over the tuned causal_conv1d_varlen / causal_conv1d_decode
entrypoints (which mirror the vendored op's own varlen/decode split).

Semantics matched to the vendored CUDA (== sgl_kernel) op:
  * depthwise causal conv1d, kernel width W, channels-first x=(dim, total).
  * silu fused into the output (SILU_ACTIVATION).
  * conv_states[cache_indices] is read as the left context when
    has_initial_state[i] is True, then refreshed in place with each request tail.
  * decode: single token per request, conv_state[idx] shifted left + new token
    appended; silu(conv) returned.  pad_slot_id=-1 entries are skipped.

Not bit-identical to the CUDA op (fp32 accumulation + tl.exp vs -use_fast_math
__expf); allclose(2e-2) in bf16.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


# ---------------------------------------------------------------------------
# varlen / prefill kernel -- vectorized 2D-tile (shifted-tile accumulate)
#
# Loads a [BLOCK_M tokens x BLOCK_N feats] tile and, for each of the WIDTH taps,
# adds w[j] * (tile shifted by j tokens).  All BLOCK_M outputs are computed in
# parallel -> coalesced, high-ILP.  The conv-state tail write is done by the
# chunk_offset==0 program AFTER compute+store so chunk 0 reads the OLD initial
# state first (no write-before-read hazard).
# ---------------------------------------------------------------------------
# seqlen and the x/o dim-strides equal the batch token count for the (dim, total)
# layout -- they must stay runtime values (no constexpr, no int specialization) or
# every distinct prompt length recompiles the kernel.
@triton.jit(do_not_specialize=["seqlen", "stride_x_dim", "stride_o_dim"])
def _causal_conv1d_fwd_tiled_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,
    dim: tl.constexpr,
    seqlen: tl.int32,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    # int64 feature offsets: idx_feats * (dim-stride == total tokens) passes 2^31
    # once dim * total_tokens does (~262k tokens at dim 8192).
    feat_x = idx_feats.to(tl.int64) * stride_x_dim
    feat_o = idx_feats.to(tl.int64) * stride_o_dim

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    if token_offset >= seqlen:
        return

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return

    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_feat = idx_feats < dim

    load_init_state = False
    if HAS_INITIAL_STATES:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)

    # ---- preload weights: w_col_j = w[:, j], shape [BLOCK_N]
    w_base = w_ptr + (idx_feats * stride_w_dim)
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask_feat, other=0.0)
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask_feat, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask_feat, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask_feat, other=0.0)

    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask_feat, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Tile is [BLOCK_M tokens (rows), BLOCK_N feats (cols)].  For each of the WIDTH taps,
    # add w_j * x[token - state_len + j].  Overlapping tap loads are served from L2.
    rows = tl.arange(0, BLOCK_M)                        # [BLOCK_M] token index within chunk
    local_out = token_offset + rows                     # [BLOCK_M] abs local token
    acc = acc_bias[None, :] + tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mfc = (idx_feats < dim)[None, :]

    # Left context (< 0) comes from conv_states (chunk_offset==0, load_init) else zero;
    # chunk>0 always reads x.
    for j in tl.static_range(KERNEL_WIDTH):
        if j == 0:
            w_j = w_col0
        elif j == 1:
            w_j = w_col1
        elif j == 2:
            w_j = w_col2
        else:
            w_j = w_col3
        src_local = local_out - state_len + j            # [BLOCK_M]
        gx = sequence_start_index + src_local
        x_ptrs = (
            x_ptr
            + (gx * stride_x_token)[:, None]
            + feat_x[None, :]
        )
        mask_x = (src_local >= 0)[:, None] & (src_local < seqlen)[:, None] & mfc
        xj = tl.load(x_ptrs, mask_x, 0.0)
        acc += w_j[None, :] * xj
        if HAS_INITIAL_STATES:
            if load_init_state:
                st_idx = state_len + src_local        # 0..state_len-1 where src_local<0
                s_ptrs = (
                    conv_states_base[None, :]
                    + (st_idx * stride_conv_state_tok)[:, None]
                )
                mask_s = (src_local < 0)[:, None] & mfc
                sj = tl.load(s_ptrs, mask_s, 0.0)
                acc += w_j[None, :] * sj

    if SILU_ACTIVATION:
        # silu(x)=x/(1+exp(-x)); exp2 lowers to the native ex2.approx SFU op (faster than the
        # polynomial tl.exp) and matches the CUDA op's -use_fast_math __expf within tol.
        acc = acc / (1.0 + tl.exp2(-acc * 1.4426950408889634))

    o_ptrs = (
        o_ptr
        + ((sequence_start_index + local_out) * stride_o_token)[:, None]
        + feat_o[None, :]
    )
    mask_o = (local_out < seqlen)[:, None] & mfc
    tl.store(o_ptrs, acc, mask_o)

    # ---- conv_state tail write, done LAST (after chunk-0 read the OLD initial state above,
    # so there is no write-before-read hazard).  chunk_offset==0 program only.
    if chunk_offset == 0:
        idx_tok = tl.arange(0, NP2_STATELEN)
        if state_len <= seqlen:
            idx_last = (seqlen - state_len) + idx_tok
            xw_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_last) * stride_x_token)[:, None]
                + feat_x[None, :]
            )
            mask_xw = (idx_last >= 0)[:, None] & (idx_last < seqlen)[:, None] & mfc
            new_cs = tl.load(xw_ptrs, mask_xw, 0.0)
        else:
            VAL = state_len - seqlen
            xw_ptrs = (
                x_ptr
                + (sequence_start_index * stride_x_token)
                + feat_x[None, :]
                + ((idx_tok - VAL) * stride_x_token)[:, None]
            )
            mask_xw = (idx_tok - VAL >= 0)[:, None] & (idx_tok - VAL < seqlen)[:, None] & mfc
            new_cs = tl.load(xw_ptrs, mask_xw, 0.0)
            if HAS_INITIAL_STATES:
                if load_init_state:
                    src_ptrs = (
                        conv_states_ptr
                        + (conv_state_batch_coord * stride_conv_state_seq)
                        + (idx_feats * stride_conv_state_dim)[None, :]
                        + ((idx_tok + seqlen) * stride_conv_state_tok)[:, None]
                    )
                    mask_src = (
                        (conv_state_batch_coord < num_cache_lines)
                        & ((idx_tok + seqlen) < state_len)[:, None]
                        & mfc
                    )
                    old = tl.load(src_ptrs, mask_src, 0.0)
                    new_cs = tl.where(mask_src, old, new_cs)
        tgt = conv_states_base[None, :] + (idx_tok * stride_conv_state_tok)[:, None]
        mask_t = (idx_tok < state_len)[:, None] & mfc
        tl.store(tgt, new_cs, mask_t)


# ---------------------------------------------------------------------------
# decode / update kernel (trimmed continuous-batching path)
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["batch"])
def _causal_conv1d_update_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    o_ptr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return

    # STEP 1: read prior conv_state window
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim
    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        col0 = tl.load(prior_tokens, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        col3 = tl.load(prior_tokens + 3 * stride_conv_state_tok, mask_w, 0.0)

    # STEP 2: shift conv_state left by seqlen, append new tokens from x
    idx_tokens = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    )
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()
    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_ptrs_target = (
        conv_states_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        acc_preload = tl.load(bias, mask=(idx_feats < dim), other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4: preload weights
    w_base = w_ptr + (idx_feats * stride_w_dim)
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d)


# ---------------------------------------------------------------------------
# Tuned entrypoints (mirror the vendored op's varlen / decode split)
# ---------------------------------------------------------------------------
def causal_conv1d_varlen(
    x: torch.Tensor,            # [conv_dim, total_tokens]
    weight: torch.Tensor,       # [conv_dim, kernel]
    conv_states: torch.Tensor,  # [num_slots, conv_dim, kernel-1] (in place)
    cu_seqlens: torch.Tensor,   # [batch+1] int32
    cache_indices: torch.Tensor,      # [batch] int32
    has_initial_state: torch.Tensor,  # [batch] bool
    bias: torch.Tensor | None = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    max_seq_len: Optional[int] = None,
    batch: Optional[int] = None,
) -> torch.Tensor:
    if x.stride(-1) != 1:
        x = x.contiguous()
    cu_seqlens = cu_seqlens.to(torch.int32)
    cache_indices = cache_indices.to(torch.int32)
    if isinstance(activation, bool) and activation:
        activation = "silu"

    out = torch.empty_like(x)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    # The tap chains below are unrolled for widths 2..4 only (wider taps would
    # silently reuse w_col3); GDN/mamba checkpoints all ship width 4.
    assert 2 <= width <= 4, f"causal_conv1d triton fallback supports width 2..4, got {width}"
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    num_cache_lines = conv_states.size(0)
    # max_seq_len / batch are host-known metadata in production (scheduler); pass them in
    # to keep this launch graph-capturable (no .item() device->host sync).
    if batch is None:
        batch = cu_seqlens.numel() - 1
    if max_seq_len is None:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seq_len = int(seq_lens.max().item())
    # grid axis 1 is ceil(max_seq_len / BLOCK_M=16); CUDA caps gridDim.y at 65535.
    assert max_seq_len <= 16 * 65535, "chunk prefill: single-call max_seq_len cap is ~1M tokens"

    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = conv_states.stride(0)
    stride_istate_dim = conv_states.stride(1)
    stride_istate_token = conv_states.stride(2)
    stride_o_dim = out.stride(0)
    stride_o_token = out.stride(1)

    def grid(META):
        return (
            batch,
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_tiled_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        cu_seqlens,
        out,
        dim,
        cu_seqlen,
        num_cache_lines,
        0,  # stride_x_seq
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        0,  # stride_o_seq
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=True,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # Fixed via H100 sweep (108-config grid; BM16/BN64/w2/s3 won or tied within
        # 5% at every T 128..8192 for dim=8192; upstream vLLM likewise hardcodes).
        BLOCK_M=16,
        BLOCK_N=64,
        num_warps=2,
        num_stages=3,
    )
    return out


def causal_conv1d_decode(
    x: torch.Tensor,                # [batch, conv_dim]
    conv_state: torch.Tensor,       # [num_slots, conv_dim, state_len>=kernel-1] (in place)
    weight: torch.Tensor,           # [conv_dim, kernel]
    conv_state_indices: torch.Tensor,  # [batch] int32
    bias: torch.Tensor | None = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    conv_state_indices = conv_state_indices.to(torch.int32)
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    x = x.unsqueeze(-1)  # [batch, dim, 1]
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    assert 2 <= width <= 4, f"causal_conv1d triton fallback supports width 2..4, got {width}"
    num_cache_lines, _, state_len_full = conv_state.size()
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    out = torch.empty_like(x)
    stride_w_dim, stride_w_width = weight.stride()
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        # Fixed via H100 sweep (36-config grid; BN128/w4 within noise of the winner
        # for bs 1..16, BN256 only +5% at bs32 -- a ~10us launch-bound kernel).
        BLOCK_N=128,
        num_warps=4,
        num_stages=2,
    )
    return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Public fallback API (names/signatures kept exactly as kernel/causal_conv1d.py
# expects; thin adapters over the tuned varlen / decode entrypoints).
# ---------------------------------------------------------------------------
def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Varlen (prefill) depthwise causal conv with fused silu; conv_states updated
    in place; returns a fresh output tensor. Adapts the vendored-op signature to the
    tuned causal_conv1d_varlen.

    query_start_loc: (batch+1) int32 cumulative seqlens.
    seq_lens_cpu: host-side per-request lengths (a Python list) -> batch / max_seq_len
        without a device->host sync.
    """
    batch = len(seq_lens_cpu)
    max_seq_len = int(max(seq_lens_cpu)) if batch else 0
    return causal_conv1d_varlen(
        x,
        weight,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        bias=bias,
        activation=activation,
        pad_slot_id=pad_slot_id,
        max_seq_len=max_seq_len,
        batch=batch,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
) -> torch.Tensor:
    """Single-token (decode) causal conv update with silu; conv_state[idx] shifted
    left + new token appended in place; returns silu(conv). Adapts the vendored-op
    signature to the tuned causal_conv1d_decode.

    Accepts x as (batch, dim) or (batch, dim, 1) and returns the matching rank
    (kernel/causal_conv1d.py passes the 3D form and squeezes the result).
    """
    assert cache_seqlens is None, "circular-buffer cache_seqlens not supported"
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    input_was_3d = x.dim() == 3
    if input_was_3d:
        assert x.shape[-1] == 1, "tuned decode handles a single token per request"
        x = x.squeeze(-1)  # [batch, dim]

    out = causal_conv1d_decode(
        x,
        conv_state,
        weight,
        conv_state_indices,
        bias=bias,
        activation=activation,
        pad_slot_id=pad_slot_id,
    )  # [batch, dim]

    if input_was_3d:
        out = out.unsqueeze(-1)  # [batch, dim, 1]
    return out


__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "causal_conv1d_varlen",
    "causal_conv1d_decode",
    "PAD_SLOT_ID",
]
