"""Vendored from sglang (flash-linear-attention) — fused sigmoid-gating gated-delta-rule
decode kernel. Borrowed verbatim (pure torch+triton, no sglang deps) from sglang's
``python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py``.
Does gating(sigmoid+softplus) + optional in-kernel l2norm + delta-rule recurrent update +
per-request state read/write-by-index in ONE kernel — no external gating or
gather/scatter/clone glue.

FreeToken additions: the verify mode can stash each token's recurrence inputs (STASH_INPUTS), and
``gdn_state_replay`` re-runs the accepted prefix from that stash to commit the live state."""
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    stash_k,
    stash_v,
    stash_g,
    stash_beta,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    STASH_INPUTS: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)

    if STASH_INPUTS:
        # the stash is indexed by batch row
        tok0 = i_n.to(tl.int64) * cache_steps
        ps_k = stash_k + (tok0 * H + i_h) * K + o_k
        ps_v = stash_v + (tok0 * HV + i_hv) * V + o_v
        ps_g = stash_g + tok0 * HV + i_hv
        ps_beta = stash_beta + tok0 * HV + i_hv
        mask_sk = mask_k & (i_v == 0) & (i_hv % (HV // H) == 0)
        mask_sv = mask_v
        mask_sg = i_v == 0

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        if STASH_INPUTS:
            b_v_in = b_v
        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        # stored after the step's math: an earlier store stalls the in-order issue on its operands
        if STASH_INPUTS:
            tl.store(ps_k, b_k, mask=mask_sk)
            tl.store(ps_v, b_v_in, mask=mask_sv)
            tl.store(ps_g, b_g, mask=mask_sg)
            tl.store(ps_beta, b_beta, mask=mask_sg)
            ps_k += H * K
            ps_v += HV * V
            ps_g += HV
            ps_beta += HV

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[
        int
    ] = None,  # kept for API compat; stride is derived from ``intermediate_states_buffer.shape[1]``
    retrieve_parent_token: Optional[torch.Tensor] = None,
    input_stash: Optional[tuple] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable

    ``input_stash`` = (k, v, g, beta) fp32 buffers of shape [rows, steps, H, K] / [rows, steps, HV, V] /
    [rows, steps, HV] / [rows, steps, HV], indexed by batch row: each token's l2-normed k,
    raw v and gates, for ``gdn_state_replay``.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    stride_a = a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    grid = (NK, NV, N * HV)

    # Per-req stride must match the buffer's allocated dim, not runtime steps
    # (they can differ under --speculative-adaptive).
    cache_stride_steps = (
        intermediate_states_buffer.shape[1]
        if intermediate_states_buffer is not None
        else 0
    )
    stash_k = stash_v = stash_g = stash_beta = None
    if input_stash is not None:
        assert not is_kda
        stash_k, stash_v, stash_g, stash_beta = input_stash
        assert all(t.dtype == torch.float32 and t.is_contiguous() for t in input_stash)
        assert stash_k.shape[2:] == (H, K) and stash_v.shape[2:] == (HV, V)
        cache_stride_steps = stash_k.shape[1]

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        stash_k=stash_k,
        stash_v=stash_v,
        stash_g=stash_g,
        stash_beta=stash_beta,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        STASH_INPUTS=input_stash is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o


@triton.jit(do_not_specialize=["R"])
def gdn_state_replay_kernel(
    state,
    stash_k,
    stash_v,
    stash_g,
    stash_beta,
    meta,
    R,
    ROWS: tl.constexpr,
    STEPS: tl.constexpr,
    SLOTS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """One (V block, committed row, layer x value head) per program; the state update is the verify kernel's, op for op."""
    i_v, i_r, i_x = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_hv = i_x % HV
    i_l = i_x // HV
    i_h = i_hv // (HV // H)
    row = tl.load(meta + i_r).to(tl.int64)
    slot = tl.load(meta + R + i_r).to(tl.int64)
    n = tl.load(meta + 2 * R + i_r)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    if slot >= 0:
        p_h = state + ((i_l * SLOTS + slot) * HV + i_hv) * K * V + o_v[None, :] * K + o_k[:, None]
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        tok = (i_l * ROWS + row) * STEPS
        p_k = stash_k + (tok * H + i_h) * K + o_k
        p_v = stash_v + (tok * HV + i_hv) * V + o_v
        p_g = stash_g + tok * HV + i_hv
        p_beta = stash_beta + tok * HV + i_hv
        for _ in range(0, n):
            b_k = tl.load(p_k, mask=mask_k, other=0)
            b_v = tl.load(p_v, mask=mask_v, other=0)
            b_g = tl.load(p_g)
            b_beta = tl.load(p_beta)
            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]
            p_k += H * K
            p_v += HV * V
            p_g += HV
            p_beta += HV
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def gdn_state_replay(
    state: torch.Tensor,
    input_stash: tuple,
    meta: torch.Tensor,
) -> None:
    """Advance live GDN states in place over each committed row's first n stashed verify tokens, every layer in one launch.

    ``state`` is the pool slab [L, slots, HV, K, V] fp32 ([V, K] raw inner layout, as the fused kernel addresses it);
    ``input_stash`` holds [L, rows, steps, ...] buffers the verify filled; ``meta`` is int32 [3, R]: verify row, live
    slot (-1 skips), token count. BV and num_warps match the verify launch so the state equals the verify's bit for bit.
    """
    stash_k, stash_v, stash_g, stash_beta = input_stash
    L, slots, HV, K, V = state.shape
    _, rows, steps, H, _ = stash_k.shape
    R = meta.shape[1]
    assert state.is_contiguous() and meta.dtype == torch.int32 and meta.is_contiguous()
    if R == 0:
        return
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    # rows and layers x heads on separate grid axes, as SGLang's ReplaySSM fold lays them out: one axis holds at most 65535 programs
    gdn_state_replay_kernel[(triton.cdiv(V, BV), R, L * HV)](
        state, stash_k, stash_v, stash_g, stash_beta, meta, R,
        ROWS=rows, STEPS=steps, SLOTS=slots, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        num_warps=1, num_stages=3,
    )
