"""The verify branch shared by the GatedDeltaNet ops: a uniform extend of ``steps`` rows per request that only reads the live state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.attention.linear import FLAMetadata
    from freetoken.kvcache.linear_state_pool import LinearStatePool, LinearVerifyScratch


def gdn_verify_mix(
    op,
    conv_in: torch.Tensor,   # [total, conv_dim], total = bs * steps
    a: torch.Tensor,         # [total, num_v_heads] raw
    b: torch.Tensor,         # [total, num_v_heads] raw
    *,
    pool: LinearStatePool,
    li: int,
    fla: FLAMetadata,
    scratch: LinearVerifyScratch,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Core output [total, num_v_heads, head_v_dim] of GatedDeltaNet ``op`` (reads its dims, conv weight, A_log and dt_bias).

    The conv rolls the live window as if every row were accepted and parks each row's window in ``scratch``; the
    recurrence leaves the live state untouched and stashes each row's inputs there. ``LinearVerifyScratch.commit``
    then rebuilds both from the accepted rows.
    """
    from freetoken.kernel.causal_conv1d import causal_conv1d_verify
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    total = conv_in.shape[0]
    bs = fla.cu_seqlens.numel() - 1
    steps = total // bs
    # free views both ways: the conv reads any strides and writes token-major output
    mixed = causal_conv1d_verify(
        conv_in.reshape(bs, steps, -1).transpose(1, 2), pool.conv_states[li], op._conv_weight(),
        fla.cache_indices, scratch.conv_window[li], fla.intermediate_state_indices,
    ).transpose(1, 2).reshape(total, -1)
    qf, kf, vf = torch.split(mixed, [op.key_dim, op.key_dim, op.value_dim], dim=-1)
    q = qf.reshape(1, total, op.num_k_heads, op.head_k_dim).to(dtype)
    k = kf.reshape(1, total, op.num_k_heads, op.head_k_dim).to(dtype)
    v = vf.reshape(1, total, op.num_v_heads, op.head_v_dim).to(dtype)
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=op.A_log, a=a, dt_bias=op.dt_bias,
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=pool.recurrent_states[li],
        initial_state_indices=fla.cache_indices,
        scale=op.head_k_dim ** -0.5, use_qk_l2norm_in_kernel=True, cu_seqlens=fla.cu_seqlens,
        disable_state_update=True,
        input_stash=scratch.input_stash(li),
        intermediate_state_indices=fla.intermediate_state_indices,
    )
    return o[0]
