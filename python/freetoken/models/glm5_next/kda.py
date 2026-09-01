"""GLM-5.3 Kimi Delta Attention recurrence adapters."""

from __future__ import annotations

import torch


def kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_a: torch.Tensor,
    beta_logits: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_source: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    gate_lower_bound: float | None = -5.0,
) -> torch.Tensor:
    """Fused vector-decay KDA update with indexed in-place recurrent state.

    ``gate_a`` and ``dt_bias`` are flattened ``[..., heads * key_dim]`` tensors,
    matching GLM's ``f_b_proj`` output and checkpoint parameter respectively.
    """
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=gate_a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=beta_logits,
        initial_state_source=state_source,
        initial_state_indices=indices,
        scale=q.shape[-1] ** -0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        is_kda=True,
        gate_lower_bound=gate_lower_bound,
    )


__all__ = ["kda_decode"]
