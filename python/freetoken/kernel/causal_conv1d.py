"""Thin wrappers over sgl_kernel's fused causal-conv1d ops (borrowed kernel, not a
Layer), used by the GatedDeltaNet conv. These replace a per-request Python loop that
did ``torch.cat([left_state, seg])`` + ``F.conv1d`` (a generic 2D depthwise kernel) —
one fused kernel instead, with the per-request conv state read/updated in place by
slot index (``cache_indices``), so it matches the LinearStatePool layout exactly:
``conv_states[num_slots, conv_dim, kernel-1]``.

Convention mirrors sglang's ``causal_conv1d.py`` sgl_kernel path. silu is applied
inside the kernel; the conv state is updated in place (no separate scatter).
"""
from __future__ import annotations

import torch

_PAD_SLOT_ID = -1


def causal_conv1d_varlen(
    x: torch.Tensor,            # [conv_dim, total_tokens] (channels-first, last dim contiguous)
    weight: torch.Tensor,       # [conv_dim, kernel]
    conv_states: torch.Tensor,  # [num_slots, conv_dim, kernel-1] (updated in place)
    cu_seqlens: torch.Tensor,   # [batch+1] int32 prefix sums of per-request lengths
    cache_indices: torch.Tensor,    # [batch] int32 slot id per request
    has_initial_state: torch.Tensor,  # [batch] bool (carry conv state across chunks)
    max_seq_len: int | None = None,   # host-known longest extend_len; None derives it on device
) -> torch.Tensor:
    """Varlen (prefill) depthwise causal conv with silu; returns silu(conv) and refreshes
    ``conv_states[cache_indices]`` with each request's tail.

    Whether ``x`` survives depends on the backend: sgl_kernel convolves ``x`` in place and
    returns it, while the triton fallback leaves ``x`` untouched and returns a new tensor.
    Callers must use the returned tensor and must not rely on ``x`` afterwards (every
    current call site builds ``x`` fresh and never reads it again).

    ``max_seq_len`` only sizes the triton launch grid. Deriving it from ``cu_seqlens`` costs a
    D2H sync per prefill, which is illegal under CUDA graph capture, so the scheduler passes the
    host value it already has."""
    from freetoken.kernel.backend import is_sgl_kernel_installed

    if not is_sgl_kernel_installed():
        from freetoken.kernel.triton.causal_conv1d_triton import (
            causal_conv1d_varlen as triton_causal_conv1d_varlen,
        )

        return triton_causal_conv1d_varlen(
            x, weight, conv_states, cu_seqlens, cache_indices, has_initial_state,
            max_seq_len=max_seq_len,
        )

    from sgl_kernel import causal_conv1d_fwd

    if x.stride(-1) != 1:
        x = x.contiguous()
    causal_conv1d_fwd(
        x, weight, None, conv_states,
        cu_seqlens.to(torch.int32), cache_indices.to(torch.int32),
        has_initial_state, True, _PAD_SLOT_ID,
    )
    return x


def causal_conv1d_decode(
    x: torch.Tensor,                # [batch, conv_dim] (one token per request)
    conv_state: torch.Tensor,       # [num_slots, conv_dim, state_len>=kernel-1] (in place)
    weight: torch.Tensor,           # [conv_dim, kernel]
    conv_state_indices: torch.Tensor,  # [batch] int32 slot id per request
) -> torch.Tensor:
    """Single-token (decode) causal conv update with silu; shifts+appends the new
    token into ``conv_state[conv_state_indices]`` in place and returns silu(conv)."""
    from freetoken.kernel.backend import is_sgl_kernel_installed

    if not is_sgl_kernel_installed():
        from freetoken.kernel.triton.causal_conv1d_triton import (
            causal_conv1d_decode as triton_causal_conv1d_decode,
        )

        return triton_causal_conv1d_decode(x, conv_state, weight, conv_state_indices)

    from sgl_kernel import causal_conv1d_update

    x = x.unsqueeze(-1)
    causal_conv1d_update(
        x, conv_state, weight, None, True, None,
        conv_state_indices.to(torch.int32), _PAD_SLOT_ID,
    )
    return x.squeeze(-1)
