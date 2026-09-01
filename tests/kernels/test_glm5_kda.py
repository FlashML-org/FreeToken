"""Numerical parity for GLM-5.3's bounded Kimi Delta Attention recurrence."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_source: torch.Tensor,
    state_indices: torch.Tensor,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total, heads, key_dim = q.shape[1:]
    q = q.float()
    k = k.float()
    v = v.float()
    a = a.float().view(total, heads, key_dim)
    dt_bias = dt_bias.float().view(heads, key_dim)
    outputs = torch.empty_like(v)
    expected_states = state_source.float().clone()
    scale = key_dim**-0.5

    for request in range(len(cu_seqlens) - 1):
        begin = int(cu_seqlens[request])
        end = int(cu_seqlens[request + 1])
        slot = int(state_indices[request])
        # Runtime storage is [head, value, key]; recurrence math is [head, key, value].
        state = expected_states[slot].transpose(-1, -2).contiguous()
        for token in range(begin, end):
            q_t = q[0, token]
            k_t = k[0, token]
            q_t = q_t / torch.sqrt((q_t * q_t).sum(-1, keepdim=True) + 1e-6)
            k_t = k_t / torch.sqrt((k_t * k_t).sum(-1, keepdim=True) + 1e-6)
            decay = lower_bound * torch.sigmoid(
                torch.exp(a_log.float()).unsqueeze(-1) * (a[token] + dt_bias)
            )
            state = state * torch.exp(decay).unsqueeze(-1)
            delta = v[0, token] - torch.einsum("hkv,hk->hv", state, k_t)
            delta = delta * torch.sigmoid(beta_logits[token].float()).unsqueeze(-1)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs[0, token] = torch.einsum("hkv,hk->hv", state, q_t * scale)
        expected_states[slot] = state.transpose(-1, -2)
    return outputs, expected_states


@pytest.mark.parametrize("lengths", [(5,), (3, 2)])
def test_glm5_bounded_kda_varlen_matches_reference(lengths):
    from freetoken.models.glm5_next.kda import kda_decode

    torch.manual_seed(53)
    device = torch.device("cuda")
    heads, key_dim, value_dim = 2, 8, 8
    total = sum(lengths)
    q = torch.randn(1, total, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, total, heads, value_dim, device=device, dtype=torch.bfloat16)
    a = torch.randn(total, heads * key_dim, device=device, dtype=torch.bfloat16)
    beta = torch.randn(total, heads, device=device, dtype=torch.bfloat16)
    a_log = torch.randn(heads, device=device, dtype=torch.float32) / 4
    dt_bias = torch.randn(heads * key_dim, device=device, dtype=torch.float32)
    cu = torch.tensor(
        (0, *torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32
    )
    indices = torch.arange(len(lengths), device=device, dtype=torch.int32)
    initial = (
        torch.randn(
            len(lengths), heads, value_dim, key_dim, device=device, dtype=torch.float32
        )
        / 10
    )
    expected_out, expected_state = _reference(
        q, k, v, a, beta, a_log, dt_bias, cu, initial, indices, -5.0
    )

    actual_state = initial.clone()
    actual_out = kda_decode(
        q,
        k,
        v,
        a,
        beta,
        A_log=a_log,
        dt_bias=dt_bias,
        state_source=actual_state,
        indices=indices,
        cu_seqlens=cu,
        gate_lower_bound=-5.0,
    )

    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(actual_state, expected_state, atol=3e-2, rtol=3e-2)
