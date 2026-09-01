from __future__ import annotations

import pytest
import torch


def _release_recurrence(q, k, v, gate_a, beta_logits, A_log, dt_bias, state):
    q = q.float()
    k = k.float()
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    beta = beta_logits.float().sigmoid()
    gate_a = gate_a.float().reshape(*q.shape[:-1], q.shape[-1])
    g = -5.0 * torch.sigmoid(A_log.float().view(1, 1, -1, 1).exp() * (gate_a + dt_bias))
    outputs = []
    state = state.float().clone()
    for t in range(q.shape[1]):
        state *= g[:, t].exp().unsqueeze(-1)
        remembered = (state * k[:, t].unsqueeze(-1)).sum(-2)
        delta = (v[:, t].float() - remembered) * beta[:, t].unsqueeze(-1)
        state += k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q[:, t].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=1).to(q.dtype), state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_kda_decode_matches_release_recurrence_on_ampere():
    from freetoken.models.glm5_next.kda import kda_decode

    torch.manual_seed(530)
    device = torch.device("cuda")
    batch, tokens, heads, dim = 1, 3, 2, 128
    q = torch.randn(batch, tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate_a = torch.randn(batch, tokens, heads * dim, device=device, dtype=torch.bfloat16)
    beta_logits = torch.randn(batch, tokens, heads, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(heads * dim, device=device, dtype=torch.float32) * 0.1
    initial = torch.randn(batch, heads, dim, dim, device=device, dtype=torch.float32) * 0.01

    # The vendored recurrent kernel stores each KxV state transposed for vectorized V tiles.
    state_source = initial.transpose(-1, -2).contiguous()
    got = kda_decode(
        q,
        k,
        v,
        gate_a,
        beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_source=state_source,
        indices=torch.tensor([0], device=device, dtype=torch.int32),
    )
    expected, expected_state = _release_recurrence(
        q, k, v, gate_a, beta_logits, A_log, dt_bias.view(1, 1, heads, dim), initial
    )
    torch.testing.assert_close(got, expected.to(got.dtype), rtol=0, atol=0.02)
    torch.testing.assert_close(
        state_source.transpose(-1, -2), expected_state, rtol=3e-4, atol=3e-4
    )
