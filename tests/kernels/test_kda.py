"""Pins for OUR divergences from the upstream-vendored KDA kernels.

The vendored kernels (freetoken/kernel/fla) are tested upstream and not re-tested
here. The eager reference replicates their exact math (safe gate ``gk =
lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))``, ``beta =
sigmoid(beta_raw)``, in-loop q/k l2norm, per-channel-decayed delta rule on a
[V, K] state) so a divergence pin can assert numerics, not just reachability.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, D = 4, 128  # head count trimmed; head_dim matches GLM-5.3 (kernel specializes on D)
LOWER_BOUND = -5.0
SCALE = D**-0.5


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _reference(
    q: torch.Tensor,  # [T, H, D] bf16
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,  # [T, H, D] bf16
    beta_raw: torch.Tensor,  # [T, H] bf16
    a_log: torch.Tensor,  # [H] fp32
    dt_bias: torch.Tensor,  # [H*D] fp32
    h0: torch.Tensor | None = None,  # [H, V, K] fp32
) -> tuple[torch.Tensor, torch.Tensor]:
    T = q.shape[0]
    h = (
        h0.clone().float()
        if h0 is not None
        else torch.zeros(H, D, D, dtype=torch.float32, device=q.device)
    )
    amp = a_log.float().exp().view(H, 1)
    bias = dt_bias.float().view(H, D)
    outs = []
    for t in range(T):
        gk = LOWER_BOUND * torch.sigmoid(amp * (g_raw[t].float() + bias))  # [H, K]
        h = h * gk.exp().unsqueeze(1)  # decay per k-channel: [H, V, K] * [H, 1, K]
        kt = _l2norm(k[t].float())
        v_err = v[t].float() - torch.einsum("hvk,hk->hv", h, kt)
        v_err = v_err * torch.sigmoid(beta_raw[t].float()).unsqueeze(-1)
        h = h + torch.einsum("hv,hk->hvk", v_err, kt)
        qt = _l2norm(q[t].float()) * SCALE
        outs.append(torch.einsum("hvk,hk->hv", h, qt))
    return torch.stack(outs), h


def _rand_inputs(T: int, seed: int = 0, device="cuda"):
    torch.manual_seed(seed)
    mk = lambda *s: torch.randn(*s, device=device, dtype=torch.bfloat16)
    q, k, v, g_raw = mk(T, H, D), mk(T, H, D), mk(T, H, D), mk(T, H, D)
    beta_raw = mk(T, H)
    a_log = torch.randn(H, device=device, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(H * D, device=device, dtype=torch.float32) * 0.5
    return q, k, v, g_raw, beta_raw, a_log, dt_bias


def _assert_close(ours, ref, tag, atol=2e-2, rtol=2e-2):
    ours, ref = ours.float(), ref.float()
    err = (ours - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-8)
    assert torch.allclose(ours, ref, atol=atol, rtol=rtol), (
        f"{tag}: max abs err {err:.5f}, rel {rel:.5f}"
    )


def test_fused_recurrent_serves_slot_zero():
    """--cache-type naive keys state by raw table_idx, so a real request can sit
    on slot 0. Upstream vLLM's kernel treats 0 as its NULL_BLOCK_ID sentinel and
    silently skips it (state frozen, garbage output); our vendored copy diverges
    to accept every non-negative slot (GDN-kernel parity). Same math as the
    parametrized reference test, just on slot 0."""
    from freetoken.kernel.fla import fused_recurrent_kda

    T = 7
    q, k, v, g_raw, beta_raw, a_log, dt_bias = _rand_inputs(T)
    ref_o, ref_h = _reference(q, k, v, g_raw, beta_raw, a_log, dt_bias)

    pool = torch.zeros(2, H, D, D, dtype=torch.float32, device="cuda")
    indices = torch.zeros((1, T), dtype=torch.int64, device="cuda")  # slot 0
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    o, _ = fused_recurrent_kda(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g_raw.unsqueeze(0), beta=beta_raw.unsqueeze(0),
        initial_state=pool,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=dt_bias,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    _assert_close(o[0], ref_o, "slot-0 output")
    _assert_close(pool[0], ref_h, "slot-0 final state")
    assert pool[1].abs().max().item() == 0.0  # only slot 0 was touched
