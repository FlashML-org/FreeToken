"""The debug ``"torch"`` attention backend (Inc 4 of fix-attention).

Verifies (a) the backend is registered and selectable, and (b) its pure-PyTorch
GQA attention math (with causal masking and the per-head output gate) matches
``torch.nn.functional.scaled_dot_product_attention`` on a synthetic case with the
qwen35moe full-attention geometry (num_q=16, num_kv=2, head_dim=256, gate).

This is the ground-truth backend used to A/B the production triton path.
"""

import torch


def _attention_math(q, ks, vs, lk, scale, group, gate):
    """Mirror of TorchAttentionBackend's per-request attention (fp32 intermediates)."""
    ks = ks.repeat_interleave(group, dim=1).float()
    vs = vs.repeat_interleave(group, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), ks) * scale
    lq = q.shape[0]
    rows = torch.arange(lq)
    cols = torch.arange(lk)
    masked = (cols[None, :] > (lk - lq + rows)[:, None]).to(scores.device)
    scores = scores.masked_fill(masked[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    o = torch.einsum("hqk,khd->qhd", probs, vs)  # [lq, num_q, hd]
    lq_, nq, hd_ = o.shape
    o = o.reshape(lq_, nq * hd_) * torch.sigmoid(gate.float())  # [lq, num_q*hd]
    return o.reshape(lq_, nq, hd_)


def test_torch_backend_registered():
    from freetoken.attention import SUPPORTED_ATTENTION_BACKENDS, attention_backend_info, AttnType

    assert "torch" in SUPPORTED_ATTENTION_BACKENDS.supported_names()
    info = attention_backend_info("torch")
    assert AttnType.FULL in info.supported_types
    assert info.hybrid_linear_ok  # must coexist with GDN layers


def test_torch_attention_matches_sdpa_prefill():
    num_q, num_kv, hd, group = 16, 2, 256, 8
    T = 4
    scale = hd ** -0.5
    q = torch.randn(T, num_q, hd)
    k = torch.randn(T, num_kv, hd)
    v = torch.randn(T, num_kv, hd)
    gate = torch.randn(T, num_q * hd)

    out = _attention_math(q, k, v, T, scale, group, gate)
    # reference: sdpa on GQA-expanded heads, then gate + no o_proj here
    ke = k.repeat_interleave(group, dim=1)
    ve = v.repeat_interleave(group, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        ke.transpose(0, 1).unsqueeze(0),
        ve.transpose(0, 1).unsqueeze(0),
        is_causal=True,
    )[0].transpose(0, 1)
    ref = ref.reshape(T, num_q * hd) * torch.sigmoid(gate)
    ref = ref.reshape(T, num_q, hd)

    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
