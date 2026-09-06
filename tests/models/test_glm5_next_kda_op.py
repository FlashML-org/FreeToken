"""Glm5NextKDA op vs an eager reference (projection/conv/gate/norm wiring).

The kernel math itself is validated in tests/kernels/test_kda.py; this test checks
the OP-level wiring: the fused in_proj split (q|k|v|b|f_a|g_a), the merged q|k|v
depthwise causal conv (+silu) against the state pool, the low-rank f/g gates, the
sigmoid-gated output RMSNorm, and prefill -> decode state continuity through
``LinearStatePool``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _single_rank_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

HIDDEN, H, D, KERNEL = 256, 4, 128, 4
P = H * D
LOWER_BOUND = -5.0


def _make_args():
    from freetoken.models.glm5_next.args import Glm5NextArgs

    n_layers = 2
    return Glm5NextArgs(
        hidden_size=HIDDEN, num_heads=8,
        q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=32, qk_rope_head_dim=0,
        v_head_dim=32, mla_nope=True, norm_eps=1e-5, max_position=4096,
        index_n_heads=0, index_head_dim=0, index_topk=0, indexer_types=(),
        indexer_rope_interleave=True, index_kpool=1, index_kpool_compress=False,
        index_kpool_always_select_tail=False,
        linear_num_heads=H, linear_head_dim=D, linear_conv_kernel_dim=KERNEL,
        linear_lower_bound=LOWER_BOUND,
        layer_types=("linear_attention",) * n_layers,
        mlp_layer_types=("dense",) * n_layers,
        mhc=False, mhc_num_residual_streams=1, hc_eps=1e-6,
        mhc_sinkhorn_iterations=0, mhc_tau=0.05, mhc_post_mult_value=2.0,
        mhc_no_norm_weight=False, swiglu_limit=None, rope_theta=10000.0,
    )


def _make_op(seed=0):
    from freetoken.models.glm5_next.kda import Glm5NextKDA

    cfg = SimpleNamespace(glm5_args=_make_args(), attn_quant="none")
    op = Glm5NextKDA(cfg, layer_id=0)
    torch.manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    op.in_proj.weight = torch.randn(3 * P + H + 2 * D, HIDDEN, device=dev, dtype=dt) * 0.05
    op.f_b_proj.weight = torch.randn(P, D, device=dev, dtype=dt) * 0.05
    op.g_b_proj.weight = torch.randn(P, D, device=dev, dtype=dt) * 0.05
    op.conv1d.weight = torch.randn(3 * P, 1, KERNEL, device=dev, dtype=dt) * 0.2
    op.A_log = torch.randn(H, device=dev, dtype=torch.float32) * 0.5
    op.dt_bias = torch.randn(P, device=dev, dtype=torch.float32) * 0.5
    op.o_norm.weight = torch.randn(D, device=dev, dtype=dt) * 0.1 + 1.0
    op.o_proj.weight = torch.randn(HIDDEN, P, device=dev, dtype=dt) * 0.05
    return op


def _make_pool(num_slots=4):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,),
        num_key_heads=H, num_value_heads=H, key_head_dim=D, value_head_dim=D,
        conv_kernel_dim=KERNEL, output_gate=True, variant="kda",
    )
    return LinearStatePool(
        group, num_slots, dtype=torch.bfloat16,
        device=torch.device("cuda"), tp_size=1,
    )


def _patch_ctx(monkeypatch, pool, batch):
    ctx = SimpleNamespace(batch=batch, linear_state_pool=pool)
    monkeypatch.setattr(
        "freetoken.models.glm5_next.kda.get_global_ctx", lambda: ctx
    )


def _l2norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _reference_forward(op, x_seq, conv_ctx=None, h0=None):
    """Eager op reference for one sequence [T, HIDDEN] (fp32 where the kernels are
    fp32). Returns (out [T, HIDDEN], conv_tail [3P, KERNEL-1], state [H, D, D])."""
    T = x_seq.shape[0]
    proj = x_seq.to(torch.bfloat16) @ op.in_proj.weight.T
    conv_in, b, f_a, g_a = torch.split(proj, [3 * P, H, D, D], dim=-1)
    g1 = (f_a @ op.f_b_proj.weight.T).float()
    g2 = g_a @ op.g_b_proj.weight.T

    # depthwise causal conv + silu over the merged q|k|v stream, with optional
    # left-context from a previous chunk (conv state semantics).
    w = op.conv1d.weight.squeeze(1).float()  # [3P, KERNEL]
    stream = conv_in.T.float()  # [3P, T]
    left = (
        conv_ctx.float()
        if conv_ctx is not None
        else torch.zeros(3 * P, KERNEL - 1, device=x_seq.device)
    )
    padded = torch.cat([left, stream], dim=1)  # [3P, KERNEL-1+T]
    conv = torch.stack(
        [(padded[:, t : t + KERNEL] * w).sum(-1) for t in range(T)], dim=1
    )
    mixed = torch.nn.functional.silu(conv).T  # [T, 3P]
    conv_tail = padded[:, -(KERNEL - 1):]

    q, k, v = (t.reshape(T, H, D) for t in torch.split(mixed, [P, P, P], dim=-1))
    # bf16 round-trip like the op (kernel inputs are bf16)
    q, k, v = q.to(torch.bfloat16).float(), k.to(torch.bfloat16).float(), v.to(torch.bfloat16).float()

    h = h0.clone() if h0 is not None else torch.zeros(H, D, D, device=x_seq.device)
    amp = op.A_log.exp().view(H, 1)
    bias = op.dt_bias.view(H, D)
    core = []
    for t in range(T):
        gk = LOWER_BOUND * torch.sigmoid(amp * (g1[t].view(H, D) + bias))
        h = h * gk.exp().unsqueeze(1)
        kt = _l2norm(k[t])
        v_err = (v[t] - torch.einsum("hvk,hk->hv", h, kt)) * torch.sigmoid(
            b[t].float()
        ).unsqueeze(-1)
        h = h + torch.einsum("hv,hk->hvk", v_err, kt)
        core.append(torch.einsum("hvk,hk->hv", h, _l2norm(q[t]) * D**-0.5))
    core = torch.stack(core)  # [T, H, D]

    xn = core.reshape(-1, D)
    rms = xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + op.o_norm.eps)
    gated = rms * op.o_norm.weight.float() * torch.sigmoid(g2.reshape(-1, D).float())
    out = gated.reshape(T, P).to(torch.bfloat16) @ op.o_proj.weight.T
    return out.float(), conv_tail, h


def _fla(cu, indices, has_init=None, fresh=None):
    from freetoken.attention.linear import FLAMetadata

    return FLAMetadata(
        cu_seqlens=cu, cache_indices=indices,
        has_initial_state=has_init, fresh_state_indices=fresh,
    )


def _assert_close(ours, ref, tag, atol=3e-2):
    err = (ours.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item() + 1e-8
    assert err / scale < atol, f"{tag}: max abs err {err:.5f} (ref scale {scale:.3f})"


def test_prefill_matches_reference(monkeypatch):
    op = _make_op()
    pool = _make_pool()
    lens = [33, 70]
    total = sum(lens)
    torch.manual_seed(10)
    x = torch.randn(total, HIDDEN, device="cuda", dtype=torch.bfloat16)

    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    has_init = torch.tensor([False, False], device="cuda")
    fresh = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    batch = SimpleNamespace(is_decode=False, fla_metadata=_fla(cu, indices, has_init, fresh))
    _patch_ctx(monkeypatch, pool, batch)

    out = op.forward(x)

    start = 0
    for i, ln in enumerate(lens):
        sl = slice(start, start + ln)
        ref_out, ref_conv, ref_h = _reference_forward(op, x[sl].float())
        _assert_close(out[sl], ref_out, f"seq{i} prefill out")
        _assert_close(pool.recurrent_states[0, i + 1], ref_h, f"seq{i} state")
        _assert_close(pool.conv_states[0, i + 1], ref_conv, f"seq{i} conv state")
        start += ln


def test_prefill_then_decode_continuity(monkeypatch):
    op = _make_op(seed=1)
    pool = _make_pool()
    T0, T1 = 40, 3
    torch.manual_seed(11)
    x = torch.randn(T0 + T1, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref_out, _, _ = _reference_forward(op, x.float())

    cu = torch.tensor([0, T0], dtype=torch.int32, device="cuda")
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    batch = SimpleNamespace(
        is_decode=False,
        fla_metadata=_fla(
            cu, indices,
            torch.tensor([False], device="cuda"),
            torch.tensor([1], dtype=torch.int64, device="cuda"),
        ),
    )
    _patch_ctx(monkeypatch, pool, batch)
    out0 = op.forward(x[:T0])
    _assert_close(out0, ref_out[:T0], "prefill out")

    for t in range(T0, T0 + T1):
        batch = SimpleNamespace(
            is_decode=True,
            fla_metadata=_fla(
                torch.tensor([0, 1], dtype=torch.int32, device="cuda"), indices
            ),
        )
        _patch_ctx(monkeypatch, pool, batch)
        out_t = op.forward(x[t : t + 1])
        _assert_close(out_t[0], ref_out[t], f"decode token {t}")


def test_chunked_prefill_continuation(monkeypatch):
    """Second prefill chunk with has_initial_state=True must continue conv AND
    recurrent state exactly (the chunked-prefill path)."""
    op = _make_op(seed=2)
    pool = _make_pool()
    T0, T1 = 64, 30
    torch.manual_seed(12)
    x = torch.randn(T0 + T1, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref_out, ref_conv, ref_h = _reference_forward(op, x.float())

    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    batch = SimpleNamespace(
        is_decode=False,
        fla_metadata=_fla(
            torch.tensor([0, T0], dtype=torch.int32, device="cuda"), indices,
            torch.tensor([False], device="cuda"),
            torch.tensor([1], dtype=torch.int64, device="cuda"),
        ),
    )
    _patch_ctx(monkeypatch, pool, batch)
    out0 = op.forward(x[:T0])
    _assert_close(out0, ref_out[:T0], "chunk0 out")

    batch = SimpleNamespace(
        is_decode=False,
        fla_metadata=_fla(
            torch.tensor([0, T1], dtype=torch.int32, device="cuda"), indices,
            torch.tensor([True], device="cuda"), None,
        ),
    )
    _patch_ctx(monkeypatch, pool, batch)
    out1 = op.forward(x[T0:])
    _assert_close(out1, ref_out[T0:], "chunk1 out")
    _assert_close(pool.recurrent_states[0, 1], ref_h, "final state")
    _assert_close(pool.conv_states[0, 1], ref_conv, "final conv state")
