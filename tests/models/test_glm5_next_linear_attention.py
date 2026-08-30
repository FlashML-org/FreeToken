from types import SimpleNamespace

import torch


class _Op:
    def __init__(self, fn):
        self._fn = fn

    def forward(self, *args):
        return self._fn(*args)


def test_glm5_kda_prefill_materializes_token_major_qkv(monkeypatch):
    """KDA indexes key dimensions as contiguous after the channel-first conv."""
    import freetoken.models.glm5_next.linear_attention as module

    total, heads, head_dim = 4, 2, 3
    qkv_dim = heads * head_dim
    projected = [
        torch.arange(total * qkv_dim, dtype=torch.float32).view(total, qkv_dim)
        + 100 * i
        for i in range(3)
    ]

    op = module.Glm5NextLinearAttention.__new__(module.Glm5NextLinearAttention)
    op.layer_id = 0
    op.num_heads = heads
    op.head_dim = head_dim
    op.qkv_dim = qkv_dim
    op.conv_dim = 3 * qkv_dim
    op.conv_kernel_size = 4
    op.gate_lower_bound = -5.0
    op.q_proj, op.k_proj, op.v_proj = (_Op(lambda _x, y=y: y) for y in projected)
    op.conv1d = SimpleNamespace(weight=torch.ones(op.conv_dim, 1, 4))
    op.f_a_proj = _Op(lambda x: torch.zeros(total, head_dim))
    op.f_b_proj = _Op(lambda x: torch.zeros(total, qkv_dim))
    op.b_proj = _Op(lambda x: torch.zeros(total, heads))
    op.g_a_proj = _Op(lambda x: torch.zeros(total, head_dim))
    op.g_b_proj = _Op(lambda x: torch.zeros(total, qkv_dim))
    op.A_log = torch.zeros(heads)
    op.dt_bias = torch.zeros(qkv_dim)
    op.o_norm = _Op(lambda x, gate: x)
    op.o_proj = _Op(lambda x: x)

    fla = SimpleNamespace(
        cu_seqlens=torch.tensor([0, total], dtype=torch.int32),
        cache_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        fresh_state_indices=None,
    )
    batch = SimpleNamespace(is_decode=False, fla_metadata=fla)
    pool = SimpleNamespace(
        local_index=lambda _layer: 0,
        conv_states=[torch.zeros(1, op.conv_dim, 3)],
        recurrent_states=[torch.zeros(1, heads, head_dim, head_dim)],
    )
    monkeypatch.setattr(
        module,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=batch, linear_state_pool=pool),
    )
    # The production conv returns channel-major storage. Its transpose has a
    # non-unit feature stride until the model materializes token-major storage.
    monkeypatch.setattr(
        module,
        "causal_conv1d_varlen",
        lambda x, *_args: x,
    )

    def capture_kda(q, k, v, *_args, **_kwargs):
        for tensor in (q, k, v):
            assert tensor.stride(-1) == 1
            assert tensor.stride(-2) == head_dim
        torch.testing.assert_close(q.reshape(total, qkv_dim), projected[0])
        torch.testing.assert_close(k.reshape(total, qkv_dim), projected[1])
        torch.testing.assert_close(v.reshape(total, qkv_dim), projected[2])
        return torch.zeros(1, total, heads, head_dim)

    monkeypatch.setattr(module, "kda_decode", capture_kda)
    result = op.forward(torch.zeros(total, 8))
    assert result.shape == (total, qkv_dim)
