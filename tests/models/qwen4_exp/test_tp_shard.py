"""TP sharding of the qwen4_exp dense weights (pure tensor math, no TP runtime needed)."""

from types import SimpleNamespace

import torch
from freetoken.models.qwen4_exp.weight import _shard, _shard_rows


def _cfg():
    g = SimpleNamespace(
        num_key_heads=4, key_head_dim=8, num_value_heads=12, value_head_dim=8
    )
    return SimpleNamespace(
        num_qo_heads=6, num_kv_heads=2, head_dim=8, linear_attention_group=lambda: g
    )


def _gather(name, t, cfg, world, dim=0):
    return torch.cat([_shard(name, t, cfg, r, world) for r in range(world)], dim=dim)


def test_shard_rows_splits_each_part_by_head():
    t = torch.arange(4 * 4 + 2 * 4).reshape(
        -1, 1
    )  # part A: 4 heads x 4 rows, part B: 2 heads x 4
    s = [_shard_rows(t, [(4, 4), (2, 4)], r, 4) for r in range(4)]
    assert all(x.shape[0] == 8 for x in s)
    assert s[0][:4].flatten().tolist() == [0, 1, 2, 3]
    assert s[3][:4].flatten().tolist() == [12, 13, 14, 15]
    # 2 kv heads over 4 ranks replicate: rank * heads // world -> heads 0, 0, 1, 1
    assert s[0][4:].equal(s[1][4:]) and s[2][4:].flatten().tolist() == [20, 21, 22, 23]


def test_shard_round_trips_the_fused_projections():
    cfg, world = _cfg(), 2
    qkv = torch.randn(6 * 16 + 2 * 8 + 2 * 8, 5)
    assert (
        _gather("model.layers.0.self_attn.qkv_proj.weight", qkv, cfg, world).shape
        == qkv.shape
    )
    q0 = _shard("model.layers.0.self_attn.qkv_proj.weight", qkv, cfg, 0, world)
    assert q0.shape[0] == 3 * 16 + 8 + 8
    torch.testing.assert_close(q0[:48], qkv[:48])  # q heads 0-2 (16 rows each: q|gate)
    torch.testing.assert_close(q0[48:56], qkv[96:104])  # kv head 0 of k
    torch.testing.assert_close(q0[56:64], qkv[112:120])  # kv head 0 of v
    kd, vd, nv = 4 * 8, 12 * 8, 12
    in_proj = torch.randn(kd + kd + vd + vd + nv + nv, 5)
    s = _shard("model.layers.1.linear_attn.in_proj.weight", in_proj, cfg, 1, world)
    assert s.shape[0] == (kd + kd + vd + vd + nv + nv) // 2
    torch.testing.assert_close(s[:16], in_proj[16:32])  # q: k heads 2,3
    torch.testing.assert_close(s[-6:], in_proj[-6:])  # a: v heads 6-11
    conv = torch.randn(kd + kd + vd, 1, 4)
    assert _shard(
        "model.layers.1.linear_attn.conv1d.weight", conv, cfg, 0, world
    ).shape == ((kd + kd + vd) // 2, 1, 4)
    a_log = torch.randn(nv)
    torch.testing.assert_close(
        _shard("model.layers.1.linear_attn.A_log", a_log, cfg, 1, world), a_log[6:]
    )


def test_shard_row_parallel_and_vocab_and_replicated():
    cfg, world = _cfg(), 2
    for name in (
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.linear_attn.out_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
    ):
        t = torch.randn(3, 8)
        torch.testing.assert_close(_gather(name, t, cfg, world, dim=1), t)
    gate_up = torch.randn(2 * 6, 3)
    s1 = _shard(
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight", gate_up, cfg, 1, world
    )
    torch.testing.assert_close(s1, torch.cat([gate_up[3:6], gate_up[9:12]]))
    emb = torch.randn(10, 3)
    torch.testing.assert_close(
        _gather("model.embed_tokens.weight", emb, cfg, world), emb
    )
    torch.testing.assert_close(_gather("lm_head.weight", emb, cfg, world), emb)
    for name in (
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.self_attn.indexer.index_qk_proj.weight",
        "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        "model.layers.1.ple.value_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ):
        t = torch.randn(4, 6)
        assert _shard(name, t, cfg, 1, world).equal(t)
    assert _shard("model.layers.0.self_attn.qkv_proj.weight", gate_up, cfg, 0, 1).equal(
        gate_up
    )
