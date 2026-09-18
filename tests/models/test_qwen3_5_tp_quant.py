"""CPU unit tests for the TP>1 quantized-checkpoint sharding (qwen3_5_moe).

Covers the weight-loader shard math (fused fp8 qkv / in_proj_qkvz with GQA KV-head
replication, block-fp8 per-part + scale sharding) and the row-parallel / column-merged
quantized linear layers (via the pure-torch FREETOKEN_DEBUG_FP8_REF reference path).

Run:  FREETOKEN_DEBUG_FP8_REF=1 pytest tests/models/test_qwen3_5_tp_quant.py
"""

import os

import torch
import torch.distributed as dist

os.environ.setdefault("FREETOKEN_DEBUG_FP8_REF", "1")

import freetoken.distributed.info as _tpinfo  # noqa: E402
from freetoken.distributed.info import DistributedInfo  # noqa: E402


def _set_tp(rank: int, size: int):
    _tpinfo._TP_INFO = DistributedInfo(rank, size)


def _ensure_single_rank_group():
    """Single-process gloo group: makes DistributedCommunicator.all_reduce a no-op
    (world_size==1), so a two-rank TP test can sum the partials by hand."""
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="file:///tmp/ft_tp_quant_test_init",
                                rank=0, world_size=1)


FP8 = torch.float8_e4m3fn


def _rand_fp8(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.1).to(FP8)


# ======================================================================================
# _pt_fp8_fuse (mixed-precision checkpoint: per-tensor FP8 attn/GDN)
# ======================================================================================
def test_pt_fp8_fuse_qkv_shard_tp2():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    # Qwen3.6-35B dims: q 4096 (16 heads x 256 x 2 for the gate), k/v 512 (2 heads x 256).
    K = 2048
    q = _rand_fp8((4096, K), 1)
    k = _rand_fp8((512, K), 2)
    v = _rand_fp8((512, K), 3)
    sq, sk, sv = 0.2, 0.3, 0.4
    act = torch.tensor(0.205)

    local_kv = 256  # 2 KV heads / 2 ranks
    out = {}
    for rank in (0, 1):
        buf = {}
        _set_tp(rank, 2)
        e = _pt_fp8_fuse("model.layers.3.self_attn.q_proj", q, torch.tensor(sq), act,
                         buf, DistributedInfo(rank, 2), (None, local_kv, local_kv))
        assert e == []
        e = _pt_fp8_fuse("model.layers.3.self_attn.k_proj", k, torch.tensor(sk), act,
                         buf, DistributedInfo(rank, 2), (None, local_kv, local_kv))
        assert e == []
        e = _pt_fp8_fuse("model.layers.3.self_attn.v_proj", v, torch.tensor(sv), act,
                         buf, DistributedInfo(rank, 2), (None, local_kv, local_kv))
        assert e is not None and e != []
        out[rank] = dict(e)

    w0, w1 = out[0]["model.layers.3.self_attn.qkv_proj.weight"], \
        out[1]["model.layers.3.self_attn.qkv_proj.weight"]
    assert w0.shape == (2048 + 256 + 256, K) and w1.shape == w0.shape
    torch.testing.assert_close(w0[:2048], q[:2048])
    torch.testing.assert_close(w0[2048:2304], k[:256])
    torch.testing.assert_close(w0[2304:], v[:256])
    torch.testing.assert_close(w1[:2048], q[2048:])
    torch.testing.assert_close(w1[2048:2304], k[256:])
    torch.testing.assert_close(w1[2304:], v[256:])
    # per-row scale: each part's scalar across its local rows
    s0 = out[0]["model.layers.3.self_attn.qkv_proj.weight_scale"]
    assert s0.shape == (2560,)
    torch.testing.assert_close(s0[:2048], torch.full((2048,), sq, dtype=torch.float32))
    torch.testing.assert_close(s0[2048:2304], torch.full((256,), sk, dtype=torch.float32))
    torch.testing.assert_close(s0[2304:], torch.full((256,), sv, dtype=torch.float32))
    # shared input_scale (max) is unchanged
    torch.testing.assert_close(out[0]["model.layers.3.self_attn.qkv_proj.input_scale"],
                               torch.tensor(0.205))


def test_pt_fp8_fuse_qkv_tp1_unchanged():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    q = _rand_fp8((4096, 64), 1)
    k = _rand_fp8((512, 64), 2)
    v = _rand_fp8((512, 64), 3)
    buf = {}
    _set_tp(0, 1)
    e = _pt_fp8_fuse("l.self_attn.q_proj", q, torch.tensor(0.2), None,
                     buf, DistributedInfo(0, 1), None)
    e = _pt_fp8_fuse("l.self_attn.k_proj", k, torch.tensor(0.3), None,
                     buf, DistributedInfo(0, 1), None)
    e = _pt_fp8_fuse("l.self_attn.v_proj", v, torch.tensor(0.4), None,
                     buf, DistributedInfo(0, 1), None)
    d = dict(e)
    torch.testing.assert_close(d["l.self_attn.qkv_proj.weight"],
                               torch.cat([q, k, v], dim=0))


def test_pt_fp8_fuse_qkvz_shard_tp2():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    # GDN: qkv = key_dim(2048) + key_dim(2048) + value_dim(4096); z = value_dim(4096).
    K = 2048
    qkv = _rand_fp8((8192, K), 7)
    z = _rand_fp8((4096, K), 8)
    buf = {}
    _set_tp(1, 2)
    e = _pt_fp8_fuse("l.linear_attn.in_proj_qkv", qkv, torch.tensor(0.1), None,
                     buf, DistributedInfo(1, 2), None)
    assert e == []
    e = _pt_fp8_fuse("l.linear_attn.in_proj_z", z, torch.tensor(0.5), None,
                     buf, DistributedInfo(1, 2), None)
    d = dict(e)
    w = d["l.linear_attn.in_proj_qkvz.weight"]
    assert w.shape == (1024 + 1024 + 2048 + 2048, K)
    torch.testing.assert_close(w[:1024], qkv[1024:2048])        # rank 1: q[1024:2048]
    torch.testing.assert_close(w[1024:2048], qkv[3072:4096])    # k[1024:2048]
    torch.testing.assert_close(w[2048:4096], qkv[4096 + 2048:])  # v[2048:4096]
    torch.testing.assert_close(w[4096:], z[2048:])              # z[2048:4096]
    s = d["l.linear_attn.in_proj_qkvz.weight_scale"]
    torch.testing.assert_close(s[:4096], torch.full((4096,), 0.1, dtype=torch.float32))
    torch.testing.assert_close(s[4096:], torch.full((2048,), 0.5, dtype=torch.float32))


def test_pt_fp8_fuse_qkv_kv_replicate_tp4():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    # 2 KV heads at TP=4 -> each KV head replicated on 2 ranks.
    K = 64
    q = _rand_fp8((4096, K), 1)
    k = _rand_fp8((512, K), 2)
    v = _rand_fp8((512, K), 3)
    local_kv = 256  # 1 head per rank, replicated
    out = {}
    for rank in range(4):
        buf = {}
        _set_tp(rank, 4)
        for name, t, s in (("q_proj", q, 0.2), ("k_proj", k, 0.3), ("v_proj", v, 0.4)):
            e = _pt_fp8_fuse(f"l.self_attn.{name}", t, torch.tensor(s), None,
                             buf, DistributedInfo(rank, 4), (None, local_kv, local_kv))
        out[rank] = dict(e)["l.self_attn.qkv_proj.weight"]
    # local layout: q 4096/4=1024, then k 256, then v 256
    # ranks 0,1 share KV head 0; ranks 2,3 share KV head 1
    torch.testing.assert_close(out[0][1024:1280], out[1][1024:1280])
    torch.testing.assert_close(out[2][1024:1280], out[3][1024:1280])
    torch.testing.assert_close(out[0][1024:1280], k[:256])
    torch.testing.assert_close(out[2][1024:1280], k[256:])
    torch.testing.assert_close(out[0][1280:], out[1][1280:])
    torch.testing.assert_close(out[0][1280:], v[:256])
    # q is a plain /4 split
    torch.testing.assert_close(out[0][:1024], q[:1024])
    torch.testing.assert_close(out[3][:1024], q[3072:])


# ======================================================================================
# _fp8_shard_part (block-fp8 checkpoint: 128x128 block scales)
# ======================================================================================
def test_fp8_shard_part_qkv_weight_and_scale():
    from freetoken.models.qwen3_5_moe.weight import _fp8_shard_part

    class Gdn:
        num_key_heads, key_head_dim = 16, 128
        num_value_heads, value_head_dim = 32, 128

    K, KB = 2048, 2048 // 128
    q = _rand_fp8((4096, K), 1)
    k = _rand_fp8((512, K), 2)
    v = _rand_fp8((512, K), 3)
    qs = torch.randn(4096 // 128, KB)
    ks = torch.randn(512 // 128, KB)
    vs = torch.randn(512 // 128, KB)
    qkv_local = (None, 256, 256)
    tp1 = DistributedInfo(1, 2)

    for suf, parts, locals_ in (
        (".weight", (q, k, v), (2048, 256, 256)),
        (".weight_scale_inv", (qs, ks, vs), (16, 2, 2)),
    ):
        got = [
            _fp8_shard_part(".self_attn.qkv_proj", i, t, suf, tp1, qkv_local, Gdn())
            for i, t in enumerate(parts)
        ]
        for g, t, local in zip(got, parts, locals_):
            assert g.shape[0] == local, (g.shape, local)
    # rank 1 takes the second half of every part
    g = _fp8_shard_part(".self_attn.qkv_proj", 0, q, ".weight", tp1, qkv_local, Gdn())
    torch.testing.assert_close(g, q[2048:])
    g = _fp8_shard_part(".self_attn.qkv_proj", 1, ks, ".weight_scale_inv", tp1, qkv_local, Gdn())
    torch.testing.assert_close(g, ks[2:])  # rank 1: second of two scale blocks


def test_fp8_shard_part_qkvz_expands():
    from freetoken.models.qwen3_5_moe.weight import _fp8_shard_part

    class Gdn:
        num_key_heads, key_head_dim = 16, 128
        num_value_heads, value_head_dim = 32, 128

    K = 2048
    qkv = _rand_fp8((8192, K), 5)
    z = _rand_fp8((4096, K), 6)
    tp0 = DistributedInfo(0, 2)
    w = _fp8_shard_part(".linear_attn.in_proj_qkvz", 0, qkv, ".weight", tp0, None, Gdn())
    assert w.shape == (1024 + 1024 + 2048, K)
    torch.testing.assert_close(w[:1024], qkv[:1024])
    torch.testing.assert_close(w[1024:2048], qkv[2048:3072])
    torch.testing.assert_close(w[2048:], qkv[4096:6144])
    wz = _fp8_shard_part(".linear_attn.in_proj_qkvz", 1, z, ".weight", tp0, None, Gdn())
    torch.testing.assert_close(wz, z[:2048])
    # gate_up groups stay full (replicated model)
    gu = torch.randn(1024, K)
    torch.testing.assert_close(
        _fp8_shard_part(".mlp.shared_expert.gate_up_proj", 0, gu, ".weight", tp0, None, Gdn()),
        gu)


def test_fp8_shard_standalone_vocab_row_conv_tp2():
    from freetoken.models.qwen3_5_moe.weight import _fp8_shard_standalone

    class Gdn:
        num_key_heads, key_head_dim = 16, 128
        num_value_heads, value_head_dim = 32, 128

    # Vocab-parallel: lm_head / embed_tokens shard rows (lm_head was previously yielded
    # full-width against the vocab-sharded ParallelLMHead -> load-time shape assert).
    w = torch.arange(8 * 4, dtype=torch.bfloat16).reshape(8, 4)
    for name in ("lm_head.weight", "model.embed_tokens.weight"):
        torch.testing.assert_close(
            _fp8_shard_standalone(name, w, DistributedInfo(0, 2), None), w[:4])
        torch.testing.assert_close(
            _fp8_shard_standalone(name, w, DistributedInfo(1, 2), None), w[4:])
    # Row-parallel: o_proj / out_proj shard the input (column) dim, weight and scale.
    o = torch.randn(16, 256)
    torch.testing.assert_close(
        _fp8_shard_standalone("model.layers.0.self_attn.o_proj.weight", o,
                              DistributedInfo(1, 2), None), o[:, 128:])
    s = torch.randn(16, 2)
    torch.testing.assert_close(
        _fp8_shard_standalone("model.layers.0.self_attn.o_proj.weight_scale_inv", s,
                              DistributedInfo(1, 2), None), s[:, 1:])
    # Per-head GDN scalars shard dim 0; conv1d shards its (k, k, v) sub-parts.
    a = torch.randn(16)
    torch.testing.assert_close(
        _fp8_shard_standalone("model.layers.0.linear_attn.A_log", a,
                              DistributedInfo(1, 2), None), a[8:])
    conv = torch.randn(2 * 2048 + 4096, 1)
    got = _fp8_shard_standalone("model.layers.0.linear_attn.conv1d.weight", conv,
                                DistributedInfo(0, 2), Gdn())
    assert got.shape == (1024 + 1024 + 2048, 1)
    torch.testing.assert_close(got[:1024], conv[:1024])
    torch.testing.assert_close(got[1024:2048], conv[2048:3072])
    torch.testing.assert_close(got[2048:], conv[4096:6144])
    # Replicated names and TP=1 pass through unchanged.
    norm = torch.randn(4)
    torch.testing.assert_close(
        _fp8_shard_standalone("model.layers.0.self_attn.o_norm.weight", norm,
                              DistributedInfo(0, 2), None), norm)
    torch.testing.assert_close(
        _fp8_shard_standalone("lm_head.weight", w, DistributedInfo(0, 1), None), w)


def test_fp8_iter_weights_tp2_sharding():
    """Full ``_iter_weights_fp8`` run at TP=2 over a tiny synthetic block-FP8 checkpoint.

    Regression: lm_head was yielded full-width (vocab, hidden) against the vocab-sharded
    ParallelLMHead -> load_state_dict shape assert at TP=2 (reported on PR #104)."""
    import tempfile

    from safetensors import safe_open

    from tests.models.tiny_fp8_ckpt import make_tiny_fp8_ckpt
    from freetoken.models.qwen3_5_moe.weight import _iter_weights_fp8

    V = 2048
    with tempfile.TemporaryDirectory() as d:
        ckpt = make_tiny_fp8_ckpt(d, vocab=V, experts=8)
        got = {}
        for rank in (0, 1):
            _set_tp(rank, 2)
            got[rank] = dict(_iter_weights_fp8(
                ckpt, torch.device("cpu"), include_non_moe=True, include_moe_experts=False))
        with safe_open(f"{ckpt}/model.safetensors", framework="pt") as f:
            full_lm_head = f.get_tensor("lm_head.weight")
    for rank, half in ((0, 0), (1, 1)):
        w = got[rank]
        # Vocab-parallel: lm_head / embed_tokens shard rows (the regression).
        assert w["lm_head.weight"].shape == (V // 2, 2048)
        assert w["model.embed_tokens.weight"].shape == (V // 2, 2048)
        torch.testing.assert_close(
            w["lm_head.weight"], full_lm_head[half * (V // 2):(half + 1) * (V // 2)])
        # Row-parallel o_proj / out_proj: input dim sharded (weight + scale).
        assert w["model.layers.3.self_attn.o_proj.weight"].shape == (2048, 2048)
        assert w["model.layers.3.self_attn.o_proj.weight_scale_inv"].shape == (16, 16)
        assert w["model.layers.0.linear_attn.out_proj.weight"].shape == (2048, 2048)
        # Column-merged qkv / in_proj_qkvz: per-part local rows (GQA KV-head halving).
        assert w["model.layers.3.self_attn.qkv_proj.weight"].shape == (4096 + 256 + 256, 2048)
        assert w["model.layers.0.linear_attn.in_proj_qkvz.weight"].shape == (4096 + 2048, 2048)
        # GDN per-head scalars / conv1d (k, k, v) sub-parts.
        assert w["model.layers.0.linear_attn.A_log"].shape == (16,)
        assert w["model.layers.0.linear_attn.conv1d.weight"].shape == (1024 + 1024 + 2048, 1, 4)
        # Replicated: router, shared expert.
        assert w["model.layers.0.mlp.gate.weight"].shape == (8, 2048)
        assert w["model.layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (1024, 2048)
    _set_tp(0, 1)


def test_mixed_iter_weights_tp2_sharding():
    """Full ``_iter_weights_attn_fp8`` run at TP=2 over a tiny synthetic mixed checkpoint.

    Regressions: a bf16 lm_head (lm_head_quant=="none") was yielded full-width against
    the vocab-sharded ParallelLMHead; and NVFP4-native shared_expert must stay replicated
    (weight AND scales unsharded) -- the main-path ``_maybe_shard`` used to shard its
    down_proj while leaving the scales full-width."""
    import tempfile

    from safetensors import safe_open

    from tests.models.tiny_fp8_ckpt import make_tiny_mixed_ckpt
    from freetoken.models.qwen3_5_moe.weight import _iter_weights_attn_fp8

    V = 2048
    with tempfile.TemporaryDirectory() as d:
        ckpt = make_tiny_mixed_ckpt(d, vocab=V, experts=8)
        got = {}
        for rank in (0, 1):
            _set_tp(rank, 2)
            got[rank] = dict(_iter_weights_attn_fp8(
                ckpt, torch.device("cpu"), include_non_moe=True, include_moe_experts=False,
                dense_nvfp4=True, lmhead_nvfp4=False))
        with safe_open(f"{ckpt}/model.safetensors", framework="pt") as f:
            full_lm_head = f.get_tensor("lm_head.weight")
    for rank, half in ((0, 0), (1, 1)):
        w = got[rank]
        # Vocab-parallel: bf16 lm_head / embed_tokens shard rows (the regression).
        assert w["lm_head.weight"].shape == (V // 2, 2048)
        assert w["model.embed_tokens.weight"].shape == (V // 2, 2048)
        torch.testing.assert_close(
            w["lm_head.weight"], full_lm_head[half * (V // 2):(half + 1) * (V // 2)])
        # Row-parallel fp8 o_proj / out_proj: input dim sharded, per-row scale full.
        assert w["model.layers.3.self_attn.o_proj.weight"].shape == (2048, 2048)
        assert w["model.layers.3.self_attn.o_proj.weight_scale"].shape == (2048,)
        assert w["model.layers.0.linear_attn.out_proj.weight"].shape == (2048, 2048)
        # Column-merged fp8 qkv / in_proj_qkvz: per-part local rows (GQA KV-head halving).
        assert w["model.layers.3.self_attn.qkv_proj.weight"].shape == (4096 + 256 + 256, 2048)
        assert w["model.layers.0.linear_attn.in_proj_qkvz.weight"].shape == (4096 + 2048, 2048)
        # GDN bf16: in_proj_ba per-head, conv1d (k, k, v) sub-parts, per-head scalars.
        assert w["model.layers.0.linear_attn.in_proj_ba.weight"].shape == (32, 2048)
        assert w["model.layers.0.linear_attn.conv1d.weight"].shape == (1024 + 1024 + 2048, 1, 4)
        assert w["model.layers.0.linear_attn.A_log"].shape == (16,)
        # NVFP4-native shared_expert stays replicated (weight AND scales unsharded).
        assert w["model.layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (1024, 1024)
        assert w["model.layers.0.mlp.shared_expert.gate_up_proj.weight_scale"].shape == (1024, 128)
        assert w["model.layers.0.mlp.shared_expert.down_proj.weight"].shape == (2048, 256)
        assert w["model.layers.0.mlp.shared_expert.down_proj.weight_scale"].shape == (2048, 32)
        # Replicated: router.
        assert w["model.layers.0.mlp.gate.weight"].shape == (8, 2048)
    _set_tp(0, 1)


def test_resident_nvfp4_expert_banks_tp_sharding():
    """Full ``_iter_weights_attn_fp8`` run with ``include_moe_experts=True`` over a tiny mixed
    checkpoint that carries the routed NVFP4 experts. Verifies the resident (fused) TP-sharded
    bank shapes and that each rank's slice is the correct slice of the checkpoint tensors:
    gate_up is column-parallel (shard the 2*I output rows), down is row-parallel (shard the I
    input cols, a packed dim)."""
    import tempfile

    from safetensors import safe_open

    from tests.models.tiny_fp8_ckpt import make_tiny_mixed_ckpt
    from freetoken.models.qwen3_5_moe.weight import _iter_weights_attn_fp8

    V = 2048
    M, H = 512, 2048  # moe_intermediate_size, hidden (tiny_fp8_ckpt dims)
    with tempfile.TemporaryDirectory() as d:
        ckpt = make_tiny_mixed_ckpt(d, vocab=V, experts=8, routed_experts=True)
        got = {}
        for rank in (0, 1):
            _set_tp(rank, 2)
            got[rank] = dict(_iter_weights_attn_fp8(
                ckpt, torch.device("cpu"), include_non_moe=True, include_moe_experts=True,
                dense_nvfp4=True, lmhead_nvfp4=False))
        with safe_open(f"{ckpt}/model.safetensors", framework="pt") as f:
            # Layer-0 expert-0 checkpoint tensors (ground truth for the sharding).
            gate = f.get_tensor("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
            up = f.get_tensor("model.language_model.layers.0.mlp.experts.0.up_proj.weight")
            down = f.get_tensor("model.language_model.layers.0.mlp.experts.0.down_proj.weight")

    i = M // 2  # per-rank intermediate at TP=2
    for rank in (0, 1):
        w = got[rank]
        pre = "model.layers.0.mlp.experts"
        gu = w[f"{pre}.gate_up_packed"]
        assert gu.shape == (8, 2 * i, H // 2), gu.shape
        assert w[f"{pre}.gate_up_scale"].shape == (8, 2 * i, H // 16)
        assert w[f"{pre}.gate_up_global"].shape == (8, 2 * i)
        assert w[f"{pre}.down_packed"].shape == (8, H, i // 2)
        assert w[f"{pre}.down_scale"].shape == (8, H, i // 16)
        assert w[f"{pre}.down_global"].shape == (8, H)
        # gate_up column-parallel: rank's gate rows [rank*i:(rank+1)*i], up rows likewise.
        torch.testing.assert_close(gu[0, :i], gate[rank * i:(rank + 1) * i])
        torch.testing.assert_close(gu[0, i:], up[rank * i:(rank + 1) * i])
        # down row-parallel: rank's packed cols [rank*(i//2):(rank+1)*(i//2)].
        torch.testing.assert_close(
            w[f"{pre}.down_packed"][0], down[:, rank * (i // 2):(rank + 1) * (i // 2)])
    _set_tp(0, 1)


def test_resident_nvfp4_expert_banks_tp1_full():
    """TP=1 resident NVFP4 banks are the full (unsharded) checkpoint tensors."""
    import tempfile

    from safetensors import safe_open

    from tests.models.tiny_fp8_ckpt import make_tiny_mixed_ckpt
    from freetoken.models.qwen3_5_moe.weight import _iter_weights_attn_fp8

    V, M, H = 2048, 512, 2048
    with tempfile.TemporaryDirectory() as d:
        ckpt = make_tiny_mixed_ckpt(d, vocab=V, experts=8, routed_experts=True)
        _set_tp(0, 1)
        w = dict(_iter_weights_attn_fp8(
            ckpt, torch.device("cpu"), include_non_moe=True, include_moe_experts=True,
            dense_nvfp4=True, lmhead_nvfp4=False))
        with safe_open(f"{ckpt}/model.safetensors", framework="pt") as f:
            gate = f.get_tensor("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
            up = f.get_tensor("model.language_model.layers.0.mlp.experts.0.up_proj.weight")
            down = f.get_tensor("model.language_model.layers.0.mlp.experts.0.down_proj.weight")
    pre = "model.layers.0.mlp.experts"
    assert w[f"{pre}.gate_up_packed"].shape == (8, 2 * M, H // 2)
    assert w[f"{pre}.down_packed"].shape == (8, H, M // 2)
    # Full bank: gate rows [0:M], up rows [M:2M]; down unsharded.
    torch.testing.assert_close(w[f"{pre}.gate_up_packed"][0, :M], gate)
    torch.testing.assert_close(w[f"{pre}.gate_up_packed"][0, M:], up)
    torch.testing.assert_close(w[f"{pre}.down_packed"][0], down)
    _set_tp(0, 1)


# ======================================================================================
# Row-parallel / column-merged quantized linears (pure-torch reference path)
# ======================================================================================
def test_fp8_pertensor_row_parallel_two_rank_sum():
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        Fp8PerTensorLinear, Fp8PerTensorRowParallel,
    )

    N, K = 256, 512
    g = torch.Generator().manual_seed(0)
    w_bf16 = torch.randn(N, K, generator=g) * 0.05
    scale = 0.25
    w_fp8 = (w_bf16 / scale).to(FP8)
    x = torch.randn(7, K, dtype=torch.bfloat16, generator=g)

    full = Fp8PerTensorLinear(K, N)
    full.weight.copy_(w_fp8)
    full.weight_scale.copy_(torch.full((N,), scale))
    full._uniform_scale = True
    ref = full.forward(x)

    partials = []
    _ensure_single_rank_group()
    for rank in (0, 1):
        _set_tp(rank, 2)
        layer = Fp8PerTensorRowParallel(K, N)
        assert layer.weight.shape == (N, K // 2)
        assert layer.weight_scale.shape == (N,)
        shard = w_fp8[:, rank * K // 2:(rank + 1) * K // 2]
        layer.weight.copy_(shard)
        layer.weight_scale.copy_(torch.full((N,), scale))
        layer._uniform_scale = True
        partials.append(layer.forward(x[:, rank * K // 2:(rank + 1) * K // 2]))
    # all_reduce is the identity on the single-rank group: the two partials must sum
    # to the full (replicated) computation.
    torch.testing.assert_close(partials[0] + partials[1], ref, atol=2e-2, rtol=2e-2)


def test_fp8_pertensor_col_merged_local_sizes():
    from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

    _set_tp(1, 2)
    layer = Fp8PerTensorColMerged(2048, [4096, 512, 512],
                                  local_output_sizes=[2048, 256, 256])
    assert layer.weight.shape == (2560, 2048)
    assert layer.weight_scale.shape == (2560,)
    # TP=1 default unchanged
    _set_tp(0, 1)
    full = Fp8PerTensorColMerged(2048, [4096, 512, 512])
    assert full.weight.shape == (5120, 2048)


def test_nvfp4_row_parallel_shapes():
    from freetoken.kernel.triton.nvfp4_linear import (
        Nvfp4DenseColMerged, Nvfp4DenseRowParallel,
    )

    _set_tp(0, 2)
    row = Nvfp4DenseRowParallel(2048, 2048)
    assert row.weight.shape == (2048, 1024 // 2)
    assert row.weight_scale.shape == (2048, 1024 // 16)
    assert row.weight_global.shape == (2048,)
    col = Nvfp4DenseColMerged(2048, [512, 512], local_output_sizes=[256, 256])
    assert col.weight.shape == (512, 2048 // 2)
    assert col.weight_scale.shape == (512, 2048 // 16)
    # load_state_dict round-trips the sharded layout (K-major repack)
    row.load_state_dict({
        "weight": torch.randint(0, 255, (2048, 512), dtype=torch.uint8),
        "weight_scale": torch.randint(1, 10, (2048, 64), dtype=torch.uint8)
        .view(torch.float8_e4m3fn),
        "weight_global": torch.rand(2048, dtype=torch.float16),
    })
    assert row.weight.shape == (1024 // 8, 2048)  # K-major [K//8, N]
    assert row.weight_scale.shape == (1024 // 16, 2048)
