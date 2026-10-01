"""mimo_v2 numerics guards: pure-torch, CPU-only, no kernels and no GPU context.

Pins the four pieces of contract math that are checkable without CUDA:
- V preprocessing (scale 0.707 -> per-head pad to 192, rank-2 flat for the {L, D}
  KV-pool store) via ``pad_value_heads``;
- the O back-slice (per-head ``[..., :128]``, head-major) via ``unpad_attention_output``;
- the noaux_tc router numerics (contract SECTION 3.4) via ``MiMoV2SparseBlock._route``
  called unbound with a stand-in ``self`` (no layers are constructed);
- the dual-rope table dispatch (full theta 1e7 / swa theta 1e4, rotary_dim 64)
  through ``parse_config`` on a toy config.
"""

from __future__ import annotations

import types

import pytest
import torch

from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
from freetoken.models.mimo_v2.attention import pad_value_heads, unpad_attention_output
from freetoken.models.mimo_v2.config import parse_config
from freetoken.models.mimo_v2.moe import MiMoV2SparseBlock
from freetoken.utils.hf import RawConfigShim

V_HEAD_DIM = 128
HEAD_DIM = 192
VALUE_SCALE = 0.707


# ---------------------------------------------------------------- V transform


def test_pad_value_heads_scale_then_pad_math():
    torch.manual_seed(0)
    T, KV = 5, 3
    v = torch.randn(T, KV * V_HEAD_DIM)
    out = pad_value_heads(v, KV, V_HEAD_DIM, HEAD_DIM, VALUE_SCALE)

    # rank-2 flat [T, kv*192]: the {L, D} layout store.cu:79 matches (mha_pool.py:130)
    assert out.ndim == 2 and out.shape == (T, KV * HEAD_DIM)
    assert out.is_contiguous()

    heads = out.view(T, KV, HEAD_DIM)
    assert torch.equal(heads[..., :V_HEAD_DIM], v.view(T, KV, V_HEAD_DIM) * VALUE_SCALE)
    assert torch.equal(heads[..., V_HEAD_DIM:], torch.zeros(T, KV, HEAD_DIM - V_HEAD_DIM))
    # scale is applied BEFORE the pad, so no padded lane can ever be non-zero
    assert torch.count_nonzero(heads[..., V_HEAD_DIM:]) == 0


def test_pad_value_heads_accepts_a_noncontiguous_qkv_split_slice():
    """forward hands it the V slice of the fused qkv split, exactly as the model does."""
    torch.manual_seed(1)
    T, KV = 4, 2
    Q_ROWS, K_ROWS = 8 * HEAD_DIM, KV * HEAD_DIM
    qkv = torch.randn(T, Q_ROWS + K_ROWS + KV * V_HEAD_DIM)
    _q, _k, v_slice = qkv.split([Q_ROWS, K_ROWS, KV * V_HEAD_DIM], dim=-1)
    assert not v_slice.is_contiguous()
    assert torch.equal(
        pad_value_heads(v_slice, KV, V_HEAD_DIM, HEAD_DIM, VALUE_SCALE),
        pad_value_heads(v_slice.contiguous(), KV, V_HEAD_DIM, HEAD_DIM, VALUE_SCALE),
    )


def test_pad_value_heads_none_scale_is_identity():
    v = torch.randn(3, 2 * V_HEAD_DIM)
    out = pad_value_heads(v, 2, V_HEAD_DIM, HEAD_DIM, None).view(3, 2, HEAD_DIM)
    assert torch.equal(out[..., :V_HEAD_DIM], v.view(3, 2, V_HEAD_DIM))
    assert torch.count_nonzero(out[..., V_HEAD_DIM:]) == 0


# ---------------------------------------------------------------- O back-slice


def test_unpad_attention_output_is_per_head_head_major_not_flat():
    T, QO = 3, 6
    o = torch.arange(T * QO * HEAD_DIM, dtype=torch.float32).view(T, QO, HEAD_DIM)
    out = unpad_attention_output(o, QO, HEAD_DIM, V_HEAD_DIM)

    assert out.ndim == 2 and out.shape == (T, QO * V_HEAD_DIM)
    # element-exact: flat index t*QO*128 + h*128 + d must equal o[t, h, d]
    assert torch.equal(out.view(T, QO, V_HEAD_DIM), o[..., :V_HEAD_DIM])
    for t in range(T):
        for h in range(QO):
            assert torch.equal(
                out[t, h * V_HEAD_DIM : (h + 1) * V_HEAD_DIM], o[t, h, :V_HEAD_DIM]
            )
    # the WRONG convention (slice the first 768 of the flat 1152 per token) must differ
    assert not torch.equal(out, o.reshape(T, -1)[:, : QO * V_HEAD_DIM])


def test_unpad_attention_output_matches_a_rank2_backend_return():
    T, QO = 2, 4
    o3 = torch.randn(T, QO, HEAD_DIM)
    assert torch.equal(
        unpad_attention_output(o3, QO, HEAD_DIM, V_HEAD_DIM),
        unpad_attention_output(o3.reshape(T, QO * HEAD_DIM), QO, HEAD_DIM, V_HEAD_DIM),
    )


# ---------------------------------------------------------------- router (§3.4)


def _toy_router(top_k: int, bias: torch.Tensor, weight: torch.Tensor) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        top_k=top_k,
        num_experts=weight.shape[0],
        norm_topk_prob=True,
        routed_scaling_factor=1.0,  # config None => HF default 1.0
        n_group=1,
        topk_group=1,
        gate=types.SimpleNamespace(weight=weight),
        e_score_correction_bias=bias,
    )


def test_router_biased_selection_unbiased_weights_renorm_and_dtypes():
    torch.manual_seed(2)
    M, H, E, K = 7, 16, 8, 3
    hidden = torch.randn(M, H)
    gate_w = torch.randn(E, H)
    # bias strong enough that biased top-k differs from unbiased top-k on row 0
    bias = torch.zeros(E, dtype=torch.float32)
    bias[1] = -10.0
    bias[2] = 10.0

    block = _toy_router(K, bias, gate_w)
    weights, ids = MiMoV2SparseBlock._route(block, hidden)

    assert ids.dtype is torch.int32 and ids.is_contiguous()
    assert weights.dtype is torch.float32 and weights.is_contiguous()
    assert ids.shape == (M, K) and weights.shape == (M, K)

    scores = torch.sigmoid(hidden.float() @ gate_w.float().T)
    choice = scores + bias  # bias AFTER sigmoid, selection only
    for row in range(M):
        # selection order: exactly the biased top-k set (topk sorted=False order is free)
        assert set(ids[row].tolist()) == set(
            torch.topk(choice[row], K).indices.tolist()
        )
        # weights: UNBIASED sigmoid scores gathered at the biased ids, renormalized
        chosen = scores[row, ids[row]]
        expected = chosen / (chosen.sum() + 1e-20)  # then * routed_scaling_factor 1.0
        assert torch.allclose(weights[row], expected, rtol=0, atol=1e-7)
        assert torch.allclose(weights[row].sum(), torch.tensor(1.0), atol=1e-6)

    # the bias really steered selection on the row whose raw ranking it flips
    biased_top = set(ids[0].tolist())
    unbiased_top = set(torch.topk(scores[0], K).indices.tolist())
    assert biased_top != unbiased_top


def test_router_scaling_factor_multiplies_renormalized_weights():
    torch.manual_seed(3)
    M, H, E, K = 4, 8, 6, 2
    hidden, gate_w = torch.randn(M, H), torch.randn(E, H)
    block = _toy_router(K, torch.zeros(E, dtype=torch.float32), gate_w)
    block.routed_scaling_factor = 2.5
    weights, ids = MiMoV2SparseBlock._route(block, hidden)
    scores = torch.sigmoid(hidden.float() @ gate_w.float().T)
    renormed = scores.gather(-1, ids)
    renormed = renormed / (renormed.sum(-1, keepdim=True) + 1e-20)
    assert torch.allclose(weights, renormed * 2.5, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("top_k", "normalize", "expected_sum"),
    [(1, True, None), (8, True, 1.0), (3, False, None)],
)
def test_router_normalizes_only_when_enabled_and_top_k_exceeds_one(top_k, normalize, expected_sum):
    torch.manual_seed(23)
    hidden = torch.randn(2, 5)
    gate_w = torch.randn(8, 5)
    block = _toy_router(top_k, torch.zeros(8, dtype=torch.float32), gate_w)
    block.norm_topk_prob = normalize
    weights, ids = MiMoV2SparseBlock._route(block, hidden)
    scores = torch.sigmoid(hidden.float() @ gate_w.float().T).gather(-1, ids.long())
    if normalize and top_k > 1:
        scores = scores / (scores.sum(-1, keepdim=True) + 1e-20)
    assert torch.allclose(weights, scores, rtol=0, atol=1e-7)
    if expected_sum is not None:
        torch.testing.assert_close(weights.sum(-1), torch.full((2,), expected_sum), atol=1e-6, rtol=0)


# ---------------------------------------------------------------- dual rope (§3.0)


def test_dual_rope_tables_dispatch_per_group():
    cfg = parse_config(
        RawConfigShim(
            {
                "architectures": ["MiMoV2ForCausalLM"],
                "model_type": "mimo_v2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "swa_num_key_value_heads": 2,
                "head_dim": 192,
                "swa_head_dim": 192,
                "vocab_size": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 256,
                "hybrid_layer_pattern": [0, 1, 0],
                "moe_layer_freq": [0, 1, 1],
                "n_routed_experts": 8,
                "rope_theta": 10000000.0,
                "swa_rope_theta": 10000.0,
                "partial_rotary_factor": 0.334,
                "sliding_window": 128,
                "add_swa_attention_sink_bias": True,
                "add_full_attention_sink_bias": False,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
            }
        )
    )
    full = cfg.attention_group_for_layer(0)
    swa = cfg.attention_group_for_layer(1)
    assert isinstance(full, FullAttentionGroupConfig) and isinstance(swa, SWAAttentionGroupConfig)
    # the layer loop picks theta by layer type: 1e7 full, 1e4 swa (contract SECTION 3.0)
    assert full.rotary_config.base == 1e7
    assert swa.rotary_config.base == 1e4
    # int(192 * 0.334) == 64 on BOTH groups, tables indexed over head_dim 192
    assert full.rotary_config.rotary_dim == 64 == swa.rotary_config.rotary_dim
    assert full.rotary_config.head_dim == swa.rotary_config.head_dim == 192
    assert full.rotary_config is not swa.rotary_config
    assert cfg.rotary_config is full.rotary_config  # top default = full theta
    assert full.rotary_config.scaling is None and swa.rotary_config.scaling is None


def test_rope_sources_must_agree_when_both_present():
    base = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "head_dim": 192,
        "vocab_size": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 256,
        "sliding_window": 128,
        "hybrid_layer_pattern": [0, 1],
        "moe_layer_freq": [0, 1],
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "add_swa_attention_sink_bias": True,
        "add_full_attention_sink_bias": False,
        "moe_intermediate_size": 32,
    }
    conflict = dict(
        base,
        rope_theta=10000000.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    with pytest.raises(ValueError, match="rope config conflict"):
        parse_config(RawConfigShim(conflict))

    agree = dict(
        base,
        rope_theta=10000000.0,
        partial_rotary_factor=0.334,
        rope_parameters={"rope_theta": 10000000.0, "partial_rotary_factor": 0.334},
    )
    cfg = parse_config(RawConfigShim(agree))
    assert cfg.rotary_config.base == 1e7 and cfg.rotary_config.rotary_dim == 64


# ------------------------------------------------- resident experts (unquantized checkpoint)


_MOE_TOY = {
    "architectures": ["MiMoV2ForCausalLM"],
    "model_type": "mimo_v2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "swa_num_key_value_heads": 2,
    "head_dim": 192,
    "swa_head_dim": 192,
    "vocab_size": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 256,
    "hybrid_layer_pattern": [0, 1, 0],
    "moe_layer_freq": [0, 1, 1],
    "rope_theta": 10000000.0,
    "swa_rope_theta": 10000.0,
    "partial_rotary_factor": 0.334,
    "sliding_window": 128,
    "add_swa_attention_sink_bias": True,
    "add_full_attention_sink_bias": False,
    "num_experts_per_tok": 2,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
}


def test_resident_moe_layer_builds_the_unquantized_gate_up_down_tree():
    """expert_quant=none + fused strategy: the sparse block's experts are a resident MoELayer
    whose state-dict tree is EXACTLY experts.gate_up_proj [E, 2I, H] / experts.down_proj [E, H, I]
    in bf16 -- the keys the bf16 iter_weights path must fill."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import MoELayer, OffloadMoELayer
    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cfg = parse_config(RawConfigShim(_MOE_TOY))
    assert cfg.expert_quant == "none" and cfg.moe_strategy == "fused"

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        block = MiMoV2SparseBlock(cfg, layer_id=1, prefix="model.layers.1.mlp")

    experts = block.experts
    assert isinstance(experts, MoELayer) and not isinstance(experts, OffloadMoELayer)
    assert experts.prefix == "model.layers.1.mlp.experts"
    state = block.state_dict()  # prefix-relative: the model root namespaces these under model.layers.N.mlp.
    expected = {
        "experts.gate_up_proj": (4, 2 * 32, 64),
        "experts.down_proj": (4, 64, 32),
    }
    assert set(expected) <= set(state)
    for key, shape in expected.items():
        assert tuple(state[key].shape) == shape, key
        assert state[key].dtype is torch.bfloat16, key
