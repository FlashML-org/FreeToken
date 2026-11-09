"""Gemma-4 E2B / E4B: per-layer inputs, KV-shared layers and the double-wide MLP, against the shapes of the released checkpoints."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch

from freetoken.models.gemma4.config import parse_config

# The layer types of the releases: every (period)th layer is full attention.
E2B = dict(layers=35, hidden=1536, ffn=6144, shared=20, double_wide=True, kv_heads=1, period=5)
E4B = dict(layers=42, hidden=2560, ffn=10240, shared=18, double_wide=False, kv_heads=2, period=6)


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))


@pytest.fixture(autouse=True)
def _rope_tables():
    """Rope tables are cached per process and cannot live on the meta device these models build on."""
    from freetoken.layers import rotary

    device = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rotary.get_rope.cache_clear()
    rotary._cos_sin_table.cache_clear()
    yield
    rotary.set_rope_device(device)
    rotary.get_rope.cache_clear()
    rotary._cos_sin_table.cache_clear()


def _is_full(release: dict, layer: int) -> bool:
    return layer % release["period"] == release["period"] - 1


def _hf_config(release: dict, **overrides):
    n = release["layers"]
    per_layer = {
        "sliding_attention": SimpleNamespace(head_dim=256, num_key_value_heads=release["kv_heads"]),
        "full_attention": SimpleNamespace(head_dim=512, num_key_value_heads=release["kv_heads"]),
    }
    fields = dict(
        model_type="gemma4_text",
        hidden_size=release["hidden"],
        intermediate_size=release["ffn"],
        num_hidden_layers=n,
        num_attention_heads=8,
        is_heterogeneous=True,
        per_layer_config=per_layer,
        vocab_size=262144,
        rms_norm_eps=1e-6,
        max_position_embeddings=131072,
        sliding_window=512,
        layer_types=["full_attention" if _is_full(release, i) else "sliding_attention" for i in range(n)],
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"},
        },
        hidden_size_per_layer_input=256,
        vocab_size_per_layer_input=262144,
        num_kv_shared_layers=release["shared"],
        use_double_wide_mlp=release["double_wide"],
        tie_word_embeddings=True,
        final_logit_softcapping=30.0,
    )
    return SimpleNamespace(architectures=["Gemma4ForCausalLM"], **(fields | overrides))


def _meta_model(config):
    from freetoken.models.gemma4.model import Gemma4ForCausalLM
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return Gemma4ForCausalLM(config)


@pytest.mark.parametrize("fields", [
    {"hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "use_double_wide_mlp": False},  # 26B-A4B, 31B
    {"hidden_size_per_layer_input": None, "num_kv_shared_layers": None, "use_double_wide_mlp": None},
])
def test_other_releases_have_no_e_args(fields):
    config = parse_config(_hf_config(E4B, **fields))

    assert config.gemma4_e_args is None


def test_kv_source_is_the_last_unshared_layer_of_each_attention_type():
    """E2B shares from layer 15: full attention hands over from 14, sliding from 13."""
    config = parse_config(_hf_config(E2B))
    full, swa = config.attention_groups
    args = config.gemma4_e_args

    assert [i for i in full.layer_ids if args.is_kv_source(i, full.layer_ids)] == [14]
    assert [i for i in swa.layer_ids if args.is_kv_source(i, swa.layer_ids)] == [13]


def test_model_shapes_match_the_e2b_checkpoint():
    state = _meta_model(parse_config(_hf_config(E2B))).state_dict()

    assert state["model.embed_tokens_per_layer.weight"].shape == (262144, 35 * 256)
    assert state["model.per_layer_model_projection.weight"].shape == (35 * 256, 1536)
    assert state["model.layers.0.per_layer_input_gate.weight"].shape == (256, 1536)
    assert state["model.layers.0.per_layer_projection.weight"].shape == (1536, 256)
    assert state["model.layers.14.feed_forward.shared_mlp.gate_up_proj.weight"].shape == (2 * 6144, 1536)
    assert state["model.layers.15.feed_forward.shared_mlp.gate_up_proj.weight"].shape == (2 * 12288, 1536)
    assert state["model.layers.15.feed_forward.shared_mlp.down_proj.weight"].shape == (1536, 12288)


def test_shared_layers_hold_no_k_or_v_weights():
    state = _meta_model(parse_config(_hf_config(E2B))).state_dict()

    assert "model.layers.14.self_attn.qkv_proj.weight" in state
    assert "model.layers.14.self_attn.k_norm.weight" in state
    assert "model.layers.15.self_attn.qkv_proj.weight" not in state
    assert "model.layers.15.self_attn.k_norm.weight" not in state
    assert state["model.layers.15.self_attn.q_proj.weight"].shape == (8 * 256, 1536)


def test_other_releases_build_without_per_layer_weights():
    state = _meta_model(parse_config(_hf_config(E4B, hidden_size_per_layer_input=0, num_kv_shared_layers=0))).state_dict()

    assert not [k for k in state if "per_layer" in k]
    assert "model.layers.41.self_attn.qkv_proj.weight" in state


def test_loader_drops_shared_k_v_but_not_the_vision_towers(tmp_path, monkeypatch):
    """The checkpoint ships k, v and k_norm for shared layers too; the tower's layers reuse those names and indices."""
    from freetoken.models.gemma4 import weight

    release = dict(E2B, layers=3, shared=1, period=3)
    prefix = "model.language_model.layers"
    layers = {
        f"{prefix}.{i}.self_attn.{leaf}.weight": torch.zeros(2, 2)
        for i in range(3)
        for leaf in ("q_proj", "k_proj", "v_proj", "k_norm")
    }
    tensors = {
        **layers,
        "model.language_model.embed_tokens_per_layer.weight": torch.zeros(2, 2),
        "model.vision_tower.encoder.layers.2.self_attn.k_proj.linear.weight": torch.zeros(2, 2),
    }
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda _: _hf_config(release))

    names = dict(weight.iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True))

    assert "model.layers.2.self_attn.q_proj.weight" in names
    assert not [k for k in names if k.startswith("model.layers.2.self_attn.") and "q_proj" not in k]
    assert "model.layers.1.self_attn.qkv_proj.weight" in names
    assert "vision_tower.encoder.layers.2.self_attn.k_proj.weight" in names
    assert names["model.embed_tokens_per_layer.weight"].device.type == "cpu"


def test_embed_weights_host_moves_only_the_untied_table():
    from freetoken.engine.engine import _host_embedding_keys

    config = parse_config(_hf_config(E2B))
    model = _meta_model(config)

    assert _host_embedding_keys(model, SimpleNamespace(embed_weights="host", model_config=config)) == {
        "model.embed_tokens_per_layer.weight"
    }
    assert _host_embedding_keys(model, SimpleNamespace(embed_weights="gpu", model_config=config)) == frozenset()


def test_embed_weights_host_is_refused_when_the_only_table_is_tied():
    from freetoken.engine.engine import _host_embedding_keys

    config = parse_config(_hf_config(E4B, hidden_size_per_layer_input=0, num_kv_shared_layers=0))

    with pytest.raises(ValueError, match="tied to its LM head"):
        _host_embedding_keys(_meta_model(config), SimpleNamespace(embed_weights="host", model_config=config))


def test_a_float_table_is_read_without_a_copy():
    """A BF16 per-layer table is 5.6 GB: a copy before the loader's own would double the load's peak RAM."""
    from freetoken.models.gguf.tensors import fused, table_matrix

    data = np.arange(24, dtype=np.uint8)  # 3 rows of 4 BF16 values, as ggml dims [4, 3]
    info = SimpleNamespace(ggml_type=30, ne=(4, 3), data=lambda: data)

    table = table_matrix(info)
    assert table.shape == (3, 4) and table.dtype == torch.bfloat16
    assert np.shares_memory(table.view(torch.uint8).numpy(), data)
    ((name, weight),) = fused("t", [info], [table], 4)
    assert name == "t.weight" and weight is table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the rope kernels need CUDA")
def test_rotating_q_alone_matches_rotating_it_with_a_real_key():
    """A shared layer has no key to rotate, so it hands the kernel one head of zeros."""
    from freetoken.layers.rotary import get_rope

    with torch.device("cuda"):
        rope = get_rope(head_dim=256, rotary_dim=256, max_position=512, base=10000.0, rope_scaling=None)
    positions = torch.arange(16, device="cuda")
    q = torch.randn(16, 8 * 256, device="cuda", dtype=torch.bfloat16)
    with_key, alone = q.clone(), q.clone()

    rope.forward(positions, with_key, torch.randn(16, 256, device="cuda", dtype=torch.bfloat16))
    zeros = torch.zeros(16, 256, device="cuda", dtype=torch.bfloat16)
    rope.forward(positions, alone, zeros)

    assert torch.equal(with_key, alone)
    assert not zeros.any()


def _f32(ne, values=None):
    """A float GGUF tensor as the reader hands it out: ggml dims, raw bytes."""
    n = int(np.prod(ne))
    raw = (values if values is not None else torch.randn(n)).to(torch.float32).numpy().view(np.uint8)
    return SimpleNamespace(ggml_type=0, ne=tuple(ne), nbytes=raw.nbytes, data=lambda raw=raw: raw, name="", path="", root="")


def test_a_fused_qkv_file_gives_a_kv_shared_layer_its_q_rows(monkeypatch):
    """llama.cpp's --fuse-qkv also fuses a KV-shared layer, whose k and v the HF checkpoint still ships; q must be
    the leading rows of attn_qkv, as llama.cpp reads it, not the k or v rows after them."""
    from freetoken.models.gemma4 import gguf as gemma4_gguf
    import freetoken.utils

    release = dict(E2B, layers=3, shared=1, period=3, hidden=8, ffn=16, double_wide=False)
    config = parse_config(_hf_config(release, vocab_size=32, vocab_size_per_layer_input=32, hidden_size_per_layer_input=4))
    H, P, L, V = 8, 4, 3, 32
    index = {"token_embd.weight": _f32((H, V)), "output_norm.weight": _f32((H,)),
             "per_layer_token_embd.weight": _f32((L * P, V)), "per_layer_model_proj.weight": _f32((H, L * P)),
             "per_layer_proj_norm.weight": _f32((P,))}
    for i in range(L):
        group = config.attention_group_for_layer(i)
        q, kv = config.num_qo_heads * group.head_dim, group.num_kv_heads * group.head_dim
        for suffix in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_norm"):
            index[f"blk.{i}.{suffix}.weight"] = _f32((H,))
        for suffix in ("attn_q_norm", "attn_k_norm"):
            index[f"blk.{i}.{suffix}.weight"] = _f32((group.head_dim,))
        index[f"blk.{i}.layer_output_scale.weight"] = _f32((1,))
        index[f"blk.{i}.attn_qkv.weight"] = _f32((H, q + 2 * kv))
        index[f"blk.{i}.attn_output.weight"] = _f32((q, H))
        for part in ("gate", "up"):
            index[f"blk.{i}.ffn_{part}.weight"] = _f32((H, 16))
        index[f"blk.{i}.ffn_down.weight"] = _f32((16, H))
        index[f"blk.{i}.inp_gate.weight"] = _f32((H, P))
        index[f"blk.{i}.proj.weight"] = _f32((P, H))
    monkeypatch.setattr(gemma4_gguf, "gguf_tensor_index", lambda path: index)
    monkeypatch.setattr(gemma4_gguf, "parse_gguf_config", lambda shim: config)
    monkeypatch.setattr(gemma4_gguf, "log_mmproj", lambda *a: None)
    monkeypatch.setattr(freetoken.utils, "cached_load_hf_config", lambda path: None)

    weights = dict(gemma4_gguf.iter_gguf_weights("fused.gguf", torch.device("cpu"), include_moe_experts=False, include_non_moe=True))

    shared = config.attention_group_for_layer(2)
    q_rows = config.num_qo_heads * shared.head_dim
    fused_rows = gemma4_gguf.float_tensor(index["blk.2.attn_qkv.weight"])
    assert torch.equal(weights["model.layers.2.self_attn.q_proj.weight"], fused_rows[:q_rows].to(torch.bfloat16))
    assert weights["model.layers.0.self_attn.qkv_proj.weight"].shape[0] == index["blk.0.attn_qkv.weight"].ne[1]
    expected = _meta_model(config).state_dict()
    assert set(weights) == set(expected)
    assert [k for k in weights if tuple(weights[k].shape) != tuple(expected[k].shape)] == []
