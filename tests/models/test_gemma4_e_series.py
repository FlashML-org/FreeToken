"""Gemma-4 E2B / E4B: per-layer inputs, KV-shared layers and the double-wide MLP, against the shapes of the released checkpoints."""
from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch

from freetoken.models.gemma4.config import parse_config
from freetoken.models.gemma4.gguf import parse_gguf_config, to_gguf

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


def _gguf_shim(tmp_path, release: dict, *, ffn):
    from freetoken.models.gguf.config import GgufConfigShim

    n = release["layers"]
    # no tensors and no KV of its own: the parser reads the shim's metadata
    path = tmp_path / "metadata_only.gguf"
    path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))
    metadata = {
        "gemma4.block_count": n,
        "gemma4.embedding_length": release["hidden"],
        "gemma4.embedding_length_per_layer_input": 256,
        "gemma4.feed_forward_length": ffn,
        "gemma4.context_length": 131072,
        "gemma4.final_logit_softcapping": 30.0,
        "gemma4.attention.head_count": 8,
        "gemma4.attention.head_count_kv": release["kv_heads"],
        "gemma4.attention.key_length": 512,
        "gemma4.attention.key_length_swa": 256,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
        "gemma4.attention.sliding_window": 512,
        "gemma4.attention.shared_kv_layers": release["shared"],
        "gemma4.attention.sliding_window_pattern": [not _is_full(release, i) for i in range(n)],
        "gemma4.rope.freq_base": 1000000.0,
        "gemma4.rope.freq_base_swa": 10000.0,
        "gemma4.rope.dimension_count": 512,
        "gemma4.rope.dimension_count_swa": 256,
    }
    return GgufConfigShim(
        architectures=["Gemma4GGUFForCausalLM"],
        model_path=str(path),
        model_type="gemma4",
        metadata=metadata,
        vocab_size=262144,
        tie_word_embeddings=True,
    )


def _meta_model(config):
    from freetoken.models.gemma4.model import Gemma4ForCausalLM
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return Gemma4ForCausalLM(config)


def test_e2b_config_doubles_the_shared_tail_only():
    config = parse_config(_hf_config(E2B))
    args = config.gemma4_e_args

    assert args.per_layer_hidden_size == 256 and args.first_kv_shared_layer == 15
    assert config.intermediate_size == 6144
    assert config.layer_intermediate_sizes == (6144,) * 15 + (12288,) * 20
    assert not args.is_kv_shared(14) and args.is_kv_shared(15)


def test_e4b_config_keeps_one_mlp_width():
    config = parse_config(_hf_config(E4B))

    assert config.gemma4_e_args.first_kv_shared_layer == 24
    assert config.layer_intermediate_sizes == ()
    assert config.intermediate_size_for_layer(41) == 10240


@pytest.mark.parametrize("fields", [
    {"hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "use_double_wide_mlp": False},  # 26B-A4B, 31B
    {"hidden_size_per_layer_input": None, "num_kv_shared_layers": None, "use_double_wide_mlp": None},
])
def test_other_releases_have_no_e_args(fields):
    config = parse_config(_hf_config(E4B, **fields))

    assert config.gemma4_e_args is None
    assert config.layer_intermediate_sizes == ()


def test_shared_layers_without_per_layer_inputs_are_refused():
    with pytest.raises(NotImplementedError, match="KV-shared"):
        parse_config(_hf_config(E4B, hidden_size_per_layer_input=0))


def test_kv_source_is_the_last_unshared_layer_of_each_attention_type():
    """E2B shares from layer 15: full attention hands over from 14, sliding from 13."""
    config = parse_config(_hf_config(E2B))
    full, swa = config.attention_groups
    args = config.gemma4_e_args

    assert [i for i in full.layer_ids if args.is_kv_source(i, full.layer_ids)] == [14]
    assert [i for i in swa.layer_ids if args.is_kv_source(i, swa.layer_ids)] == [13]


def test_gguf_reads_a_per_layer_mlp_width_list(tmp_path):
    widths = [6144] * 15 + [12288] * 20
    config = parse_gguf_config(_gguf_shim(tmp_path, E2B, ffn=widths))

    assert config.layer_intermediate_sizes == tuple(widths)
    assert config.intermediate_size == 6144
    assert config.num_kv_heads == 1
    assert config.gemma4_e_args.first_kv_shared_layer == 15


def test_gguf_reads_one_mlp_width_and_one_kv_head_count(tmp_path):
    config = parse_gguf_config(_gguf_shim(tmp_path, E4B, ffn=10240))

    assert config.layer_intermediate_sizes == ()
    assert config.intermediate_size == 10240
    assert config.num_kv_heads == 2
    assert config.gemma4_e_args.first_kv_shared_layer == 24


def test_full_layers_store_v_on_e_series_gguf(tmp_path):
    config = parse_gguf_config(_gguf_shim(tmp_path, E4B, ffn=10240))

    assert config.attention_groups[0].k_eq_v is False


def test_to_gguf_names_the_per_layer_tensors():
    assert to_gguf("model.embed_tokens_per_layer", {}) == ("per_layer_token_embd",)
    assert to_gguf("model.per_layer_model_projection", {}) == ("per_layer_model_proj",)
    assert to_gguf("model.layers.3.per_layer_input_gate", {}) == ("blk.3.inp_gate",)
    assert to_gguf("model.layers.3.per_layer_projection", {}) == ("blk.3.proj",)


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


def test_clipped_vision_towers_serve_text_only():
    """E2B / E4B's tower clips its linears, which the tower does not apply: text must still serve."""
    from freetoken.models.register import get_model_spec

    (encoder,) = get_model_spec("Gemma4ForConditionalGeneration").encoders

    assert "clipped linears" in encoder.unserved(SimpleNamespace(use_clipped_linears=True))
    assert encoder.unserved(SimpleNamespace(use_clipped_linears=False)) is None


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
