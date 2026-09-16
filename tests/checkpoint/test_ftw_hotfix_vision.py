"""The hotfix tool's vision repair: picking the encoder's checkpoint tensors and letting the family reader name them."""

import importlib.util
import json
import os
import pathlib
import shutil

import pytest

from freetoken.models.config import VISION_KEY_PREFIXES

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ftw_hotfix.py"

# env var -> local checkpoint, as tests/README.md lists them
_CHECKPOINTS = ("FREETOKEN_QWEN3VL_MODEL", "FREETOKEN_GEMMA4_MODEL", "FREETOKEN_GEMMA4_UNIFIED_MODEL",
                "FREETOKEN_GLM53_MODEL", "FREETOKEN_MUSE_MODEL", "FREETOKEN_MINIMAX_M3_MODEL")


@pytest.fixture(scope="module")
def hotfix():
    spec = importlib.util.spec_from_file_location("ftw_hotfix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name, tower", [
    ("model.visual.blocks.0.attn.qkv.weight", True),
    ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", True),
    ("model.embed_vision.embedding_projection.weight", True),
    ("model.vision_embedder.patch_embedding.weight", True),
    ("model.vision_adapter.fc1.weight", True),
    ("vision_tower.vision_model.embeddings.patch_embedding.weight", True),
    ("multi_modal_projector.linear_1.weight", True),
    ("patch_merge_mlp.linear_2.bias", True),
    ("vision.patch_embed.proj.weight", True),
    ("aligner.w1.weight", True),
    ("image_start", True),
    ("image_end", True),
    ("image_newline", True),
    ("model.embed_audio.embedding_projection.weight", False),
    ("model.language_model.layers.0.self_attn.q_proj.weight", False),
    ("model.layers.3.mlp.experts.0.gate_proj.weight", False),
    ("lm_head.weight", False),
])
def test_checkpoint_tower_names(hotfix, name, tower):
    assert hotfix.is_checkpoint_tower_name(name) is tower


def _config_dir(hotfix, checkpoint, tmp_path):
    """An FTW-shaped dir holding only the checkpoint's config files and an empty index, enough to build the model on the meta device."""
    for f in os.listdir(checkpoint):
        if f.endswith((".json", ".py", ".jinja")) and not f.startswith("model"):
            shutil.copy(os.path.join(checkpoint, f), tmp_path)
    index = {"format": hotfix.FORMAT_TAG, "version": 1, "align": hotfix.ALIGN, "shard_limit": 8 << 30, "total_bytes": 0, "tensors": [], "shards": []}
    (tmp_path / hotfix.INDEX_NAME).write_text(json.dumps(index))
    return str(tmp_path)


@pytest.mark.needs_weights
@pytest.mark.parametrize("env", _CHECKPOINTS)
def test_reader_emits_every_declared_encoder_tensor(hotfix, env, tmp_path):
    checkpoint = os.environ.get(env)
    if not checkpoint:
        pytest.skip(f"{env} not set")
    ftw_like = _config_dir(hotfix, checkpoint, tmp_path)
    _, expected, _, _ = hotfix.expected_tensors(ftw_like, resident_experts=False)
    declared = {n: shape for n, (shape, _) in expected.items() if n.startswith(VISION_KEY_PREFIXES)}
    assert declared, "the family declares no encoder tensors"
    source = hotfix.TensorSource(None, checkpoint)
    tower_names = [n for n in source.weight_map if hotfix.is_checkpoint_tower_name(n)]
    got = hotfix.read_tower(source, ftw_like, tower_names)
    assert sorted(set(declared) - set(got)) == []
    assert [n for n in declared if tuple(got[n].shape) != declared[n]] == []


def test_every_family_with_an_encoder_has_an_encoder_only_reader():
    from freetoken.models.register import _MODEL_REGISTRY, _load_attr

    for spec in _MODEL_REGISTRY.values():
        if spec.encoders:
            assert callable(_load_attr(spec.module, "iter_vision_weights")), spec.module


def test_native_v41_hotfix_reads_only_vision_tensors(hotfix, tmp_path, monkeypatch):
    import safetensors.torch
    import torch

    from freetoken.models.deepseek_v41 import weight
    from freetoken.models.weight import load_vision_weight

    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    config = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "n_layers": 1,
              "compress_ratios": [0], "kv_source_layers": [], "index_source_layers": [],
              "candidate_source_layer": -1, "engram_layer_ids": [], "engram_num_embeddings": [],
              "vision_n_layers": 1, "quantization_config": {"expert_dtype": "nvfp4"}}
    (checkpoint / "config.json").write_text(json.dumps(config))
    tensors = {
        "vision.patch_embed.proj.weight": torch.arange(6, dtype=torch.bfloat16).view(2, 3),
        "vision.norm.weight": torch.arange(2, dtype=torch.float32),
        "aligner.w1.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        "image_start": torch.full((4,), 1.0, dtype=torch.bfloat16),
        "image_end": torch.full((4,), 2.0, dtype=torch.bfloat16),
        "image_newline": torch.full((4,), 3.0, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(tensors, checkpoint / "vision.safetensors")
    index = {name: "vision.safetensors" for name in tensors}
    index["head.weight"] = "unavailable-text.safetensors"
    index["layers.0.ffn.experts.0.w1.weight"] = "unavailable-experts.safetensors"
    index["layers.0.engram.embed.weight"] = "unavailable-engram.safetensors"
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    reads = []
    original_get = weight._ShardReader.get

    def get(reader, name):
        reads.append(name)
        return original_get(reader, name)

    monkeypatch.setattr(weight._ShardReader, "get", get)
    direct = dict(load_vision_weight(str(checkpoint), torch.device("cpu")))
    assert set(direct) == set(reads) == set(tensors)
    reads.clear()
    ftw_like = tmp_path / "ftw"
    ftw_like.mkdir()
    ftw_dir = _config_dir(hotfix, checkpoint, ftw_like)
    source = hotfix.TensorSource(None, str(checkpoint))
    selected = [name for name in source.weight_map if hotfix.is_checkpoint_tower_name(name)]
    got = hotfix.read_tower(source, ftw_dir, selected)
    assert set(got) == set(tensors)
    assert set(reads) == set(tensors)
    for name, expected in tensors.items():
        assert direct[name].dtype == expected.dtype
        assert torch.equal(direct[name], expected)
        assert got[name].dtype == expected.dtype
        assert torch.equal(got[name], expected)
