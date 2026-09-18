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
