from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch


def test_shard_tensor_splits_vocab_with_ceil_partition():
    from freetoken.models.loader import shard_tensor

    value = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)

    rank0 = shard_tensor(
        "model.embed_tokens.weight",
        value,
        rank=0,
        world_size=2,
        num_kv_heads=1,
    )
    rank1 = shard_tensor(
        "model.embed_tokens.weight",
        value,
        rank=1,
        world_size=2,
        num_kv_heads=1,
    )

    assert rank0.tolist() == value[:3].tolist()
    assert rank1.tolist() == value[3:5].tolist()


def test_iter_root_safetensor_files_from_index_keeps_only_root_index_shards(tmp_path):
    import json
    from freetoken.models.loader import iter_root_safetensor_files_from_index

    root_a = tmp_path / "model-00000-of-00002.safetensors"
    root_b = tmp_path / "model-00001-of-00002.safetensors"
    ignored = tmp_path / "consolidated.safetensors"
    metal = tmp_path / "metal" / "model.safetensors"
    original = tmp_path / "original" / "model.safetensors"
    for path in (root_a, root_b, ignored, metal, original):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": root_a.name,
                    "lm_head.weight": root_b.name,
                    "bad": "metal/model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    files = iter_root_safetensor_files_from_index(str(tmp_path))

    assert files == [str(root_a), str(root_b)]


def test_iter_root_safetensor_files_from_index_rejects_subdirectory_model_path(tmp_path):
    import pytest
    from freetoken.models.loader import iter_root_safetensor_files_from_index

    subdir = tmp_path / "original"
    subdir.mkdir()

    with pytest.raises(ValueError, match="root GPT-OSS"):
        iter_root_safetensor_files_from_index(str(subdir))


def test_iter_root_safetensor_files_from_index_requires_root_shards(tmp_path):
    import pytest
    from freetoken.models.loader import iter_root_safetensor_files_from_index

    with pytest.raises(ValueError, match="No root GPT-OSS safetensors shards"):
        iter_root_safetensor_files_from_index(str(tmp_path))


def test_shard_tensor_replicates_kv_heads_when_tp_exceeds_kv_heads():
    from freetoken.models.loader import shard_tensor

    value = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)

    rank0 = shard_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        value,
        rank=0,
        world_size=4,
        num_kv_heads=2,
    )
    rank1 = shard_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        value,
        rank=1,
        world_size=4,
        num_kv_heads=2,
    )
    rank2 = shard_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        value,
        rank=2,
        world_size=4,
        num_kv_heads=2,
    )
    rank3 = shard_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        value,
        rank=3,
        world_size=4,
        num_kv_heads=2,
    )

    assert rank0.tolist() == value[:1].tolist()
    assert rank1.tolist() == value[:1].tolist()
    assert rank2.tolist() == value[1:2].tolist()
    assert rank3.tolist() == value[1:2].tolist()


def test_merge_stream_buffers_qkv_until_all_slots_arrive():
    from freetoken.models.loader import MergeRule, iter_merged_tensors

    tensors = [
        ("model.layers.0.self_attn.q_proj.weight", torch.full((1, 2), 1.0)),
        ("model.layers.0.self_attn.v_proj.weight", torch.full((1, 2), 3.0)),
        ("model.layers.0.self_attn.k_proj.weight", torch.full((1, 2), 2.0)),
    ]
    rules = {
        ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
        ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
        ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    }

    merged = list(iter_merged_tensors(tensors, rules, model_name="test"))

    assert len(merged) == 1
    assert merged[0][0] == "model.layers.0.self_attn.qkv_proj.weight"
    assert merged[0][1].tolist() == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]


def test_iter_merged_tensors_reports_incomplete_merge_with_model_name():
    import pytest
    from freetoken.models.loader import MergeRule, iter_merged_tensors

    rules = {
        ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
        ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
        ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    }

    with pytest.raises(AssertionError, match="test.*Incomplete merge groups"):
        list(iter_merged_tensors([("x.q_proj.weight", torch.zeros(1, 1))], rules, model_name="test"))


def test_stack_expert_tensors_after_all_experts_arrive():
    from freetoken.models.loader import iter_stacked_experts

    expert_pattern = re.compile(
        r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$"
    )
    tensors = [
        ("model.layers.0.mlp.experts.1.gate_up_proj.weight", torch.full((1, 2), 11.0)),
        ("model.layers.0.mlp.experts.0.gate_up_proj.weight", torch.full((1, 2), 10.0)),
    ]

    packed = list(
        iter_stacked_experts(
            tensors,
            num_experts=2,
            model_name="qwen3_moe",
            expert_pattern=expert_pattern,
        )
    )

    assert len(packed) == 1
    assert packed[0][0] == "model.layers.0.mlp.experts.gate_up_proj"
    assert packed[0][1].shape == (2, 1, 2)
    assert packed[0][1][0].tolist() == [[10.0, 10.0]]
    assert packed[0][1][1].tolist() == [[11.0, 11.0]]


def test_stacked_expert_pieces_pair_each_layer_in_arrival_order():
    from freetoken.moe.expert_pieces import stacked_expert_pieces

    config = SimpleNamespace(num_layers=2, num_experts=2)
    tensors = [
        ("model.layers.1.mlp.experts.down_proj", torch.full((2, 4, 3), 11.0)),
        ("model.layers.0.mlp.experts.gate_up_proj", torch.full((2, 3, 4), 2.0)),
        ("model.layers.1.mlp.experts.gate_up_proj", torch.full((2, 3, 4), 3.0)),
        ("model.layers.0.mlp.experts.down_proj", torch.full((2, 4, 3), 10.0)),
    ]

    pieces = list(stacked_expert_pieces(tensors, config))

    assert [(layer, e0, e1) for layer, e0, e1, _ in pieces] == [(1, 0, 2), (0, 0, 2)]
    assert torch.equal(pieces[0][3]["gate_up"], torch.full((2, 3, 4), 3.0))
    assert torch.equal(pieces[0][3]["down"], torch.full((2, 4, 3), 11.0))
    assert torch.equal(pieces[1][3]["gate_up"], torch.full((2, 3, 4), 2.0))
    with pytest.raises(ValueError, match="Missing MoE expert source layers"):
        list(stacked_expert_pieces(tensors[:3], config))


@pytest.mark.parametrize("per_layer", [False, True], ids=["flat", "per-layer"])
@pytest.mark.parametrize("quant_format", ["q4_0", "unknown-format"])
def test_ftw_legacy_q4_0_banks_keep_native_bytes(tmp_path, monkeypatch, per_layer, quant_format):
    from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name, load_ftw_banks

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    writer = FTWWriter(str(tmp_path), shard_limit=8192)
    sources = {
        "gate_up": torch.arange(2 * 2 * 72, dtype=torch.int64).to(torch.uint8).reshape(2, 2, 72),
        "down": torch.arange(2 * 2 * 36, dtype=torch.uint8).reshape(2, 2, 36),
    }
    for name, value in sources.items():
        if per_layer:
            for layer, rows in enumerate(value):
                writer.add_tensor(layer_bank_entry_name(name, layer), rows, kind="experts_bank")
        else:
            writer.add_tensor(name, value.flatten(0, 1), kind="experts_bank")
    writer.finalize({"quant_format": quant_format, "expert_bank_num_layers": 2})

    if quant_format != "q4_0":
        with pytest.raises(KeyError, match="unknown-format"):
            load_ftw_banks(str(tmp_path), num_layers=2, layer_residency=["pageable"] * 2)
        return
    banks = load_ftw_banks(str(tmp_path), num_layers=2, layer_residency=["pageable"] * 2)
    assert banks.quant_format == "q4_0"
    assert banks.kind is None and banks.kernel is None
    assert banks.layer_residency == ["pageable"] * 2
    assert set(banks.sources) == set(sources)
    for name, value in sources.items():
        assert len(banks.sources[name]) == 2
        for layer, rows in enumerate(banks.sources[name]):
            assert rows.dtype is torch.uint8
            torch.testing.assert_close(rows, value[layer], rtol=0, atol=0)


@pytest.mark.parametrize("include_vision", [False, True])
def test_ftw_text_only_filters_native_and_qwen_vision_weights(tmp_path, include_vision):
    from freetoken.checkpoint.ftw import FTWWriter
    from freetoken.models.weight import load_weight

    vision_keys = ("vision.blocks.0.norm1.weight", "aligner.mlp.0.weight",
                   "image_start", "image_end", "image_newline", "visual.blocks.0.norm1.weight")
    text_keys = ("embed.weight", "layers.0.attn.wq_a.weight", "head.weight")
    weights = {name: torch.full((2, 3), float(i), dtype=torch.bfloat16)
               for i, name in enumerate((*vision_keys, *text_keys))}
    writer = FTWWriter(str(tmp_path), shard_limit=8192)
    for name, value in weights.items():
        writer.add_tensor(name, value)
    writer.finalize({})

    actual = dict(load_weight(str(tmp_path), torch.device("cpu"), include_vision=include_vision))
    assert set(actual) == set(weights if include_vision else text_keys)
    for name, value in actual.items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("with_input_scale", [False, True])
def test_ftw_activation_scale_follows_the_current_model_scheme(tmp_path, with_input_scale):
    from freetoken.checkpoint.ftw import FTWWriter
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict
    from freetoken.models.weight import load_weight

    layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    layer.register_buffer("weight_scale", torch.empty((), dtype=torch.float32))
    if with_input_scale:
        layer.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
    model = torch.nn.ModuleDict({"linear": layer})
    weights = {"linear.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
               "linear.weight_scale": torch.tensor(0.25), "linear.input_scale": torch.tensor(0.5)}
    writer = FTWWriter(str(tmp_path), shard_limit=8192)
    for name, value in weights.items():
        writer.add_tensor(name, value)
    writer.finalize({})

    loaded = _materialize_loaded_weight_state_dict(
        model.state_dict(), load_weight(str(tmp_path), torch.device("cpu")), device=torch.device("cpu"),
    )
    model.load_state_dict(loaded, strict=True)
    assert set(loaded) == set(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
