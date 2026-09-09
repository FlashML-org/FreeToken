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
