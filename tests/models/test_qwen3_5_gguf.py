from __future__ import annotations

from types import SimpleNamespace

import torch


def _shim():
    metadata = {
        "qwen35moe.block_count": 41,
        "qwen35moe.nextn_predict_layers": 1,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.context_length": 262144,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.rope.dimension_count": 64,
        "qwen35moe.rope.freq_base": 10_000_000.0,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.inner_size": 4096,
        "qwen35moe.ssm.group_count": 16,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.expert_feed_forward_length": 512,
        "qwen35moe.expert_shared_feed_forward_length": 512,
    }
    return SimpleNamespace(
        metadata=metadata,
        model_path="ornith.gguf",
        vocab_size=248320,
        tie_word_embeddings=False,
        architectures=["Qwen3_5MoeGGUFForConditionalGeneration"],
    )


def test_qwen35moe_gguf_config_excludes_mtp_and_builds_hybrid_groups(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import gguf

    monkeypatch.setattr(reader, "gguf_tensor_type", lambda path, name: 12)
    monkeypatch.setattr(gguf, "_expert_types", lambda shim: ((12, 14),) * 40)
    config = gguf.parse_gguf_config(_shim())

    assert config.num_layers == 40
    assert config.rotary_config.max_position == 262144
    assert config.rotary_config.rotary_dim == 64
    assert config.num_experts == 256
    assert config.num_experts_per_tok == 8
    assert config.moe_intermediate_size == 512
    assert config.shared_expert_intermediate_size == 512
    assert config.expert_quant == "gguf"
    assert config.gguf_embed_quant == 12
    assert len(config.gguf_expert_types) == 40

    linear = config.linear_attention_group()
    assert linear is not None
    assert len(linear.layer_ids) == 30
    assert linear.num_key_heads == 16
    assert linear.num_value_heads == 32
    assert linear.key_head_dim == linear.value_head_dim == 128
    full = next(group for group in config.attention_groups if group.name == "full")
    assert tuple(full.layer_ids) == tuple(range(3, 40, 4))


def test_qwen35moe_gguf_registry_mapping():
    from freetoken.models import register
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY

    key = "Qwen3_5MoeGGUFForConditionalGeneration"
    assert GGUF_ARCH_TO_REGISTRY["qwen35moe"] == key
    assert key in register._MODEL_REGISTRY

    from freetoken.models.gguf.tokenizer import _TOKENIZER_ARCH

    assert _TOKENIZER_ARCH["qwen35moe"] == "qwen3_moe"


def test_inverse_v_permutation_restores_grouped_head_order(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import gguf

    monkeypatch.setattr(reader, "gguf_tensor_type", lambda path, name: 12)
    monkeypatch.setattr(gguf, "_expert_types", lambda shim: ((12, 14),) * 40)
    config = gguf.parse_gguf_config(_shim())
    grouped = torch.arange(32 * 3).reshape(32 * 3, 1)
    # llama.cpp stores [G0v0, G1v0, ..., G0v1, G1v1, ...].
    tiled = grouped.reshape(16, 2, 3).permute(1, 0, 2).reshape(32 * 3, 1)
    assert torch.equal(gguf._undo_v_rows(tiled, config, 3), grouped)
