from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "laguna-s-2.1-metadata.gguf"


def _safetensors_config():
    layers = 48
    return SimpleNamespace(
        architectures=["LagunaForCausalLM"],
        num_hidden_layers=layers,
        head_dim=128,
        max_position_embeddings=1_048_576,
        num_attention_heads_per_layer=[48 if i % 4 == 0 else 72 for i in range(layers)],
        layer_types=[
            "full_attention" if i % 4 == 0 else "sliding_attention"
            for i in range(layers)
        ],
        rope_parameters={
            "full_attention": {
                "rope_theta": 500_000.0,
                "rope_type": "yarn",
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.4852030263919618,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        mlp_only_layers=[0],
        num_key_value_heads=8,
        hidden_size=3072,
        vocab_size=100352,
        intermediate_size=12288,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        num_experts=256,
        num_experts_per_tok=10,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
        norm_topk_prob=True,
        moe_routed_scaling_factor=2.5,
        sliding_window=512,
        quantization_config={
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "targets": [
                        r"re:.*layers\.\d+\..*(gate_proj|up_proj|down_proj)$"
                    ],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "group_size": 32,
                        "strategy": "group",
                        "symmetric": True,
                    },
                }
            },
            "ignore": [
                r"re:^model\.layers\.(?:40|41|42|43|44|45|46|47)\.mlp\.experts\.[0-9]+\.(?:gate_proj|up_proj|down_proj)$"
            ],
        },
    )


def test_laguna_config_parse_and_attention_groups():
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.laguna.gguf import parse_gguf_config

    cfg = parse_gguf_config(build_gguf_shim(str(FIXTURE)))

    assert cfg.num_layers == 48
    assert cfg.num_qo_heads == 72
    assert all(v == 48 for v in cfg.num_qo_heads_per_layer[0::4])
    assert all(v == 72 for i, v in enumerate(cfg.num_qo_heads_per_layer) if i % 4 != 0)
    assert cfg.qo_heads(0) == 48
    assert cfg.qo_heads(1) == 72
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.hidden_size == 3072
    assert cfg.vocab_size == 100352
    assert cfg.first_k_dense_replace == 1
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 10
    assert cfg.moe_intermediate_size == 1024
    assert cfg.shared_expert_intermediate_size == 1024
    assert cfg.n_shared_experts == 1
    assert cfg.routed_scaling_factor == 2.5
    assert cfg.norm_topk_prob is True
    assert cfg.tie_word_embeddings is False
    assert cfg.model_type == "laguna"
    assert cfg.moe_enabled is True
    assert cfg.use_qk_norm is True
    assert cfg.rms_norm_eps == pytest.approx(1e-6)
    assert cfg.rotary_config.max_position == 262144

    assert len(cfg.attention_groups) == 2
    full = cfg.attention_groups[0]
    swa = cfg.attention_groups[1]

    assert tuple(full.layer_ids) == tuple(range(0, 48, 4))
    assert tuple(swa.layer_ids) == tuple(i for i in range(0, 48) if i not in set(full.layer_ids))
    assert swa.sliding_window == 512

    assert full.rotary_config.rotary_dim == 64
    assert full.rotary_config.base == 500000.0
    assert full.rotary_config.scaling is not None
    assert full.rotary_config.scaling["rope_type"] == "yarn"
    assert full.rotary_config.scaling["factor"] == 32.0
    assert full.rotary_config.scaling["attention_factor"] == 1.0
    assert full.rotary_config.scaling["beta_fast"] == 32.0
    assert full.rotary_config.scaling["beta_slow"] == 1.0
    assert full.rotary_config.scaling["original_max_position_embeddings"] == 8192

    assert swa.rotary_config.rotary_dim == 128
    assert swa.rotary_config.base == 10000.0
    assert swa.rotary_config.scaling is None

    assert cfg.gguf_embed_quant is None


def test_laguna_gguf_arch_registry_map():
    from freetoken.models import register
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY

    assert GGUF_ARCH_TO_REGISTRY["laguna"] == "LagunaGGUFForCausalLM"
    assert "LagunaGGUFForCausalLM" in register._MODEL_REGISTRY


def test_laguna_safetensors_config_and_mixed_expert_types():
    from freetoken.models import register
    from freetoken.models.gguf.dequant import GGML_BF16, GGML_Q4_0
    from freetoken.models.laguna.config import parse_config

    cfg = parse_config(_safetensors_config())
    assert cfg.architectures == ["LagunaForCausalLM"]
    assert cfg.num_layers == 48 and cfg.num_moe_layers == 47
    assert cfg.rotary_config.max_position == 1_048_576
    assert cfg.rotary_config.rotary_dim == 64
    assert cfg.attention_groups[1].sliding_window == 512
    assert cfg.expert_quant == cfg.moe_weight_format == "laguna_int4"
    assert cfg.gguf_expert_types == (
        ((GGML_Q4_0, GGML_Q4_0),) * 39
        + ((GGML_BF16, GGML_BF16),) * 8
    )
    assert "LagunaForCausalLM" in register._MODEL_REGISTRY


def test_laguna_tokenizer_and_eos_tokens():
    from freetoken.models.gguf.tokenizer import gguf_eos_token_ids, load_gguf_tokenizer

    tok = load_gguf_tokenizer(str(FIXTURE))
    assert tok.eos_token_id == 2
    assert gguf_eos_token_ids(str(FIXTURE), tok) == {2, 24}
    assert tok.chat_template

    ids = tok("def foo(): return 1").input_ids
    assert tok.decode(ids).endswith("def foo(): return 1")
