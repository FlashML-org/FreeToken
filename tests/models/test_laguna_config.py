from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "laguna-s-2.1-metadata.gguf"


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
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
    from freetoken.models import register

    assert GGUF_ARCH_TO_REGISTRY["laguna"] == "LagunaGGUFForCausalLM"
    assert "LagunaGGUFForCausalLM" in getattr(register, "_MODEL_REGISTRY")


def test_laguna_tokenizer_and_eos_tokens():
    from freetoken.models.gguf.tokenizer import gguf_eos_token_ids, load_gguf_tokenizer

    tok = load_gguf_tokenizer(str(FIXTURE))
    assert tok.eos_token_id == 2
    assert gguf_eos_token_ids(str(FIXTURE), tok) == {2, 24}
    assert tok.chat_template

    ids = tok("def foo(): return 1").input_ids
    assert tok.decode(ids).endswith("def foo(): return 1")
