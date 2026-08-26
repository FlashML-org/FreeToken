from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import build_ngram_ids
from freetoken.models.qwen4_exp.weight import _rename, _try_fuse
from freetoken.models.register import get_model_spec


def _config():
    text = SimpleNamespace(
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        head_dim=256,
        rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 10_000_000},
        indexer_budget=2048,
        max_position_embeddings=262_144,
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        eos_token_id=248044,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        split_ngram_parts=128,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        hidden_act="silu",
        num_hidden_layers=4,
        num_attention_heads=24,
        hidden_size=2560,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        norm_topk_prob=None,
        tie_word_embeddings=False,
    )
    return SimpleNamespace(
        text_config=text,
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        image_token_id=248056,
    )


def test_qwen4_config_uses_exact_qsa_prefix():
    config = parse_config(_config())
    assert config.rotary_config.max_position == 2048
    assert config.expert_quant == "fp8_block"
    assert config.attn_quant == "none"
    assert config.qwen4_args.ple_layer_ids == (1,)
    assert config.qwen4_args.output_gate_type == "sigmoid"
    assert config.requires_naive_cache
    assert not config.supports_cuda_graph
    assert config.is_linear_layer(0)
    assert not config.is_linear_layer(3)


def test_qwen4_config_accepts_transformers_sparse_attention_alias():
    hf_config = _config()
    hf_config.text_config.layer_types[-1] = "qwen_sparse_attention"
    config = parse_config(hf_config)
    assert not config.is_linear_layer(3)


def test_qwen4_registry_entry():
    spec = get_model_spec("Qwen4ExpForConditionalGeneration")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"


def test_qwen4_weight_names():
    assert _rename("model.language_model.layers.1.ple.key_proj.weight") == (
        "model.layers.1.ple.key_proj.weight"
    )
    assert _rename("model.visual.blocks.0.attn.qkv.weight") is None
    assert _rename("model.language_model.layers.3.self_attn.indexer.q_layernorm.weight") is None


def test_qwen4_projection_fusion_order():
    buffers = {}
    base = "model.layers.3.self_attn."
    parts = [
        ("q_proj.weight", torch.full((2, 3), 1.0)),
        ("k_proj.weight", torch.full((1, 3), 2.0)),
        ("v_proj.weight", torch.full((1, 3), 3.0)),
    ]
    assert _try_fuse(base + parts[0][0], parts[0][1], buffers) == ()
    assert _try_fuse(base + parts[1][0], parts[1][1], buffers) == ()
    name, fused = _try_fuse(base + parts[2][0], parts[2][1], buffers)
    assert name == base + "qkv_proj.weight"
    assert fused[:, 0].tolist() == [1.0, 1.0, 2.0, 3.0]


def test_ngram_hash_resets_at_eos():
    tokens = torch.tensor([4, 5, 99, 6, 7])
    multipliers = torch.tensor([3, 5, 7])
    sizes = torch.tensor([101, 103])
    offsets = torch.tensor([0, 101])
    ids = build_ngram_ids(
        tokens,
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
        multipliers=multipliers,
        vocab_sizes=sizes,
        offsets=offsets,
    )
    assert ids.shape == (5, 2)
    expected_bigram_after_eos = (6 * 3) ^ (99 * 5)
    assert ids[3, 0].item() == expected_bigram_after_eos % 101
