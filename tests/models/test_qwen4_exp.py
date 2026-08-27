from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import (
    _PLELayer,
    _preload_ple_enabled,
    _tokens_for_ngram_forward,
    build_ngram_ids,
)
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
    assert config.supports_cuda_graph
    assert config.is_linear_layer(0)
    assert not config.is_linear_layer(3)


def test_qwen4_config_accepts_transformers_sparse_attention_alias():
    hf_config = _config()
    hf_config.text_config.layer_types[-1] = "qwen_sparse_attention"
    config = parse_config(hf_config)
    assert not config.is_linear_layer(3)


def test_qwen4_config_accepts_modelopt_nvfp4_experts():
    hf_config = _config()
    hf_config.quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
    }

    config = parse_config(hf_config)

    assert config.expert_quant == "nvfp4"
    assert config.weight_block_size is None
    assert config.attn_quant == "none"
    assert config.dense_quant == "none"


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


def test_ngram_history_includes_inflight_overlap_token():
    req = SimpleNamespace(input_ids=torch.tensor([4, 5, 6]), device_len=4)

    tokens = _tokens_for_ngram_forward(req, torch.tensor([7], device="cpu"))

    assert tokens.tolist() == [4, 5, 6, 7]
    assert req.input_ids.tolist() == [4, 5, 6]


def test_ngram_history_does_not_duplicate_drained_token():
    req = SimpleNamespace(input_ids=torch.tensor([4, 5, 6, 7]), device_len=4)

    tokens = _tokens_for_ngram_forward(req, torch.tensor([7], device="cpu"))

    assert tokens.tolist() == [4, 5, 6, 7]


def test_ngram_history_can_select_only_required_suffix():
    req = SimpleNamespace(input_ids=torch.tensor([4, 5, 6]), device_len=5)

    tokens = _tokens_for_ngram_forward(
        req,
        torch.tensor([7, 8], device="cpu"),
        start=2,
    )

    assert tokens.tolist() == [6, 7, 8]
    assert _tokens_for_ngram_forward(req, torch.tensor([7, 8]), start=4).tolist() == [8]


def test_incremental_ngram_hash_matches_full_history():
    tokens = torch.tensor([10, 99, 4, 5, 6, 7])
    kwargs = {
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "eos_token_id": 99,
        "multipliers": torch.tensor([3, 5, 7]),
        "vocab_sizes": torch.tensor([101, 103]),
        "offsets": torch.tensor([0, 101]),
    }

    full = build_ngram_ids(tokens, **kwargs)
    history_start = 3
    incremental = build_ngram_ids(tokens[history_start:], **kwargs)

    assert torch.equal(incremental[2], full[5])


def test_ple_graph_convolution_matches_eager(monkeypatch):
    def make_layer():
        layer = object.__new__(_PLELayer)
        layer.hidden_size = 2
        layer.hc_count = 2
        layer.state_len = 1
        layer.dilation = 1
        layer.conv1d = SimpleNamespace(weight=torch.randn(4, 1, 2))
        layer._conv_state_pool = None
        return layer

    req = SimpleNamespace(table_idx=1, extend_len=1, cached_len=0)
    linear_pool = SimpleNamespace(conv_states=torch.empty(1, 3, 1, 1))
    hidden = torch.randn(1, 4)

    eager = make_layer()
    eager_batch = SimpleNamespace(
        is_decode=True,
        reqs=[req],
        padded_reqs=[req],
        cuda_graph_capture=False,
        linear_table_idx=torch.tensor([1], dtype=torch.int32),
    )
    context = SimpleNamespace(batch=eager_batch, linear_state_pool=linear_pool)
    monkeypatch.setattr("freetoken.models.qwen4_exp.model.get_global_ctx", lambda: context)
    eager_output = eager._short_conv(hidden)

    graph = make_layer()
    graph.conv1d.weight.copy_(eager.conv1d.weight)
    context.batch = SimpleNamespace(
        is_decode=True,
        reqs=[req],
        padded_reqs=[req],
        cuda_graph_capture=True,
        linear_table_idx=torch.tensor([1], dtype=torch.int32),
    )
    graph_output = graph._short_conv(hidden)

    torch.testing.assert_close(graph_output, eager_output)
    torch.testing.assert_close(graph._conv_state_pool, eager._conv_state_pool)


def test_qwen4_ple_preload_is_opt_in(monkeypatch):
    monkeypatch.delenv("FREETOKEN_QWEN4_PLE_PRELOAD", raising=False)
    assert not _preload_ple_enabled()

    monkeypatch.setenv("FREETOKEN_QWEN4_PLE_PRELOAD", "true")
    assert _preload_ple_enabled()
