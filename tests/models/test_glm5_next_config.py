from __future__ import annotations

from types import SimpleNamespace


def _release_config():
    """Small, self-contained projection of zai-org/GLM-5.3-Flash config.json."""
    num_layers = 45
    layer_types = [
        "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
        for i in range(num_layers)
    ]
    return SimpleNamespace(
        model_type="glm5_next",
        architectures=["Glm5NextForConditionalGeneration"],
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
        text_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            layer_types=layer_types,
            mlp_layer_types=["dense"] * 3 + ["sparse"] * 42,
            indexer_types=["full"] * num_layers,
            hidden_size=4096,
            vocab_size=154880,
            intermediate_size=12288,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            max_position_embeddings=1048576,
            rope_theta=None,
            num_attention_heads=64,
            linear_attn_config={
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
            qk_head_dim=256,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            v_head_dim=256,
            kv_lora_rank=512,
            q_lora_rank=1536,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
            index_kpool=4,
            hc_mult=4,
            hc_eps=1e-6,
            hc_sinkhorn_iters=20,
            n_routed_experts=288,
            num_experts_per_tok=8,
            moe_intermediate_size=2048,
            norm_topk_prob=True,
            first_k_dense_replace=3,
            n_shared_experts=1,
            routed_scaling_factor=2.5,
            n_group=1,
            topk_group=1,
            swiglu_limit=10.0,
        ),
    )


def test_official_fp8_config_maps_hybrid_geometry():
    from freetoken.models.glm5_next.config import parse_config

    config = parse_config(_release_config())

    assert config.num_layers == 45
    assert config.num_moe_layers == 42
    assert config.expert_quant == "fp8_block"
    assert config.weight_block_size == (128, 128)
    assert config.attn_quant == "none"
    assert config.glm5_args.hc_mult == 4
    assert config.glm5_args.index_kpool == 4
    assert config.linear_attention_group().layer_ids[:4] == (0, 1, 2, 4)
    full = config.attention_group_for_layer(3)
    assert full.layer_ids == tuple(range(3, 45, 4))
    assert full.mla and full.head_dim == 512
    assert full.index_ratio == 4


def test_nvfp4_config_keeps_only_routed_experts_quantized():
    from freetoken.models.glm5_next.config import _quant_modes

    config = SimpleNamespace(
        quantization_config={
            "quant_algo": "NVFP4",
            "ignore": [
                "*.self_attn.q_proj",
                "*.mlp.gate_proj",
                "lm_head",
            ],
        }
    )
    assert _quant_modes(config) == ("nvfp4", "none", "none", "none", None)


def test_sparse_block_propagates_release_swiglu_limit(monkeypatch):
    from freetoken.models.glm5_next import moe

    captured = {}

    def fake_make_moe_layer(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(moe, "make_moe_layer", fake_make_moe_layer)
    monkeypatch.setattr(
        moe, "LinearReplicated", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(moe, "Glm5NextMLP", lambda *args, **kwargs: SimpleNamespace())
    config = SimpleNamespace(
        num_experts_per_tok=8,
        num_experts=288,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        hidden_size=128,
        first_k_dense_replace=3,
        n_shared_experts=1,
        moe_intermediate_size=64,
        swiglu_limit=10.0,
    )

    moe.Glm5NextSparseBlock(config, layer_id=3)

    assert captured["extra_attrs"] == {"swiglu_limit": 10.0}
