from types import SimpleNamespace


def _glm53_nvfp4_config():
    """Small projection of LibertAIDAI/GLM-5.3-NVFP4 config.json."""
    return SimpleNamespace(
        model_type="glm_moe_dsa",
        architectures=["GlmMoeDsaForCausalLM"],
        quantization_config={"quant_algo": "NVFP4", "quant_method": "modelopt"},
        hidden_size=6144,
        num_hidden_layers=78,
        vocab_size=154880,
        intermediate_size=12288,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        max_position_embeddings=1048576,
        rope_parameters={"rope_theta": 8000000, "rope_type": "default"},
        num_attention_heads=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        rope_interleave=True,
        indexer_rope_interleave=True,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        indexer_types=[
            "full" if layer < 3 or (layer - 2) % 4 == 0 else "shared"
            for layer in range(78)
        ],
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        norm_topk_prob=True,
        first_k_dense_replace=3,
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        attention_bias=False,
    )


def test_glm53_nvfp4_maps_existing_glm_moe_dsa_path():
    from freetoken.models.glm_moe_dsa.config import parse_config

    config = parse_config(_glm53_nvfp4_config())

    assert config.architectures == ["GlmMoeDsaForCausalLM"]
    assert (config.num_layers, config.hidden_size) == (78, 6144)
    assert (config.num_experts, config.num_experts_per_tok) == (256, 8)
    assert config.expert_quant == "nvfp4"
    assert config.glm_dsa_args.qk_head_dim == 256
    assert config.glm_dsa_args.index_topk == 2048
    full = config.attention_groups[0]
    assert full.mla and full.head_dim == 576
    assert full.num_index_layers == 21


def test_glm53_is_primary_full_glm_aot_checkpoint():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS

    entry = next(m for m in SUPPORTED_MODELS if m.architecture == "GlmMoeDsaForCausalLM")
    assert entry.name == "LibertAIDAI/GLM-5.3-NVFP4"
    assert "nvidia/GLM-5.2-NVFP4" in entry.aliases
    assert (entry.hidden_size, entry.moe_intermediate_size, entry.top_k) == (6144, 2048, 8)
