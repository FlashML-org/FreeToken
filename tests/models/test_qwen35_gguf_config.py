"""Unit tests for the metadata-only Qwen3.5 MoE GGUF configuration adapter.

These tests use the public Qwen3.6-35B-A3B GGUF geometry recorded on GMKtec EVO-X2.
They prove the parser's architecture translation without requiring a 22 GiB model
file or a GPU in the test process.
"""

from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.qwen3_5_moe.config import parse_gguf_config


def _qwen35moe_shim() -> GgufConfigShim:
    """Return a minimal Qwen3.6-35B-A3B GGUF metadata shim for parser coverage."""
    return GgufConfigShim(
        architectures=["Qwen3_5MoeGGUFForCausalLM"],
        model_path="qwen35b-a3b-q4-k-m.gguf",
        model_type="qwen35moe",
        metadata={
            "qwen35moe.block_count": 40,
            "qwen35moe.context_length": 262144,
            "qwen35moe.embedding_length": 2048,
            "qwen35moe.attention.head_count": 16,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_used_count": 8,
            "qwen35moe.expert_feed_forward_length": 512,
            "qwen35moe.expert_shared_feed_forward_length": 512,
            "qwen35moe.ssm.conv_kernel": 4,
            "qwen35moe.ssm.state_size": 128,
            "qwen35moe.ssm.group_count": 16,
            "qwen35moe.ssm.inner_size": 4096,
            "qwen35moe.full_attention_interval": 4,
            "qwen35moe.rope.dimension_count": 64,
            "qwen35moe.rope.freq_base": 10_000_000.0,
        },
        vocab_size=151936,
        tie_word_embeddings=False,
    )


def test_qwen35moe_gguf_metadata_maps_to_the_official_hybrid_geometry():
    """Qwen's GGUF SSM fields recreate the published Gated DeltaNet dimensions."""
    config = parse_gguf_config(_qwen35moe_shim())

    assert (config.num_layers, config.hidden_size, config.num_experts) == (40, 2048, 256)
    assert (config.num_qo_heads, config.num_kv_heads, config.head_dim) == (16, 2, 256)
    assert (config.num_experts_per_tok, config.moe_intermediate_size) == (8, 512)
    assert config.rotary_config.rotary_dim == 64
    assert (config.expert_quant, config.moe_weight_format) == ("q4_k_q5_k", "q4_k_q5_k")
    assert config.gguf_q6_down_layer_ids == ()  # metadata-only shim has no tensor table

    linear, full = config.attention_groups
    assert linear.layer_ids == tuple(index for index in range(40) if (index + 1) % 4)
    assert full.layer_ids == tuple(index for index in range(40) if not (index + 1) % 4)
    assert (linear.num_key_heads, linear.num_value_heads) == (16, 32)
    assert (linear.key_head_dim, linear.value_head_dim, linear.conv_kernel_dim) == (128, 128, 4)


def test_qwen35moe_gguf_rejects_an_invalid_ssm_value_head_partition():
    """A malformed GGUF cannot silently create a Gated DeltaNet with fractional heads."""
    shim = _qwen35moe_shim()
    metadata = dict(shim.metadata)
    metadata["qwen35moe.ssm.inner_size"] = 4095
    malformed = GgufConfigShim(
        architectures=shim.architectures,
        model_path=shim.model_path,
        model_type=shim.model_type,
        metadata=metadata,
        vocab_size=shim.vocab_size,
        tie_word_embeddings=shim.tie_word_embeddings,
    )

    try:
        parse_gguf_config(malformed)
    except ValueError as exc:
        assert "value-head groups" in str(exc)
    else:
        raise AssertionError("expected malformed Gated DeltaNet geometry to be rejected")


def test_qwen35_dense_gguf_metadata_maps_to_dense_hybrid_geometry():
    """Qwen3.8's qwen35 metadata selects the existing dense hybrid model branch."""
    base = _qwen35moe_shim()
    metadata = {
        key.replace("qwen35moe.", "qwen35.", 1): value
        for key, value in base.metadata.items()
        if "expert_" not in key and key != "qwen35moe.expert_count"
    }
    metadata.update(
        {
            "qwen35.block_count": 64,
            "qwen35.embedding_length": 5120,
            "qwen35.feed_forward_length": 17408,
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.ssm.inner_size": 6144,
            "qwen35.ssm.time_step_rank": 48,
        }
    )
    dense = GgufConfigShim(
        architectures=["Qwen3_5MoeGGUFForCausalLM"],
        model_path="qwen38-27b-q4-k-m.gguf",
        model_type="qwen35",
        metadata=metadata,
        vocab_size=248320,
        tie_word_embeddings=False,
    )

    config = parse_gguf_config(dense)

    assert (config.num_layers, config.hidden_size, config.intermediate_size) == (64, 5120, 17408)
    assert config.num_experts == 0
    assert config.moe_enabled is False
    assert config.moe_weight_format == "qwen35_dense"
    assert config.expert_quant == "none"
