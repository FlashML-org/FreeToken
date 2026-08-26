from __future__ import annotations

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.utils.hf import RawConfigShim


def _raw_checkpoint_config() -> RawConfigShim:
    """Raw config shape used when installed Transformers predates Qwen4-Exp."""
    return RawConfigShim(
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "image_token_id": 248056,
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
            "text_config": {
                "model_type": "qwen4_exp_text",
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "qwen_sparse_attention",
                ],
                "head_dim": 256,
                "rope_parameters": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10_000_000,
                    "rope_type": "default",
                },
                "indexer_budget": 2048,
                "max_position_embeddings": 262_144,
                "num_key_value_heads": 2,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "eos_token_id": 248044,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "ngram_vocab_size_base": 20_000_000,
                "split_ngram_parts": 128,
                "indexer_compress_ratio": 4,
                "output_gate_type": "sigmoid",
                "hidden_act": "silu",
                "num_hidden_layers": 4,
                "num_attention_heads": 24,
                "hidden_size": 2560,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "tie_word_embeddings": False,
            },
        }
    )


def test_qwen4_raw_config_uses_official_topk_normalization_default():
    config = parse_config(_raw_checkpoint_config())

    assert config.norm_topk_prob is True
    assert config.rotary_config.max_position == 2048
