"""Captured NVFP4 configs: s-zaizen 179b7cda and LibertAIDAI dfce15b9."""

import json
from pathlib import Path

import pytest

from freetoken.models.deepseek_v41.args import load_args
from freetoken.models.deepseek_v41.config import parse_config


@pytest.fixture
def checkpoint_config():
    return json.loads((Path(__file__).parent / "fixtures/deepseek_v41_nvfp4_config.json").read_text())


@pytest.fixture
def libertai_config():
    return json.loads((Path(__file__).parent / "fixtures/deepseek_v41_libertai_nvfp4_config.json").read_text())


def test_target_mixed_quant_and_multimodal_config(checkpoint_config):
    config = parse_config(checkpoint_config)
    args = config.dsv41_args
    assert (config.num_layers, config.hidden_size, config.num_experts, config.num_experts_per_tok) == (40, 5120, 384, 6)
    assert config.expert_quant == "nvfp4"
    assert config.weight_block_size == (32, 32)
    assert (config.hidden_act, config.hidden_act_alpha, config.swiglu_limit) == ("swiglu_clamp", 1.0, 10.0)
    assert config.is_multimodal and args.vision_enabled
    assert args.vision_n_layers == 32 and config.image_token_id == 129264
    assert args.engram_layer_ids == (1, 14)
    assert (args.engram_dtype, args.engram_block_size, args.engram_scale_fmt) == ("fp8", 32, "ue8m0")
    assert args.kv_source_layers == (2, 8, 14, 20)
    assert args.index_source_layers == (2, 8, 14, 20, 24, 28, 32, 36)
    assert len(args.compress_ratios) == 43 and args.n_mtp_layers == 3
    assert config.attention_groups[0].layer_ids == tuple(range(40))
    assert config.rotary_config.max_position == 1048576
    assert config.rms_norm_eps == 1e-20


def test_parse_needs_no_local_inference_file(checkpoint_config):
    from freetoken.utils.hf import RawConfigShim

    hf_config = RawConfigShim(checkpoint_config, _name_or_path="s-zaizen/DeepSeek-V4.1-Flash-NVFP4")
    assert parse_config(hf_config) == parse_config(checkpoint_config)


def test_explicit_vision_disable_preserves_native_quantization(checkpoint_config):
    from freetoken.models.register import checkpoint_quant_config, get_model_spec
    from freetoken.layers.quantization import QuantKind

    checkpoint_config["vision_config"] = None
    config = parse_config(checkpoint_config)
    assert not config.is_multimodal and not config.dsv41_args.vision_enabled
    spec = get_model_spec("DeepseekV41ForCausalLM")
    assert spec.encoders[0].modalities == ("image",) and spec.mm_processor is not None
    quant = checkpoint_quant_config("unused", checkpoint_config, spec)
    assert quant.scheme_for_name("layers.2.attn.wq_a").kind is QuantKind.FP8_BLOCK
    assert quant.scheme_for_name("layers.2.ffn.experts").kind is QuantKind.NVFP4


def test_libertai_native_nvfp4_with_packed_engram(libertai_config):
    from freetoken.utils.hf import RawConfigShim

    config = parse_config(libertai_config)
    args = config.dsv41_args
    assert config.expert_quant == "nvfp4" and config.weight_block_size == (32, 32)
    assert (config.num_layers, config.hidden_size, config.num_experts) == (40, 5120, 384)
    assert config.is_multimodal and args.vision_n_layers == 32
    assert (args.engram_dtype, args.engram_block_size, args.engram_scale_fmt) == ("fp4", 32, "ue8m0")
    assert args.engram_num_embeddings == (384006168, 384016682)
    assert "moe_quant_algo" not in libertai_config["quantization_config"]
    assert "quantized_layers" not in libertai_config["quantization_config"]
    assert parse_config(RawConfigShim(libertai_config)) == config


@pytest.mark.parametrize("key,value,match", [
    ("expert_dtype", "fp4", "NVFP4"),
    ("expert_block_size", 32, "group_size"),
    ("expert_scale_fmt", "ue8m0", "E4M3"),
    ("expert_global_scale", False, "global scale"),
    ("engram_dtype", "bf16", "FP8 or FP4"),
    ("engram_block_size", 16, "block_size=32"),
    ("engram_scale_fmt", "e4m3", "UE8M0"),
])
def test_reject_incompatible_native_storage(libertai_config, key, value, match):
    libertai_config["quantization_config"][key] = value
    with pytest.raises(ValueError, match=match):
        parse_config(libertai_config)


def test_engram_quant_metadata_overrides_text_storage(libertai_config, tmp_path):
    libertai_config["text_config"]["engram_dtype"] = "fp8"
    (tmp_path / "config.json").write_text(json.dumps(libertai_config))
    assert load_args(tmp_path).engram_dtype == "fp4"
    assert load_args(tmp_path, engram_dtype="fp8").engram_dtype == "fp8"


def test_legacy_fp4_label_alone_does_not_imply_nvfp4(checkpoint_config):
    del checkpoint_config["quantization_config"]["moe_quant_algo"]
    del checkpoint_config["quantization_config"]["quantized_layers"]
    with pytest.raises(ValueError, match="NVFP4"):
        parse_config(checkpoint_config)


@pytest.mark.parametrize("key,value,match", [
    ("moe_quant_algo", "MXFP4", "NVFP4"),
    ("group_size", 32, "group_size"),
    ("weight_block_size", [128, 128], "32x32"),
    ("scale_fmt", "float", "UE8M0"),
])
def test_reject_incompatible_quantization(checkpoint_config, key, value, match):
    checkpoint_config["quantization_config"][key] = value
    with pytest.raises(ValueError, match=match):
        parse_config(checkpoint_config)


def test_reject_missing_layer_quantization(checkpoint_config):
    del checkpoint_config["quantization_config"]["quantized_layers"]["layers.39.ffn.experts"]
    with pytest.raises(ValueError, match="backbone layer 39"):
        parse_config(checkpoint_config)


def test_infer_nvfp4_from_per_layer_quantization(checkpoint_config):
    del checkpoint_config["quantization_config"]["moe_quant_algo"]
    assert parse_config(checkpoint_config).expert_quant == "nvfp4"


@pytest.mark.parametrize("field,value", [
    ("kv_source_layer_ids", [2, 8, 14]),
    ("index_source_layer_ids", [8, 14, 20]),
    ("compress_ratios", [0] * 39),
    ("engram_num_embeddings", [4]),
    ("gate_temp", 0),
])
def test_invalid_attention_or_engram_layout(checkpoint_config, field, value):
    checkpoint_config["text_config"][field] = value
    with pytest.raises(ValueError):
        load_args(checkpoint_config)
