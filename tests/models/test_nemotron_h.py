from types import SimpleNamespace

import torch

from freetoken.distributed import get_tp_info, set_tp_info
from freetoken.models.nemotron_h.config import parse_config
from freetoken.models.nemotron_h.model import NemotronHForCausalLM
from freetoken.models.register import get_model_spec
from freetoken.moe.expert_banks import bank_bytes_estimate
from freetoken.moe.fused_nvfp4 import _run_act
from freetoken.utils import torch_dtype


def _hf_config():
    layers = ["mamba", "moe", "mamba", "attention", "moe"]
    quantized = {
        "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
        "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
        "backbone.layers.1.mixer.fc1_latent_proj": {"quant_algo": "FP8"},
        "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "NVFP4"},
        "backbone.layers.1.mixer.experts.0.down_proj": {"quant_algo": "NVFP4"},
    }
    return SimpleNamespace(
        layers_block_type=layers,
        quantization_config={"quant_algo": "MIXED_PRECISION", "quantized_layers": quantized},
        head_dim=128,
        max_position_embeddings=262144,
        rope_theta=10000.0,
        n_groups=8,
        mamba_num_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        conv_kernel=4,
        chunk_size=128,
        moe_latent_size=1024,
        moe_shared_expert_intermediate_size=5376,
        num_key_value_heads=2,
        num_hidden_layers=len(layers),
        num_attention_heads=32,
        hidden_size=4096,
        vocab_size=131072,
        intermediate_size=2688,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        n_routed_experts=512,
        num_experts_per_tok=22,
        moe_intermediate_size=2688,
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        n_group=1,
        topk_group=1,
        model_type="nemotron_h",
        architectures=["NemotronHForCausalLM"],
    )


def test_config_maps_mixer_and_expert_geometry():
    config = parse_config(_hf_config())
    assert config.num_moe_layers == 2
    assert config.moe_layer_ids == (1, 4)
    assert config.expert_hidden_size == 1024
    assert not config.expert_gated
    assert config.single_stream_only
    assert config.attention_groups[0].layer_ids == (0, 2)
    assert config.attention_groups[1].layer_ids == (3,)
    assert config.nemotron_h_args.module_quant(
        "backbone.layers.0.mixer.in_proj"
    ) == "fp8_pertensor"


def test_offload_model_has_no_resident_expert_tensors():
    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(0, 1)
    config = parse_config(_hf_config())
    object.__setattr__(config, "moe_backend", "offload")
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = NemotronHForCausalLM(config)
    state = model.state_dict()
    assert not any(".experts." in key for key in state)
    assert state["backbone.layers.0.mixer.in_proj.weight"].dtype == torch.float8_e4m3fn
    assert state["backbone.layers.1.mixer.gate.weight"].shape == (512, 4096)


def test_ungated_nvfp4_bank_estimate():
    config = parse_config(_hf_config())
    H, I = 1024, 2688
    per_expert = I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2)
    assert bank_bytes_estimate(config) == 2 * 512 * per_expert


def test_relu2_expert_activation_is_ungated():
    x = torch.tensor([[-2.0, 0.5, 3.0]])
    out = torch.empty_like(x)
    _run_act("relu2", x, out, 1.702, 7.0)
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.25, 9.0]]))


def test_registry_entry():
    spec = get_model_spec("NemotronHForCausalLM")
    assert spec.module == "freetoken.models.nemotron_h"
