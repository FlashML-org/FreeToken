from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
)
from freetoken.models.kimi_k3.attention import KimiDeltaAttention
from freetoken.models.kimi_k3.config import (
    _development_ignores,
    detect_kimi_mxfp4,
    parse_config,
)
from freetoken.moe.benchbw import WORKLOADS
from freetoken.moe.fused_mxfp4 import dequant_mxfp4_blocks, kimi_situ


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _official_config():
    full = list(range(4, 93, 4)) + [93]
    kda = [layer for layer in range(1, 94) if layer not in full]
    quant = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "format": "mxfp4-pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 32,
                    "strategy": "group",
                    "scale_dtype": "torch.uint8",
                },
            }
        },
        "ignore": [
            r"re:.*self_attn.*",
            r"re:.*shared_experts.*",
            r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
            r"re:.*lm_head.*",
            r"re:.*vision_tower.*",
            r"re:.*mm_projector.*",
        ],
    }
    text = _ns(
        num_hidden_layers=93,
        hidden_size=7168,
        num_attention_heads=96,
        num_key_value_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        linear_attn_config={
            "full_attn_layers": full,
            "kda_layers": kda,
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
        routed_expert_hidden_size=3584,
        latent_moe_use_norm=True,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=12,
        mla_use_nope=True,
        mla_use_output_gate=True,
        first_k_dense_replace=1,
        hidden_act="situ",
        moe_router_activation_func="sigmoid",
        topk_method="noaux_tc",
        max_position_embeddings=1048576,
        vocab_size=163840,
        intermediate_size=33792,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        num_experts=896,
        num_experts_per_token=16,
        moe_intermediate_size=3072,
        moe_renormalize=True,
        num_shared_experts=2,
        routed_scaling_factor=1.0,
        num_expert_group=1,
        topk_group=1,
        quantization_config=quant,
    )
    return _ns(
        model_type="kimi_k3",
        architectures=["KimiK3ForConditionalGeneration"],
        text_config=text,
    )


def _development_config():
    config = _official_config()
    text = config.text_config
    text.num_hidden_layers = 8
    text.hidden_size = 1024
    text.num_attention_heads = 8
    text.num_key_value_heads = 8
    text.q_lora_rank = 256
    text.kv_lora_rank = 128
    text.qk_nope_head_dim = 64
    text.qk_rope_head_dim = 32
    text.v_head_dim = 64
    text.linear_attn_config = {
        "full_attn_layers": [4, 8],
        "kda_layers": [1, 2, 3, 5, 6, 7],
        "num_heads": 8,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    }
    text.routed_expert_hidden_size = 512
    text.attn_res_block_size = 4
    text.max_position_embeddings = 4096
    text.intermediate_size = 2048
    text.num_experts = 8
    text.num_experts_per_token = 2
    text.moe_intermediate_size = 256
    text.num_shared_experts = 1
    quant = copy.deepcopy(text.quantization_config)
    group = quant["config_groups"]["group_0"]
    group["targets"] = [r"re:.*block_sparse_moe.*"]
    group["input_activations"] = {
        "dynamic": True,
        "num_bits": 4,
        "type": "float",
        "group_size": 32,
        "strategy": "group",
        "scale_dtype": "torch.uint8",
    }
    quant["ignore"] = sorted(_development_ignores(text))
    del text.quantization_config
    config.quantization_config = quant
    return config


class KimiK3ConfigTests(unittest.TestCase):
    def test_kda_conv_weight_matches_activation_dtype(self):
        import torch

        # The released checkpoint holds KDA convolution weights in FP32, while
        # the fused causal-convolution kernel requires same-typed activations
        # and weights on the BF16 inference path.
        attn = object.__new__(KimiDeltaAttention)
        attn.q_conv1d = _ns(weight=torch.ones(2, 1, 3, dtype=torch.float32))
        attn.k_conv1d = _ns(weight=torch.full((2, 1, 3), 2.0, dtype=torch.float32))
        attn.v_conv1d = _ns(weight=torch.full((2, 1, 3), 3.0, dtype=torch.float32))

        got = attn._conv_weight(torch.bfloat16)

        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(got.float(), torch.tensor([[1.0] * 3] * 2 + [[2.0] * 3] * 2 + [[3.0] * 3] * 2))

    def test_kda_decay_adapter_matches_official_bounded_gate(self):
        import torch
        import torch.nn.functional as F

        # Build only the scalar state needed by the adapter. Constructing the
        # 753B model just to validate this algebra would obscure the contract.
        attn = object.__new__(KimiDeltaAttention)
        attn.num_heads = 2
        attn.gate_lower_bound = -5.0
        attn.A_log = torch.tensor([-0.7, 0.2, 0.9], dtype=torch.float32)
        attn.dt_bias = torch.tensor(
            [-0.4, 0.1, 0.5, -0.4, 0.1, 0.5], dtype=torch.float32
        )
        a = torch.tensor([[-2.0, -0.5, 0.0, 0.25, 1.0, 3.0]], dtype=torch.float32)

        adapted = attn._safe_a(a)
        actual = -F.softplus(adapted + attn.dt_bias)
        decay = attn.A_log.exp().repeat(attn.num_heads)
        expected = attn.gate_lower_bound * torch.sigmoid(decay * (a + attn.dt_bias))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_mxfp4_reserved_scale_byte_uses_largest_finite_exponent(self):
        import torch

        # Every nibble is +0.5. Scale 0xff must follow the production-kernel
        # contract and clamp to exponent byte 254, not become an infinity.
        blocks = torch.full((1, 1, 16), 0x11, dtype=torch.uint8)
        scales = torch.full((1, 1), 255, dtype=torch.uint8)
        got = dequant_mxfp4_blocks(blocks, scales, out_dtype=torch.float32)
        expected = torch.full((1, 32), 0.5 * 2.0**127, dtype=torch.float32)
        torch.testing.assert_close(got, expected)

    def test_situ_interleaved_matches_checkpoint_formula(self):
        import torch

        gate_up = torch.tensor([[1.0, 2.0, -3.0, 4.0]], dtype=torch.float32)
        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
        expected = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        expected *= 25.0 * torch.tanh(up / 25.0)
        torch.testing.assert_close(kimi_situ(gate_up), expected)

    def test_bandwidth_probe_uses_real_kimi_expert_geometry(self):
        workload = WORKLOADS["kimi-k3"]
        self.assertEqual(
            (workload.hidden, workload.inter, workload.experts, workload.top_k),
            (3584, 3072, 896, 16),
        )
        self.assertEqual(workload.formats, ("mxfp4_triton",))
        self.assertEqual(workload.activation, "situ")
        self.assertEqual((workload.swiglu_alpha, workload.swiglu_limit), (4.0, 25.0))

    def test_public_checkpoint_geometry_is_normalized(self):
        config = parse_config(_official_config())
        self.assertEqual(config.num_layers, 93)
        self.assertEqual(config.num_experts, 896)
        self.assertEqual(config.num_experts_per_tok, 16)
        self.assertEqual(config.expert_quant, "mxfp4")
        self.assertEqual(config.moe_weight_format, "mxfp4")
        self.assertEqual(config.num_moe_layers, 92)
        self.assertEqual(config.kimi_k3_args.routed_expert_hidden_size, 3584)
        self.assertEqual(config.kimi_k3_args.mla_latent_dim, 576)
        linear = next(
            g
            for g in config.attention_groups
            if isinstance(g, LinearGatedDeltaGroupConfig)
        )
        mla = next(
            g
            for g in config.attention_groups
            if isinstance(g, FullAttentionGroupConfig)
        )
        self.assertEqual(len(linear.layer_ids), 69)
        self.assertEqual(len(mla.layer_ids), 24)
        self.assertEqual(mla.layer_ids[:2], (3, 7))
        self.assertEqual(mla.layer_ids[-1], 92)
        self.assertEqual(set(linear.layer_ids) | set(mla.layer_ids), set(range(93)))
        self.assertFalse(config.supports_hybrid_radix)
        self.assertTrue(mla.mla)
        self.assertEqual(mla.head_dim, 576)
        self.assertEqual(mla.rotary_config.rotary_dim, 0)

    def test_development_checkpoint_geometry_and_top_level_quantization(self):
        config = parse_config(_development_config())
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.num_moe_layers, 7)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.kimi_k3_args.routed_expert_hidden_size, 512)
        self.assertEqual(config.kimi_k3_args.kda_head_dim, 32)
        self.assertIsNone(config.kimi_k3_args.kda_gate_lower_bound)
        self.assertEqual(config.expert_quant, "mxfp4")

    def test_mxfp4_detector_fails_closed_on_nvfp4_geometry(self):
        config = _official_config()
        quant = copy.deepcopy(config.text_config.quantization_config)
        quant["config_groups"]["group_0"]["weights"].update(
            group_size=16, strategy="tensor_group"
        )
        config.text_config.quantization_config = quant
        with self.assertRaisesRegex(ValueError, "unsupported Kimi-K3 MXFP4 geometry"):
            detect_kimi_mxfp4(config.text_config)

    def test_mxfp4_detector_rejects_dense_quantization_contract(self):
        config = _official_config()
        config.text_config.quantization_config["ignore"].remove(r"re:.*self_attn.*")
        with self.assertRaisesRegex(ValueError, "routed-expert-only"):
            detect_kimi_mxfp4(config.text_config)

    def test_incomplete_attention_map_is_rejected_before_allocation(self):
        config = _official_config()
        config.text_config.linear_attn_config["kda_layers"].pop()
        with self.assertRaisesRegex(ValueError, "attention layer map is incomplete"):
            parse_config(config)

    def test_changed_situ_contract_is_rejected(self):
        config = _official_config()
        config.text_config.activation_situ_beta = 3.0
        with self.assertRaisesRegex(ValueError, "SiTU kernels require"):
            parse_config(config)


if __name__ == "__main__":
    unittest.main()
