"""Resident Qwen3.5 MoE construction and metadata-only budget tests."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.config import ModelConfig, RotaryConfig
from freetoken.models.gguf.dequant import GGML_Q5_K, GGML_Q6_K
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE


@pytest.fixture(scope="module", autouse=True)
def _single_rank_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _config(*, expert_quant="none", moe_weight_format=None, down_types=()):
    return ModelConfig(
        num_layers=1,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=64,
        hidden_size=256,
        vocab_size=320,
        intermediate_size=0,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(64, 64, 1024, 10000.0, None),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=256,
        norm_topk_prob=True,
        model_type="qwen3_5_moe",
        architectures=["Qwen35moeForCausalLM"],
        moe_enabled=True,
        expert_quant=expert_quant,
        moe_weight_format=moe_weight_format,
        gguf_down_quant_types=down_types,
        shared_expert_intermediate_size=256,
    )


@pytest.mark.parametrize("down_type", [GGML_Q5_K, GGML_Q6_K])
def test_gguf_resident_construction_uses_native_packed_shapes(down_type):
    config = _config(moe_weight_format="gguf", expert_quant="gguf", down_types=(down_type,))
    with torch.device("meta"):
        moe = Qwen3_5MoE(config, layer_id=0)
    experts = moe.experts
    assert experts.weight_format == "gguf"
    assert experts.gate_up_proj.shape == (2, 512, 144)
    assert experts.down_proj.shape[-1] == (176 if down_type == GGML_Q5_K else 210)
    assert experts.gate_up_proj.dtype == torch.uint8
    assert experts.down_proj.dtype == torch.uint8


def test_bf16_resident_construction_unchanged():
    config = _config()
    with torch.device("meta"):
        moe = Qwen3_5MoE(config, layer_id=0)
    assert moe.experts.weight_format == "bf16"
    assert moe.experts.gate_up_proj.shape == (2, 512, 256)
    assert moe.experts.down_proj.shape == (2, 256, 256)


def test_fp8_resident_construction_unchanged():
    config = _config(expert_quant="fp8_block")
    with torch.device("meta"):
        moe = Qwen3_5MoE(config, layer_id=0)
    assert moe.experts.weight_format == "fp8_block"
    assert moe.experts.gate_up_proj.shape == (2, 512, 256)


def test_resident_budget_is_allocation_free(monkeypatch):
    from freetoken.engine import resident_budget

    class Pool:
        @classmethod
        def kv_cost(cls, _config):
            return 1024, 2048, 16, 0

    model_config = SimpleNamespace(
        moe_weight_format="gguf",
        linear_attention_group=lambda: None,
    )
    config = SimpleNamespace(
        model_path="synthetic.gguf",
        model_config=model_config,
        max_seq_len=64,
        page_size=16,
        max_running_req=1,
        cache_type="naive",
        tp_info=SimpleNamespace(size=1),
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(resident_budget, "_gguf_payload_bytes", lambda _path: (10000, 700))
    monkeypatch.setattr(resident_budget, "_total_vram_bytes", lambda _free: 8 * (1 << 30))
    monkeypatch.setattr("freetoken.kvcache.resolve_pool_class", lambda _mc: Pool)
    monkeypatch.setattr("freetoken.kvcache.linear_state_pool.state_pool_bytes", lambda *_a, **_k: 3000)

    budget = resident_budget.estimate_gguf_resident_budget(config.model_path, config, 2 * (1 << 30))

    assert budget.packed_model_bytes == 10000
    assert budget.kv_bytes == 6144
    assert budget.gdn_state_bytes == 3000
    assert budget.graph_reserve_bytes == 768 * (1 << 20)
    assert budget.peak_load_scratch_bytes == 512 * (1 << 20)
    assert budget.safety_bytes == 1_500 * (1 << 20)
    assert budget.required_bytes > budget.free_bytes
    assert budget.fits is False
