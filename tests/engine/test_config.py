"""EngineConfig.model_config: the config copy that family parsers are handed."""

from __future__ import annotations

import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.models.config import ModelConfig, RotaryConfig


def _minimal_model_config() -> ModelConfig:
    return ModelConfig(
        num_layers=1, num_qo_heads=1, num_kv_heads=1, head_dim=8, hidden_size=8, vocab_size=8,
        intermediate_size=8, rms_norm_eps=1e-6, hidden_act="silu", tie_word_embeddings=True,
        num_experts=0, num_experts_per_tok=1, moe_intermediate_size=8, norm_topk_prob=True,
        model_type="gemma4", architectures=["Gemma4GGUFForCausalLM"],
        rotary_config=RotaryConfig(head_dim=8, rotary_dim=8, max_position=128, base=10_000.0, scaling=None),
    )


def test_a_frozen_gguf_shim_keeps_the_sections_it_does_not_have(monkeypatch):
    """A GGUF checkpoint's config is a frozen dataclass (GgufConfigShim) with no encoder section
    at all, so clearing the towers this process does not build must not assign one to it."""
    import freetoken.engine.config as engine_config
    from freetoken.models.gguf.config import GgufConfigShim

    shim = GgufConfigShim(
        architectures=["Gemma4GGUFForCausalLM"], model_path="/nonexistent", model_type="gemma4",
        metadata={}, vocab_size=8, tie_word_embeddings=True,
    )
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path, overrides: shim)
    monkeypatch.setattr(engine_config, "checkpoint_quant_config", lambda *args: None)

    seen: dict[str, object] = {}

    def _parse(hf_config):
        seen["vision_config"] = getattr(hf_config, "vision_config", None)
        seen["audio_config"] = getattr(hf_config, "audio_config", None)
        seen["is_the_checkpoint_config"] = hf_config is shim
        return _minimal_model_config()

    monkeypatch.setattr(engine_config, "_load_attr", lambda module, attr: _parse)
    config = EngineConfig(model_path="/nonexistent", tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16)

    assert config.model_config.model_type == "gemma4"
    assert seen == {"vision_config": None, "audio_config": None, "is_the_checkpoint_config": False}
