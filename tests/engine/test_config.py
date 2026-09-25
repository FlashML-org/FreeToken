from dataclasses import dataclass
from types import SimpleNamespace

import torch

import freetoken.engine.config as engine_config_module
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig


@dataclass(frozen=True)
class _FrozenHfConfig:
    architectures: list[str]


@dataclass(frozen=True)
class _ParsedConfig:
    quant: object = None


def test_model_config_normalizes_encoder_sections_on_frozen_copy(monkeypatch):
    """Disabled encoder fields must not require a mutable Transformers config."""
    source_config = _FrozenHfConfig(architectures=["unused"])
    config = EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
    )
    object.__setattr__(config, "hf_config", source_config)
    object.__setattr__(
        config,
        "model_spec",
        SimpleNamespace(encoders=(), module="unused", parse_config="unused"),
    )
    object.__setattr__(config, "active_encoders", ())

    parsed = {}
    quant = object()

    def parse_config(hf_config):
        parsed["vision_config"] = hf_config.vision_config
        parsed["audio_config"] = hf_config.audio_config
        return _ParsedConfig()

    monkeypatch.setattr(engine_config_module, "checkpoint_quant_config", lambda *args: quant)
    monkeypatch.setattr(engine_config_module, "set_quant_config", lambda value: None)
    monkeypatch.setattr(engine_config_module, "_load_attr", lambda *args: parse_config)

    assert config.model_config.quant is quant
    assert parsed == {"vision_config": None, "audio_config": None}
    assert not hasattr(source_config, "vision_config")
    assert not hasattr(source_config, "audio_config")
