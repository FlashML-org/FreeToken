"""Unified-memory (UMA) MoE strategy resolution (DGX Spark / GB10, Jetson).

CPU-only: exercises _is_unified_memory_gpu (env override) and
_fused_resident_ok (format gating) without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import _fused_resident_ok, _is_unified_memory_gpu


@pytest.fixture
def uma_env(monkeypatch):
    def set_env(value: str | None):
        if value is None:
            monkeypatch.delenv("FREETOKEN_UNIFIED_MEMORY", raising=False)
        else:
            monkeypatch.setenv("FREETOKEN_UNIFIED_MEMORY", value)

    return set_env


def test_env_override_forces_unified(uma_env):
    uma_env("1")
    assert _is_unified_memory_gpu() is True
    uma_env("true")
    assert _is_unified_memory_gpu() is True


def test_env_override_forces_discrete(uma_env):
    # must win even on real UMA hardware: this is the escape hatch when
    # cudaDevAttrIntegrated lies
    uma_env("0")
    assert _is_unified_memory_gpu() is False
    uma_env("off")
    assert _is_unified_memory_gpu() is False


def test_probe_falls_back_cleanly(uma_env, monkeypatch):
    # no CUDA / probe failure -> discrete (offload stays the safe default)
    uma_env(None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _is_unified_memory_gpu() is False


def test_fused_resident_ok_plain_formats():
    assert _fused_resident_ok(SimpleNamespace(expert_quant="none")) is True
    assert _fused_resident_ok(SimpleNamespace(expert_quant="none", moe_weight_format="bf16")) is True
    assert _fused_resident_ok(SimpleNamespace(expert_quant="fp8_block")) is True


def test_adjust_config_selects_fused_on_uma(monkeypatch):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    import freetoken.engine.engine as engine_module

    monkeypatch.setattr(engine_module, "_is_unified_memory_gpu", lambda index=None: True)
    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        attention_backend="fi",
        moe_cache_rate=0.3,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="none",
            moe_strategy="auto",
        ),
    )

    engine_module._adjust_config(config)

    assert config.moe_strategy == "fused"
    assert config.moe_cache_size == 0
    assert config.moe_cache_rate is None


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_fused_resident_ok_rejects_quantized_experts(fmt):
    # mirrors the engine's own fused guard; nvfp4 has resident_view but the
    # combination is not yet validated
    assert _fused_resident_ok(SimpleNamespace(expert_quant=fmt)) is False


@pytest.mark.parametrize("fmt", ["mxfp4", "q4_0"])
def test_fused_resident_ok_rejects_gguf_weight_formats(fmt):
    # GGUF banks dispatch on the offload cache's format tag; no resident path
    assert _fused_resident_ok(SimpleNamespace(expert_quant="none", moe_weight_format=fmt)) is False
