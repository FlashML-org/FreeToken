"""ROCm (AMD) attention-backend resolution: no NVIDIA-native backend may be selected.

On ROCm, is_sm90/100 gates are False and flashinfer/sgl_kernel are treated as
unavailable, so auto resolution must fall through to the portable Triton backend
for FULL-attention models.
"""

import pytest


def _engine_config(**overrides):
    from types import SimpleNamespace

    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=False,
            num_layers=10,
            expert_quant="none",
        ),
    )
    return config


def _patch_rocm(monkeypatch):
    from freetoken.engine import engine
    from freetoken.kernel import backend

    monkeypatch.setattr(engine, "is_sm100_family", lambda: False)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: False)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: False)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: False)
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(engine, "is_rocm", lambda: True)


def test_rocm_auto_resolves_to_triton(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_rocm(monkeypatch)
    config = _engine_config(attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == "triton"


def test_rocm_explicit_nvidia_backend_rejected(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_rocm(monkeypatch)
    # flashinfer/sgl absent and arch gates false -> trtllm/fa/fi requirements unmet.
    for backend in ("fi", "fa", "trtllm"):
        config = _engine_config(attention_backend=backend)
        with pytest.raises(RuntimeError):
            _adjust_config(config)


def test_sgl_flash_attn_unavailable_on_rocm(monkeypatch):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    assert engine._sgl_flash_attn_available() is False
