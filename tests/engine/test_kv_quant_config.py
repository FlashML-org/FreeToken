"""Config-time gates for ``--kv-cache-dtype`` (EngineConfig.kv_quant).

fp8 KV is only half a feature: the pool has to store it AND the attention backend has
to read the scales. Everything here must fail while the config is still a dataclass --
after weights are resident, a wrong combination has already cost a load and (worse)
fi/fa/trtllm would happily attend over raw e4m3 codes and produce plausible garbage.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.models.config import KVCacheGroupSpec


def _spec(name, attn_type, *, mla=False, sliding_window=None, index_head_dim=0, index_ratio=1):
    return KVCacheGroupSpec(
        name=name,
        layer_ids=(0, 1),
        num_kv_heads=1,
        head_dim=64,
        sliding_window=sliding_window,
        mla=mla,
        index_head_dim=index_head_dim,
        num_index_layers=2 if index_head_dim else 0,
        index_ratio=index_ratio,
        attn_type=attn_type,
    )


def _model_config(kind):
    mc = SimpleNamespace(
        model_type=kind,
        single_stream_only=False,
        is_moe=False,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=False,
        num_layers=4,
        rotary_config=SimpleNamespace(max_position=1024),
    )
    specs = {
        "full": (_spec("full", AttnType.FULL),),
        "swa": (
            _spec("full", AttnType.FULL),
            _spec("swa", AttnType.SWA, sliding_window=128),
        ),
        "mla": (_spec("full", AttnType.MLA, mla=True),),
        "dsa": (_spec("full", AttnType.DSA, mla=True, index_head_dim=128),),
        "dsv4": (_spec("dsv4", AttnType.DSV4, sliding_window=128),),
        "bsa": (_spec("full", AttnType.BSA, index_head_dim=128),),
        "qsa": (_spec("full", AttnType.QSA, index_head_dim=128, index_ratio=4),),
    }[kind]
    if kind == "swa":
        mc.has_swa_attention = True
    if kind == "dsv4":
        mc.dsv4_args = SimpleNamespace(window_size=128)
    if kind == "qsa":
        mc.has_linear_attention = True
    mc.kv_cache_group_specs = lambda: specs
    return mc


def _config(kind, **overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )
    object.__setattr__(config, "model_config", _model_config(kind))
    return config


def _patch_fast_machine(monkeypatch):
    """A machine where every fast external backend is available, so an fp8 result can
    only come from the gate and not from a missing package."""
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: False)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: True)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: True)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: True)


def test_kv_quant_spellings():
    from freetoken.engine.engine import _resolve_kv_quant

    assert _resolve_kv_quant("auto") == "none"
    assert _resolve_kv_quant("bf16") == "none"
    assert _resolve_kv_quant("FP8") == "fp8"
    assert _resolve_kv_quant(None) == "none"
    with pytest.raises(ValueError, match="kv-cache-dtype"):
        _resolve_kv_quant("q8")


def test_only_the_backends_that_read_scales_declare_fp8_support():
    from freetoken.attention import SUPPORTED_ATTENTION_BACKENDS, attention_backend_info

    fp8 = {
        name
        for name in SUPPORTED_ATTENTION_BACKENDS.supported_names()
        if attention_backend_info(name).supports_fp8_kv
    }
    # triton serves the plain paged / hybrid-SWA pools and applies the scales in
    # kernel/triton/attention.py; qsa_sparse dequantizes the rows it selects in
    # kernel/triton/qsa/attend.py. Nothing else may join this set: fi/fa/trtllm (and the
    # dsa/dsv4_sparse kernels) have no scale path and would attend over raw e4m3 codes,
    # producing plausible garbage instead of an error.
    assert fp8 == {"triton", "qsa_sparse"}


def test_auto_avoids_the_fast_backends_for_fp8(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_fast_machine(monkeypatch)
    # Same machine where a 16-bit cache auto-selects the sm90 "fa,fi" tree...
    plain = _config("full", attention_backend="auto")
    _adjust_config(plain)
    assert plain.kv_quant == "none"
    assert plain.attention_backend == "fa,fi"

    quantized = _config("full", attention_backend="auto", kv_quant="fp8")
    _adjust_config(quantized)
    assert quantized.attention_backend == "triton"


@pytest.mark.parametrize("backend", ["fi", "fa", "trtllm", "fi,triton", "triton,fi"])
def test_explicit_unsupported_backend_is_rejected(monkeypatch, backend):
    from freetoken.engine.engine import _adjust_config

    _patch_fast_machine(monkeypatch)
    monkeypatch.setattr(
        "freetoken.engine.engine.is_sm100_family", lambda: True
    )  # let trtllm clear its own arch gate first
    config = _config("full", attention_backend=backend, kv_quant="fp8", page_size=1)
    with pytest.raises(ValueError, match="kv-cache-dtype fp8"):
        _adjust_config(config)


def test_explicit_triton_is_accepted(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_fast_machine(monkeypatch)
    config = _config("full", attention_backend="triton", kv_quant="fp8")
    _adjust_config(config)
    assert config.kv_quant == "fp8"


@pytest.mark.parametrize("kind", ["mla", "dsa", "dsv4", "bsa"])
def test_pool_families_without_a_scale_read_path_are_rejected(monkeypatch, kind):
    from freetoken.engine.engine import _adjust_config

    _patch_fast_machine(monkeypatch)  # the rejection must not depend on this box's wheels
    config = _config(kind, attention_backend="auto", kv_quant="fp8")
    with pytest.raises(ValueError, match="kv-cache-dtype fp8"):
        _adjust_config(config)


def test_qsa_keeps_fp8_available(monkeypatch):
    """The QSA pool is the block-sparse family whose K/V rows do reach a kernel that can
    dequantize them; the compressed index keys selection scores against are a separate,
    always-16-bit tier, so the gate has nothing left to refuse here."""
    from freetoken.engine.engine import _adjust_config

    _patch_fast_machine(monkeypatch)
    config = _config("qsa", attention_backend="auto", kv_quant="fp8")
    _adjust_config(config)
    assert config.kv_quant == "fp8"
    assert "qsa_sparse" in config.attention_backend


def test_swa_pool_keeps_fp8_available(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_fast_machine(monkeypatch)
    config = _config("swa", attention_backend="auto", kv_quant="fp8")
    _adjust_config(config)
    assert config.kv_quant == "fp8" and config.attention_backend == "triton"
