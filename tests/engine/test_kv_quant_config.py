from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.models.config import KVCacheGroupSpec


def _config(*, dtype=torch.bfloat16, tp=1, kind=AttnType.FULL, **kwargs):
    spec = KVCacheGroupSpec(name="full", layer_ids=tuple(range(48)), num_kv_heads=4,
                            head_dim=128, attn_type=kind, sliding_window=None)
    config = EngineConfig(model_path="/tmp/fp8-kv-test", dtype=dtype,
                          tp_info=DistributedInfo(rank=0, size=tp), **kwargs)
    object.__setattr__(config, "model_config", SimpleNamespace(
        has_swa_attention=kind == AttnType.SWA, has_linear_attention=False,
        is_moe=False, num_layers=48, expert_quant="none",
        kv_cache_group_specs=lambda: (spec,),
    ))
    return config


@pytest.fixture(autouse=True)
def _hardware(monkeypatch):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: True)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: False)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: True)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: True)


def test_fp8_auto_selects_a_scale_aware_backend():
    from freetoken.engine.engine import _adjust_config

    config = _config(kv_quant="fp8")
    _adjust_config(config)
    assert config.attention_backend == "triton"


@pytest.mark.parametrize("backend", ["fi", "fa", "trtllm", "triton,fi"])
def test_fp8_rejects_backends_that_do_not_read_row_scales(backend):
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="requires --attention-backend triton"):
        _adjust_config(_config(kv_quant="fp8", attention_backend=backend))


@pytest.mark.parametrize("kwargs,match", [
    ({"dtype": torch.float16}, "bfloat16"),
    ({"tp": 2}, "TP=1"),
    ({"speculative_algorithm": "mtp"}, "speculative decoding"),
    ({"kind": AttnType.SWA}, "MHA/GQA"),
    ({"kind": AttnType.MLA}, "MHA/GQA"),
    ({"kind": AttnType.QSA}, "MHA/GQA"),
    ({"kv_quant": "int8"}, "unknown --kv-cache-dtype"),
])
def test_unsupported_fp8_configs_fail_before_weights_load(kwargs, match):
    from freetoken.engine.engine import _adjust_config

    args = {"kv_quant": "fp8", **kwargs}
    with pytest.raises(ValueError, match=match):
        _adjust_config(_config(**args))


def test_qwen3_pool_cost_includes_both_scale_slabs():
    from freetoken.kvcache.mha_pool import MHAKVCache

    plain = _config()
    quant = _config(kv_quant="fp8")
    assert MHAKVCache.kv_cost(plain)[0] == 98304
    assert MHAKVCache.kv_cost(quant)[0] == 50688
