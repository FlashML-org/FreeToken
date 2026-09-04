"""Config-time gating of --kv-cache-dtype against the pool families the quant
path actually reaches (see _validate_kv_cache_dtype).

The validator must own every rejection message for model families whose pools
are constructed without a quant arg, and must never suggest --attention-backend
triton for a model triton cannot serve.
"""

import pytest

from freetoken.attention import AttnType
from freetoken.engine.engine import _validate_kv_cache_dtype
from freetoken.models.config import KVCacheGroupSpec


class _Quant:
    enabled = True
    name = "q4_0"


class _DisabledQuant:
    enabled = False
    name = "q4_0"


def _config(backend="triton", quant=_Quant):
    class Cfg:
        kv_quant = quant()
        attention_backend = backend
    return Cfg()


def _spec(**overrides):
    fields = dict(
        name="g0", layer_ids=(0,), num_kv_heads=8, head_dim=128,
        sliding_window=None, attn_type=AttnType.FULL,
    )
    fields.update(overrides)
    return KVCacheGroupSpec(**fields)


def _model_config(specs):
    class MC:
        def kv_cache_group_specs(self):
            return tuple(specs)
    return MC()


def test_disabled_or_absent_quant_is_noop():
    _validate_kv_cache_dtype(_config(quant=_DisabledQuant), _model_config([_spec()]))
    class NoQuant:
        attention_backend = "triton"
        kv_quant = None
    _validate_kv_cache_dtype(NoQuant, _model_config([_spec()]))


def test_dsv4_rejected_by_attn_type_not_by_mla_fields():
    # DSV4 specs carry mla=False / index_head_dim=0 on purpose (config.py), so the
    # MLA gate cannot see them; the attn_type gate must.
    spec = _spec(name="dsv4", attn_type=AttnType.DSV4, head_dim=512, sliding_window=256)
    with pytest.raises(ValueError, match="dsv4"):
        _validate_kv_cache_dtype(_config(), _model_config([spec]))


def test_bsa_qsa_rejected_by_attn_type():
    bsa = _spec(name="m3", attn_type=AttnType.BSA, index_head_dim=128, num_index_layers=12)
    with pytest.raises(ValueError, match="bsa"):
        _validate_kv_cache_dtype(_config(), _model_config([bsa]))
    qsa = _spec(
        name="flash_next", attn_type=AttnType.QSA, index_head_dim=128,
        num_index_layers=12, index_ratio=4,
    )
    with pytest.raises(ValueError, match="qsa"):
        _validate_kv_cache_dtype(_config(), _model_config([qsa]))


def test_mla_rejected():
    spec = _spec(name="mla", mla=True)
    with pytest.raises(ValueError, match="MLA/DSA"):
        _validate_kv_cache_dtype(_config(), _model_config([spec]))


def test_non_triton_backend_rejected():
    with pytest.raises(ValueError, match="triton"):
        _validate_kv_cache_dtype(
            _config(backend="dsv4_sparse"), _model_config([_spec()])
        )


def test_full_swa_model_passes():
    specs = [_spec(), _spec(name="swa", attn_type=AttnType.SWA, sliding_window=512)]
    _validate_kv_cache_dtype(_config(), _model_config(specs))


def test_head_dim_not_multiple_of_block_rejected():
    with pytest.raises(ValueError, match="head_dim"):
        _validate_kv_cache_dtype(_config(), _model_config([_spec(head_dim=100)]))
