import pytest
import torch

from freetoken.engine.config import EngineConfig, KVStorageType
from freetoken.kvcache.base import KVStorageDescriptor, spec_kv_bytes_per_token


def test_q8_descriptor_matches_b10434_row_contract():
    desc = KVStorageDescriptor(KVStorageType.Q8_0)
    assert desc.bytes_per_block == 34
    assert desc.row_bytes(2048) == 2048 + 2 * 64
    assert desc.bytes_per_token(num_layers=1, num_kv_heads=4, head_dim=2048) == 2 * 4 * 2176
    with pytest.raises(ValueError, match="divisible"):
        desc.row_bytes(33)


def test_q8_spec_price_counts_local_heads_and_both_slabs():
    class TP:
        size = 2

    class Spec:
        name = "full"
        mla = False
        index_head_dim = 0
        num_index_layers = 0
        index_ratio = 1
        head_dim = 64
        num_kv_heads = 8
        num_layers = 3

    class Config:
        kv_storage_type = KVStorageType.Q8_0
        tp_info = TP()

    assert spec_kv_bytes_per_token(Spec(), Config()) == 2 * 3 * 4 * (64 + 4)


def test_engine_config_normalizes_kv_storage_type_without_touching_gpu():
    cfg = EngineConfig(model_path="dummy", tp_info=type("TP", (), {"rank": 0, "size": 1})(), dtype=torch.bfloat16, kv_storage_type="q8_0")
    assert cfg.kv_storage_type is KVStorageType.Q8_0
