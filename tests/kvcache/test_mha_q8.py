import pytest
import torch

from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kernel.triton.q8_kv import quantize_row_q8_0_ref, store_q8_cache


def test_q8_store_matches_reference_and_keeps_rows_independent():
    values = torch.tensor(
        [[[0.5, -0.5, 1.0, -1.0] * 8], [[2.0, 0.0, -2.0, 0.0] * 8]], dtype=torch.float32
    )
    payload = torch.empty((4, 1, 32), dtype=torch.int8)
    scales = torch.empty((4, 1, 1), dtype=torch.float16)
    store_q8_cache(
        k_payload=payload,
        v_payload=payload.clone(),
        k_scales=scales,
        v_scales=scales.clone(),
        indices=torch.tensor([2, 0], dtype=torch.int32),
        k=values,
        v=values,
    )
    expected_payload, expected_scales = quantize_row_q8_0_ref(values)
    assert torch.equal(
        payload.index_select(0, torch.tensor([2, 0])).reshape(2, 32), expected_payload
    )
    assert torch.equal(
        scales.index_select(0, torch.tensor([2, 0])), expected_scales.reshape(2, 1, 1)
    )


def test_q8_duplicate_destination_rejected_before_store():
    with pytest.raises(ValueError, match="duplicate"):
        store_q8_cache(
            k_payload=torch.empty((2, 1, 32), dtype=torch.int8),
            v_payload=torch.empty((2, 1, 32), dtype=torch.int8),
            k_scales=torch.empty((2, 1, 1), dtype=torch.float16),
            v_scales=torch.empty((2, 1, 1), dtype=torch.float16),
            indices=torch.tensor([0, 0], dtype=torch.int32),
            k=torch.ones((2, 1, 32)),
            v=torch.ones((2, 1, 32)),
        )


def test_q8_mha_pool_reports_packed_unit_bytes_and_generation():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=3,
        head_dim=32,
        num_pages=2,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        storage_type="q8_0",
    )
    assert pool.is_quantized
    assert pool.unit_bytes() == (2 * 3 * 2 * 34, 0)
    before = pool.pointer_generation
    pool.rebuild(3)
    assert pool.pointer_generation == before + 1
    assert pool.k_cache_view(0).payload.dtype is torch.int8
    assert pool.k_cache_view(0).scales.dtype is torch.float16


def test_q8_zero_rows_initialize_payload_and_scales():
    payload = torch.full((1, 1, 32), 9, dtype=torch.int8)
    scales = torch.full((1, 1, 1), 3, dtype=torch.float16)
    store_q8_cache(
        k_payload=payload, v_payload=payload.clone(),
        k_scales=scales, v_scales=scales.clone(),
        indices=torch.tensor([0], dtype=torch.int32),
        k=torch.zeros((1, 1, 32)), v=torch.zeros((1, 1, 32)),
    )
    assert torch.count_nonzero(payload) == 0
    assert torch.count_nonzero(scales) == 0
