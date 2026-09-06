from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TVM-FFI JIT kernels need a GPU"
)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("masked", [False, True])
def test_indexing_jit_matches_torch(index_dtype, masked):
    from freetoken.kernel.index import indexing

    torch.manual_seed(21)
    weights = torch.randn(7, 64, device="cuda", dtype=torch.float32)
    indices = torch.tensor([6, 0, 4, 2], dtype=index_dtype, device="cuda")
    output = torch.full((indices.numel(), weights.shape[1]), -1.0, device="cuda")
    vocab_range = (1, 4) if masked else None

    actual = indexing(weights, indices, output=output, vocab_range=vocab_range)
    expected = weights[indices.to(torch.long) - (vocab_range[0] if masked else 0)]
    if masked:
        valid = (indices >= vocab_range[0]) & (indices < vocab_range[0] + vocab_range[1])
        expected = torch.where(valid[:, None], expected, torch.zeros_like(expected))

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)


def test_indexing_jit_empty_output():
    from freetoken.kernel.index import indexing

    weights = torch.arange(8 * 64, device="cuda", dtype=torch.float32).reshape(8, 64)
    indices = torch.empty(0, dtype=torch.int32, device="cuda")
    output = torch.empty((0, 64), device="cuda", dtype=torch.float32)

    actual = indexing(weights, indices, output=output)

    assert actual.shape == (0, 64)
    assert actual.data_ptr() == output.data_ptr()


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_cache_jit_matches_torch(index_dtype):
    from freetoken.kernel.store import store_cache

    torch.manual_seed(22)
    k_cache = torch.zeros(8, 64, device="cuda", dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    indices = torch.tensor([7, 1, 5, 3], dtype=index_dtype, device="cuda")
    k = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)

    store_cache(k_cache, v_cache, indices, k, v)
    torch.cuda.synchronize()

    expected_k = torch.zeros_like(k_cache)
    expected_v = torch.zeros_like(v_cache)
    expected_k[indices.to(torch.long)] = k
    expected_v[indices.to(torch.long)] = v
    torch.testing.assert_close(k_cache, expected_k)
    torch.testing.assert_close(v_cache, expected_v)
