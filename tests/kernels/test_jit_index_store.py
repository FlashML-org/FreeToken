import pytest
import torch

from freetoken.kernel import indexing, store_cache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)


def test_indexing_jit_matches_torch_on_cold_and_warm_loads():
    weights = torch.arange(8 * 64, dtype=torch.float32, device="cuda").reshape(8, 64)

    for values in ([7, 2, 0], [1, 6, 3]):
        indices = torch.tensor(values, dtype=torch.int32, device="cuda")
        actual = indexing(weights, indices)
        torch.testing.assert_close(actual, weights[indices.long()])


def test_store_jit_matches_torch_on_cold_and_warm_loads():
    k_cache = torch.zeros((8, 64), dtype=torch.float32, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    indices = torch.tensor([5, 0, 3], dtype=torch.int64, device="cuda")

    for offset in (0.0, 1000.0):
        k = torch.arange(3 * 64, dtype=torch.float32, device="cuda").reshape(3, 64)
        k = k + offset
        v = k + 500.0
        store_cache(k_cache, v_cache, indices, k, v)
        torch.testing.assert_close(k_cache[indices], k)
        torch.testing.assert_close(v_cache[indices], v)

    untouched = torch.tensor([1, 2, 4, 6, 7], device="cuda")
    torch.testing.assert_close(k_cache[untouched], torch.zeros((5, 64), device="cuda"))
    torch.testing.assert_close(v_cache[untouched], torch.zeros((5, 64), device="cuda"))
