import pytest
import torch

from freetoken.kernel import indexing, store_cache
from freetoken.kernel.index import _jit_index_module, num_splits_for
from freetoken.kernel.store import _jit_store_module
from freetoken.kernel.utils import KernelConfig


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("row_bytes", "expected_splits"), [(256, 1), (1024, 2), (2048, 4)])
def test_indexing_jit_matches_torch(dtype, index_dtype, row_bytes, expected_splits):
    width = row_bytes // dtype.itemsize
    weights = torch.randn((17, width), dtype=dtype, device="cuda")
    assert num_splits_for(row_bytes) == expected_splits
    output = torch.full((7, width), float("nan"), dtype=dtype, device="cuda")

    for values in ([16, 2, 0, 5, 5, 1, 8], [1, 6, 3, 0, 7, 16, 2]):
        indices = torch.tensor(values, dtype=index_dtype, device="cuda")
        actual = indexing(weights, indices, output=output)
        assert actual is output
        torch.testing.assert_close(actual, weights[indices.long()], rtol=0, atol=0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [64, 256, 512])
def test_masked_indexing_zeros_indices_outside_vocab_range(index_dtype, width):
    weights = torch.randn((6, width), dtype=torch.float32, device="cuda")
    indices = torch.tensor([-1, 9, 10, 15, 16, 99, 12], dtype=index_dtype, device="cuda")
    output = torch.full((7, width), float("nan"), device="cuda")

    actual = indexing(weights, indices, output=output, vocab_range=(10, 6))
    local = indices.long() - 10
    valid = (local >= 0) & (local < len(weights))
    expected = torch.zeros_like(output)
    expected[valid] = weights[local[valid]]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_store_jit_matches_torch(dtype, index_dtype, strided):
    width = 256
    cache_width = width * 2 if strided else width
    input_width = width * 3 if strided else width
    k_bank = torch.full((17, cache_width), -1.0, dtype=dtype, device="cuda")
    v_bank = torch.full_like(k_bank, -2.0)
    k_cache, v_cache = k_bank[:, :width], v_bank[:, :width]
    expected_k, expected_v = k_bank.clone(), v_bank.clone()

    for values in ([16, 0, 3, 9, 2, 5, 7], [1, 6, 3, 0, 7, 16, 2]):
        indices = torch.tensor(values, dtype=index_dtype, device="cuda")
        k = torch.randn((7, input_width), dtype=dtype, device="cuda")[:, :width]
        v = torch.randn((7, input_width), dtype=dtype, device="cuda")[:, :width]
        store_cache(k_cache, v_cache, indices, k, v)
        expected_k[indices.long(), :width] = k
        expected_v[indices.long(), :width] = v
        # Compare the backing storage too: padding and untouched rows must survive.
        torch.testing.assert_close(k_bank, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(v_bank, expected_v, rtol=0, atol=0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("masked", [False, True])
def test_index_store_graph_replay_on_nondefault_stream(index_dtype, masked):
    weights = torch.randn((17, 512), dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(7, dtype=index_dtype, device="cuda")
    slots = torch.tensor([16, 0, 3, 9, 2, 5, 7], dtype=index_dtype, device="cuda")
    k_cache = torch.empty_like(weights)
    v_cache = torch.empty_like(weights)
    vocab_range = (10, len(weights)) if masked else None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        # Resolve the JIT modules before capture; the graph must use this stream.
        gathered = indexing(weights, indices, vocab_range=vocab_range)
        store_cache(k_cache, v_cache, slots, gathered, gathered)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            gathered = indexing(weights, indices, vocab_range=vocab_range)
            store_cache(k_cache, v_cache, slots, gathered, gathered)

        for step in range(3):
            weights.normal_()
            values = torch.tensor([16, 0, 3, 9, 2, 5, 7], dtype=index_dtype, device="cuda")
            values = (values + step) % len(weights)
            if masked:
                values += 10
                values[0], values[-1] = 9, 27
            indices.copy_(values)
            slots.copy_(torch.roll(slots, 1))
            k_cache.fill_(-1)
            v_cache.fill_(-2)
            graph.replay()

            local = indices.long() - (10 if masked else 0)
            valid = (local >= 0) & (local < len(weights))
            expected = torch.zeros_like(gathered)
            expected[valid] = weights[local[valid]]
            expected_k = torch.full_like(k_cache, -1)
            expected_v = torch.full_like(v_cache, -2)
            expected_k[slots.long()] = expected
            expected_v[slots.long()] = expected
            torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
            torch.testing.assert_close(k_cache, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)
    stream.synchronize()


@pytest.mark.skipif(torch.version.hip is None, reason="PDL rejection is ROCm-specific")
@pytest.mark.parametrize("kernel", ["index", "store"])
def test_rocm_rejects_explicit_pdl(kernel):
    config = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=True)
    weights = torch.zeros((8, 64), device="cuda")
    indices = torch.tensor([1, 0, 3], dtype=torch.int32, device="cuda")
    output = torch.empty((3, 64), device="cuda")

    if kernel == "index":
        module = _jit_index_module(256, config=config)
        args = (weights, indices, output, None)
    else:
        module = _jit_store_module(256, config=config)
        args = (weights, weights.clone(), indices, output, output)

    with pytest.raises(RuntimeError, match="Programmatic dependent launch is unavailable on ROCm"):
        module.launch(*args)


@pytest.mark.parametrize("kernel", ["index", "store"])
@pytest.mark.parametrize("invalid", ["cpu", "mixed_device", "index_dtype"])
def test_jit_rejects_invalid_tensor_contract(kernel, invalid):
    device = "cpu" if invalid == "cpu" else "cuda"
    weights = torch.zeros((8, 64), device=device)
    indices = torch.zeros(
        1,
        dtype=torch.float32 if invalid == "index_dtype" else torch.int32,
        device="cpu" if invalid == "mixed_device" else device,
    )
    output = torch.empty((1, 64), device=device)
    message = {
        "cpu": "Device value .* not in the allowed options",
        "mixed_device": "Device mismatch",
        "index_dtype": "Dtype value .* not in the allowed options",
    }[invalid]

    with pytest.raises(RuntimeError, match=message):
        if kernel == "index":
            indexing(weights, indices, output=output)
        else:
            store_cache(weights, weights.clone(), indices, output, output)
