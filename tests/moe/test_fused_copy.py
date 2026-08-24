"""The fused multi-bank ``copy_missing`` path must move exactly the same bytes as the
legacy per-bank ``fast_index_copy_jit`` loop, for every miss count (including the
zero-copy case), across banks of differing per-row sizes.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# mxfp4_triton 6-bank schema with mixed 16B-aligned per-row sizes (bytes), >=256 so the
# legacy per-bank kernel's vectorized template is valid. Sizes not covered by a model in
# kernel/aot_models.py must be listed in its TEST_FEATURE_SIZES so the per-bank kernels
# stay prebuilt under FREETOKEN_DISABLE_JIT=1.
FEATS = [8192, 512, 256, 4096, 512, 256]


def _build_cache(num_layers, num_experts, cache_size):
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=dev, cache_policy="lru", prefill_overlap=False, quant_format="mxfp4_triton",
    )
    schema = _BANK_SCHEMAS["mxfp4_triton"]
    # Views into one flat tensor: only per-layer addressing matters here, not
    # independent allocations.
    sources = {
        name: list(torch.randint(0, 256, (num_layers * num_experts, feat), dtype=torch.uint8, device=dev)
                   .split(num_experts))
        for name, feat in zip(schema, FEATS)
    }
    cache.set_bank_sources(sources)  # also builds the fused-copy descriptor
    return cache


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("num_indices", [0, 1, 4, 8])
def test_fused_copy_matches_per_bank(num_indices):
    num_layers, num_experts, cache_size = 8, 8, 32
    layer_id = 3  # exercise a non-zero per-layer source selection, not just layer 0
    cache = _build_cache(num_layers, num_experts, cache_size)
    assert cache._copy_fused_ok, "fused copy should activate for 16B-aligned banks"
    # copy_missing resolves the per-layer source through this (normally set by
    # ensure_experts/materialize_layer); poked directly here since this test drives
    # evict_slots/src_indices/num_indices by hand.
    cache._pending_src_layer = layer_id

    cache.num_indices.fill_(num_indices)
    if num_indices:
        dev = torch.device("cuda")
        cache.evict_slots[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % cache_size
        # src_indices are layer-local expert rows (0..num_experts) under the new contract.
        cache.src_indices[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % num_experts

    # reference: legacy per-bank loop
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = False
    cache.copy_missing()
    torch.cuda.synchronize()
    ref = [c.clone() for _, c in cache.banks]

    # fused multi-bank launch
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = True
    cache.copy_missing()
    torch.cuda.synchronize()

    for b, (r, (_, c)) in enumerate(zip(ref, cache.banks)):
        assert torch.equal(r, c), f"bank {b} (feat={FEATS[b]}) fused != per-bank at num_indices={num_indices}"


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("layer_id", [0, 1])
def test_fused_copy_variable_source_rows_fill_cache_prefix(layer_id):
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=8,
        device=dev,
        cache_policy="lru",
        prefill_overlap=False,
        quant_format="gguf",
    )
    sources = {
        "gate_up": [
            torch.randint(0, 256, (4, 256), dtype=torch.uint8, device=dev),
            torch.randint(0, 256, (4, 512), dtype=torch.uint8, device=dev),
        ],
        "down": [
            torch.randint(0, 256, (4, 512), dtype=torch.uint8, device=dev),
            torch.randint(0, 256, (4, 256), dtype=torch.uint8, device=dev),
        ],
    }
    cache.set_bank_sources(sources)
    assert cache._copy_fused_ok
    assert cache._variable_bank_rows == {"gate_up", "down"}

    cache._pending_src_layer = layer_id
    cache.num_indices.fill_(2)
    cache.evict_slots[:2] = torch.tensor([1, 5], dtype=torch.int32, device=dev)
    cache.src_indices[:2] = torch.tensor([3, 0], dtype=torch.int32, device=dev)
    for _, bank in cache.banks:
        bank.fill_(0xA5)
    cache.copy_missing()
    torch.cuda.synchronize()

    for name, (_, bank) in zip(cache.bank_schema, cache.banks):
        source = sources[name][layer_id]
        feat = source.shape[1]
        torch.testing.assert_close(bank[1, :feat], source[3])
        torch.testing.assert_close(bank[5, :feat], source[0])
        if feat < bank.shape[1]:
            assert torch.all(bank[1, feat:] == 0xA5)
            assert torch.all(bank[5, feat:] == 0xA5)
