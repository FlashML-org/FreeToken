"""The fused multi-bank ``copy_missing`` path must move exactly the same bytes as the
legacy per-bank ``fast_index_copy_jit`` loop, for every miss count (including the
zero-copy case), across banks of differing per-row sizes.
"""

from __future__ import annotations

import pytest
import torch

import freetoken.moe.offload_cache as offload_cache
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
def test_fused_copy_matches_per_bank(num_indices, monkeypatch):
    # Exercise the fused kernel explicitly even on Windows/ROCm, where serving now
    # defaults to the stable per-bank path.
    monkeypatch.setattr(offload_cache, "_FUSED_COPY", True)
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


def test_fused_copy_defaults_off_on_windows_rocm(monkeypatch):
    monkeypatch.delenv("FREETOKEN_FUSED_COPY", raising=False)
    monkeypatch.setattr(offload_cache.os, "name", "nt")
    monkeypatch.setattr(offload_cache.torch.version, "hip", "test-rocm")

    assert not offload_cache._fused_copy_enabled()


def test_fused_copy_default_stays_on_for_other_platforms(monkeypatch):
    monkeypatch.delenv("FREETOKEN_FUSED_COPY", raising=False)
    monkeypatch.setattr(offload_cache.os, "name", "posix")
    monkeypatch.setattr(offload_cache.torch.version, "hip", "test-rocm")

    assert offload_cache._fused_copy_enabled()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_fused_copy_explicit_override_enables_windows_rocm(monkeypatch, value):
    monkeypatch.setenv("FREETOKEN_FUSED_COPY", value)
    monkeypatch.setattr(offload_cache.os, "name", "nt")
    monkeypatch.setattr(offload_cache.torch.version, "hip", "test-rocm")

    assert offload_cache._fused_copy_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_fused_copy_explicit_override_disables_other_platforms(monkeypatch, value):
    monkeypatch.setenv("FREETOKEN_FUSED_COPY", value)
    monkeypatch.setattr(offload_cache.os, "name", "posix")
    monkeypatch.setattr(offload_cache.torch.version, "hip", "test-rocm")

    assert not offload_cache._fused_copy_enabled()
