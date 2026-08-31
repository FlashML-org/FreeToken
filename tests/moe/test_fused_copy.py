"""The fused multi-bank ``copy_missing`` path must move exactly the same bytes as the
legacy per-bank ``fast_index_copy_jit`` loop, for every miss count (including the
zero-copy case), across banks of differing per-row sizes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# mxfp4_triton 6-bank schema with mixed 16B-aligned per-row sizes (bytes), >=256 so the
# legacy per-bank kernel's vectorized template is valid. Sizes not covered by a model in
# kernel/aot_models.py must be listed in its TEST_FEATURE_SIZES so the per-bank kernels
# stay prebuilt under FREETOKEN_DISABLE_JIT=1.
FEATS = [8192, 512, 256, 4096, 512, 256]


def test_rocm_graph_capture_uses_per_bank_copy(monkeypatch):
    calls = []
    tensor = torch.empty(1)
    cache = SimpleNamespace(
        banks=[([tensor], tensor)],
        _pending_src_layer=0,
        _pending_whole_layer=False,
        _unpinned_layers=frozenset(),
        _copy_fused_ok=True,
        _copy_dst_ptrs=tensor,
        _copy_src_ptrs=[tensor],
        _copy_feat_bytes=tensor,
        evict_slots=tensor,
        src_indices=tensor,
        num_indices=tensor,
    )
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        "freetoken.kernel.fast_index_copy.fast_index_copy_multi_jit",
        lambda *args: calls.append("fused"),
    )
    monkeypatch.setattr(
        "freetoken.kernel.fast_index_copy_jit",
        lambda *args: calls.append("legacy"),
    )

    OffloadMoeCache.copy_missing(cache)

    assert calls == ["legacy"]


@CUDA
@pytest.mark.slow
def test_qwen36_sized_pinned_rows_survive_rocm_graph_replay():
    if not torch.version.hip:
        pytest.skip("ROCm regression")

    layers, experts, cache_size, topk = 40, 256, 2117, 8
    cache = OffloadMoeCache(
        num_layers=layers,
        num_experts=experts,
        cache_size=cache_size,
        device=torch.device("cuda"),
        quant_format="q4_k_q5_k",
    )
    # Qwen3.6-35B-A3B: H=2048, I=512. Keep the real packed bytes per expert.
    features = {"gate_up": 2 * 512 * (2048 // 256) * 144,
                "down": 2048 * (512 // 256) * 176}
    sources = {}
    fingerprints = torch.arange(experts, dtype=torch.uint8)
    for name, feature_bytes in features.items():
        source = torch.empty((experts, feature_bytes), dtype=torch.uint8, pin_memory=True)
        source[:, 0] = fingerprints
        sources[name] = [source] * layers
    cache.set_bank_sources(sources)
    assert cache._copy_fused_ok

    patterns = (
        torch.arange(experts, dtype=torch.int32, device="cuda")[:, None]
        + torch.arange(topk, dtype=torch.int32, device="cuda")[None, :]
    ) % experts
    routes = [patterns[layer * 13 % experts].clone() for layer in range(layers)]
    slots = [torch.empty_like(route) for route in routes]

    def body():
        for layer, (route, layer_slots) in enumerate(zip(routes, slots)):
            layer_slots.copy_(route)
            cache.ensure_experts(layer, layer_slots)
            cache.copy_missing()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
    side.synchronize()
    cache.reset()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        body()
    cache.reset()
    torch.cuda.synchronize()

    for step in range(512):
        for layer, route in enumerate(routes):
            route.copy_(patterns[(step + layer * 13) % experts])
        graph.replay()
    torch.cuda.synchronize()

    for route, layer_slots in zip(routes, slots):
        expected = route.cpu().tolist()
        resident = layer_slots.cpu().tolist()
        for _, bank in cache.banks:
            assert bank[resident, 0].cpu().tolist() == expected


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
