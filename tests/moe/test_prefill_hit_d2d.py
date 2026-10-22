"""Prefill hit-D2D split: resident experts must be gathered device-side, misses
H2D'd via cudaMemcpyBatchAsync, and the buffer must end up byte-identical to the
full-layer copy for every mix -- including experts resident in the volatile
buffer slots (< 2 * num_experts), which must be re-fetched over PCIe."""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
JIT = pytest.mark.skipif(
    os.getenv("FREETOKEN_DISABLE_JIT", "").strip().lower() in {"1", "true", "yes", "on"},
    reason="batch_memcpy has no AOT prebuild; needs runtime JIT",
)


def _cuda_at_least(major: int, minor: int) -> bool:
    cuda = torch.version.cuda
    if cuda is None:
        return False
    return tuple(int(x) for x in cuda.split(".")[:2]) >= (major, minor)

BATCH_API = pytest.mark.skipif(
    torch.version.hip is None and not _cuda_at_least(13, 0),
    reason="the cudaMemcpyBatchAsync binding needs CUDA >= 13.0",
)

NUM_LAYERS, E, CACHE_SIZE = 3, 8, 24  # hit region = slots [16, 24)


def _make_cache() -> tuple[OffloadMoeCache, dict[str, list[torch.Tensor]]]:
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    return cache, sources


def _seed_resident(cache, sources, layer_id: int, expert_id: int, slot: int) -> None:
    cache.slot_for_id[layer_id, expert_id] = slot
    cache.id_of_slot[slot] = layer_id * E + expert_id
    for name, per_layer in sources.items():
        cache.bank_caches[name][slot].copy_(per_layer[layer_id][expert_id])


@CUDA
@JIT
@BATCH_API
def test_batch_memcpy_roundtrip():
    from freetoken.kernel.batch_memcpy import batch_memcpy_jit

    rows, feat = 16, 1024
    src = torch.randint(0, 256, (rows, feat), dtype=torch.uint8).pin_memory()
    dst = torch.zeros(rows, feat, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(rows)
    dst_ptrs = torch.tensor([dst[i].data_ptr() for i in range(rows)], dtype=torch.int64)
    src_ptrs = torch.tensor([src[p].data_ptr() for p in perm.tolist()], dtype=torch.int64)
    sizes = torch.full((rows,), feat, dtype=torch.int64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        batch_memcpy_jit(dst_ptrs, src_ptrs, sizes, stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(dst.cpu(), src[perm])


@CUDA
@JIT
@BATCH_API
def test_batch_memcpy_probe_waits_for_current_stream():
    from freetoken.kernel.batch_memcpy import load_batch_memcpy

    # Compile and initialize the binding before queuing work so JIT cannot drain it.
    load_batch_memcpy()
    # Warm probe-sized allocations and the stream pool to avoid first-use syncs.
    for _ in range(4):
        block = torch.zeros(16, dtype=torch.uint8, device="cuda")
        del block
        pinned = torch.arange(16, dtype=torch.uint8).pin_memory()
        del pinned
    streams = [torch.cuda.Stream() for _ in range(40)]
    torch.cuda.synchronize()
    del streams

    try:
        torch.cuda._sleep(1_500_000_000)
        assert not torch.cuda.current_stream().query(), "backlog drained before the probe ran"
        load_batch_memcpy()
    finally:
        torch.cuda.synchronize()


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_matches_sources():
    cache, sources = _make_cache()
    # layer 1 residents: two real hits in the hit region, one expert stuck in a
    # buffer slot (< 2E) whose cache bytes are poisoned -- it must come back via
    # H2D, so poison must never surface in the buffer.
    _seed_resident(cache, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache, sources, layer_id=1, expert_id=3, slot=20)
    _seed_resident(cache, sources, layer_id=1, expert_id=0, slot=2)
    for name in sources:
        cache.bank_caches[name][2].fill_(float("nan"))

    cache.begin_prefill()
    assert cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    for layer_id in (0, 1):
        views = cache.wait_prefill_layer(layer_id)
        torch.cuda.synchronize()
        for view, (name, per_layer) in zip(views, sources.items()):
            assert torch.equal(view.cpu(), per_layer[layer_id]), (layer_id, name)
        cache.release_prefill_layer(layer_id)
    # prefetch(0) had no hits; prefetch(1) served experts 1 and 3 from the cache.
    assert cache.prefill_hit_rows == 2
    assert cache.prefill_total_rows == 2 * E
    # the buffer-slot resident was invalidated (its slot belongs to buffer 0)...
    assert int(cache.slot_for_id[1, 0].item()) == -1
    # ...while true hits keep their cache residency.
    assert int(cache.slot_for_id[1, 1].item()) == 17
    assert int(cache.slot_for_id[1, 3].item()) == 20


@CUDA
@JIT
@BATCH_API
@pytest.mark.parametrize("nhit", [0, E])
def test_prefill_hit_d2d_pure_extremes(nhit):
    cache, sources = _make_cache()
    for e in range(nhit):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[2]), name
    assert cache.prefill_hit_rows == nhit


@CUDA
def test_prefill_hit_d2d_noop_without_spare_slots():
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=2 * E,  # the cpu backend's pinned geometry: no hit region
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    cache.begin_prefill()
    assert not cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    views = cache.wait_prefill_layer(0)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[0]), name
