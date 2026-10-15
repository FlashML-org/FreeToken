"""Hybrid decode's bandwidth-matched fetch split.

Covers profile reading and the real ensure kernel against an independent bandwidth
oracle, timestamp-LRU policy, and complementary CPU/GPU routing contracts.
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import PROFILE_VERSION, default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction

Q = 1 << 16


def _balanced_fetch(missing: int, fraction: float) -> int:
    """Exhaustively minimize the slower side, rather than mirroring the kernel's rounding."""
    q = min(Q, max(0, round(fraction * Q)))
    return min(range(missing + 1), key=lambda fetched: max(fetched * (Q - q), (missing - fetched) * q))


def test_balanced_fetch_tracks_fraction():
    for fraction in (0.1, 0.415, 0.454, 0.7, 1.0):
        for missing in range(65):
            fetched = _balanced_fetch(missing, fraction)
            assert 0 <= fetched <= missing
            assert abs(fetched - fraction * missing) <= 1
    assert _balanced_fetch(3, 0.415) == 1
    assert _balanced_fetch(4, 0.415) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None
    # so is one written before the current set of CPU formats
    path.write_text(json.dumps({**prof, "version": PROFILE_VERSION - 1}))
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": PROFILE_VERSION, "gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="hybrid ensure requires CUDA and the matching Triton/Flashlib stack")
@pytest.mark.parametrize("max_fetch,fraction", [(0, 0.0), (1, 0.0), (3, 0.0), (8, 0.0), (0, 0.415), (1, 0.7), (2, 1.0)])
@pytest.mark.parametrize("clamp", [False, True])
@pytest.mark.parametrize("by_recency", [False, True])
def test_hybrid_routes_and_cache_match_independent_policy(monkeypatch, max_fetch, fraction, clamp, by_recency):
    from freetoken.moe import offload_kernels
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(offload_kernels, "_HYBRID_FETCH_BY_RECENCY", by_recency)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=12, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=max_fetch,
        hybrid_fetch_fraction=fraction,
    )
    cache.collect_stats = True
    # Seed resident hits through the production metadata path; no expert weights or copies.
    cache.materialize_layer(1)
    generator = torch.Generator().manual_seed(41)
    previous = torch.tensor([[0, 0, 1, 2], [3, 1, 0, 3]], dtype=torch.int32)
    for step in range(24):
        layer = (step // 3) % 2
        raw_ids = previous.clone() if step % 3 else torch.randint(0, 8, (2, 4), generator=generator, dtype=torch.int32)
        raw_ids[0, 1] = raw_ids[0, 0]  # deduplicate fetches, not the routes themselves
        previous = raw_ids.clone()
        weights = torch.rand(2, 4, generator=generator)
        weights[0, 0] = 0  # a zero-weight GPU route still has a GPU slot and no CPU id
        before = cache.slot_for_id.cpu().clone()
        owners = cache.id_of_slot.cpu().clone()
        usage = cache.usage.cpu().clone()
        recency = cache.expert_recency.cpu().clone()
        tick = int(cache.step.item()) + 1
        active = set(raw_ids.flatten().tolist())
        missing = {e for e in active if before[layer, e] < 0}
        count = _balanced_fetch(len(missing), fraction) if fraction > 0 else min(len(missing), max_fetch)
        fetched = sorted(missing, key=lambda e: (-int(recency[layer, e]), e) if by_recency else e)[:count]
        # Declarative LRU policy: active residents cannot be evicted; ties use the lower slot.
        active_global = {layer * 8 + e for e in active}
        victims = sorted((s for s in range(12) if int(owners[s]) not in active_global),
                         key=lambda s: (int(usage[s]), s))[:count]
        expected = before.clone()
        expected_owners = owners.clone()
        expected_usage = usage.clone()
        for e in active - missing:
            expected_usage[int(before[layer, e])] = tick
        for e, slot in zip(fetched, victims):
            old = int(owners[slot])
            if old >= 0:
                expected[old // 8, old % 8] = -1
            expected[layer, e] = slot
            expected_owners[slot] = layer * 8 + e
            expected_usage[slot] = tick

        ids = raw_ids.cuda()
        device_weights = weights.cuda()
        slots, gpu_w, cpu_ids = cache.ensure_experts_hybrid(layer, ids, device_weights, clamp=clamp)
        assert slots.data_ptr() == ids.data_ptr()
        assert int(cache.num_missing_full.item()) == len(missing)
        assert int(cache.num_indices.item()) == count
        assert cache.src_indices[:count].cpu().tolist() == fetched
        assert cache.evict_slots[:count].cpu().tolist() == victims
        assert torch.equal(cache.slot_for_id.cpu(), expected)
        assert torch.equal(cache.id_of_slot.cpu(), expected_owners)
        assert torch.equal(cache.usage.cpu(), expected_usage)
        assert int(cache.step.item()) == tick
        expected_recency = recency.clone()
        if by_recency:
            expected_recency[layer, list(active)] = tick
        assert torch.equal(cache.expert_recency.cpu(), expected_recency)

        expected_slots = expected[layer, raw_ids.long()]
        on_gpu = expected_slots >= 0
        assert torch.equal(slots.cpu(), expected_slots.clamp_min(0) if clamp else expected_slots)
        assert torch.equal(gpu_w.cpu(), torch.where(on_gpu, weights, 0.0))
        assert torch.equal(cpu_ids.cpu(), torch.where(on_gpu, -1, raw_ids))
        assert torch.equal(device_weights.cpu(), weights)
        assert torch.equal(cache.active_mask.cpu().bool(), torch.tensor([e in active for e in range(8)]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="hybrid ensure graph replay requires CUDA and the matching kernel stack")
@pytest.mark.parametrize("clamp", [False, True])
def test_hybrid_route_graph_replay_switches_cpu_work_on_and_off(clamp):
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(num_layers=1, num_experts=8, cache_size=8, device=torch.device("cuda"),
                            quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=0)
    cache.materialize_layer(0)
    # Model a partially resident layer without copying weights or requiring a CPU executor.
    cache.slot_for_id[0, 4:] = -1
    cache.id_of_slot[4:] = -1
    ids = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    weights = torch.ones(1, 4, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache.ensure_experts_hybrid(0, ids, weights, clamp=clamp)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        routes = cache.ensure_experts_hybrid(0, ids, weights, clamp=clamp)
    for raw in ([4, 5, 4, 6], [0, 1, 2, 3], [7, 0, 7, 1], [3, 2, 1, 0]):
        source = torch.tensor([raw], dtype=torch.int32, device="cuda")
        ids.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        on_gpu = source < 4
        assert torch.equal(routes.cpu_ids, torch.where(on_gpu, -1, source))
        assert torch.equal(routes.gpu_w, torch.where(on_gpu, weights, 0.0))
        expected_slots = torch.where(on_gpu, source, 0 if clamp else -1)
        assert torch.equal(routes.gpu_slots, expected_slots)
