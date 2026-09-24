"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
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


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("frac", [0.1, 0.415, 1.0])
@pytest.mark.parametrize("by_recency", [True, False])
def test_hybrid_fraction_gpu_matches_cpu_reference(frac, by_recency, monkeypatch):
    import freetoken.moe.offload_kernels as kernels
    monkeypatch.setattr(kernels, "_HYBRID_FETCH_BY_RECENCY", by_recency)
    torch.manual_seed(0)
    num_experts, cache_size, top_k = 32, 40, 8

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item())
        assert _balanced_fetch(missing, frac_q16) <= fetched <= missing
        if step == 0 or not by_recency:
            assert fetched == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cap", [0, 1, 32])
def test_hybrid_fixed_cap_unchanged(cap):
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=cap,
    )
    for _ in range(3):
        ids = torch.arange(8, dtype=torch.int32).cuda()
        cache.ensure_experts_hybrid(0, ids)
        assert int(cache.num_indices.item()) == min(cap, int(cache.num_missing_full.item()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_auto_split_admits_reuse_within_resident_lru_window():
    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=8, cache_size=8, device=torch.device("cuda"),
            quant_format="bf16", decode_target="hybrid",
            hybrid_fetch_fraction=0.1,
        )
    gpu, ref = make(), make()
    def step(layer, experts, fetched):
        ids = torch.tensor([experts], device="cuda", dtype=torch.int32)
        expected = ids.cpu()
        gpu.ensure_experts_hybrid(layer, ids)
        ref.ensure_experts_hybrid(layer, expected)
        assert int(gpu.num_indices.item()) == int(ref.num_indices.item()) == fetched
        assert torch.equal(ids.cpu(), expected)
        assert torch.equal(gpu.slot_for_id, ref.slot_for_id)
    step(0, [0, 1], 0)
    step(0, [0, 1], 2)
    step(0, [0, 1], 0)  # Hits, even though the bandwidth split prefers CPU for two misses.
    for experts in ([0, 1], [2, 3], [4, 5], [6, 7]):
        step(1, experts, 0)
        step(1, experts, 2)
    assert (gpu.slot_for_id[0, :2] == -1).all()
    step(0, [0, 1], 0)  # Old reuse falls outside the current residency window.
    step(0, [0, 1], 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reuse_admission_compares_each_victim_during_graph_replay():
    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=8, cache_size=8, device=torch.device("cuda"),
            quant_format="bf16", decode_target="hybrid",
            hybrid_fetch_fraction=0.1,
        )
    gpu, ref = make(), make()
    raw = torch.tensor([[6, 7]], device="cuda", dtype=torch.int32)
    gpu.ensure_experts_hybrid(1, raw.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots = raw.clone()
        gpu.ensure_experts_hybrid(1, slots)
    for cache in (gpu, ref):
        cache.reset()
        cache.id_of_slot.copy_(torch.arange(8, device="cuda"))
        cache.slot_for_id[0].copy_(torch.arange(8, device="cuda"))
        cache.usage.fill_(8)
        cache.usage[0] = 2
        cache.step.fill_(10)
        cache.expert_recency[1, 6:8] = torch.tensor([7, 6], device="cuda")
    expected = raw.cpu()
    ref.ensure_experts_hybrid(1, expected)
    graph.replay()
    assert gpu.num_indices.item() == ref.num_indices.item() == 1
    assert slots[0, 0].item() == 0 and slots[0, 1].item() == -1
    assert torch.equal(slots.cpu(), expected)
    assert torch.equal(gpu.slot_for_id, ref.slot_for_id)
