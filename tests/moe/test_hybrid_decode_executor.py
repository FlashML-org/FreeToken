"""Hybrid routing/order contracts and real CPU/GPU decode parity."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.hybrid_decode import HybridDecodeExecutor, HybridDecodeRequest
from freetoken.moe.offload_cache import OffloadMoeCache


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize("placement", ["cpu", "gpu", "mixed"])
def test_routes_match_unsplit_weighted_experts(overlap, collect_stats, placement):
    # Expert and slot IDs deliberately differ; include repeated IDs and zero weights.
    raw_ids = torch.tensor([[3, 1, 3], [2, 0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.0, 0.3], [0.0, 0.4, 0.6]])
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    experts = torch.tensor([2.0, 5.0, 11.0, 17.0])
    slot_for_expert = torch.tensor([2, 3, 0, 1], dtype=torch.int32)
    slots = torch.tensor([11.0, 17.0, 2.0, 5.0])
    resident = {
        "cpu": torch.zeros_like(raw_ids, dtype=torch.bool),
        "gpu": torch.ones_like(raw_ids, dtype=torch.bool),
        "mixed": torch.tensor([[True, False, True], [True, False, False]]),
    }[placement]
    rewritten = torch.where(resident, slot_for_expert[raw_ids.long()], -1)
    events, captured = [], {}

    def ensure(layer_id, ids):
        assert layer_id == 2
        torch.testing.assert_close(ids, raw_ids)
        ids.copy_(rewritten)
        events.append("ensure")

    def partial(x, w, ids, bank):
        values = bank[ids.clamp_min(0).long()] * (ids >= 0)
        return x * (w * values).sum(dim=-1, keepdim=True)

    def submit(layer_id, x, w, ids):
        assert layer_id == 2 and x is hidden and w is weights
        events.append("submit")
        captured["cpu_ids"] = ids.clone()
        return partial(x, w, ids, experts)

    def sync(pending):
        events.append("sync")
        return pending

    def gpu(x, w, ids):
        events.append("gpu")
        captured["gpu_ids"], captured["gpu_weights"] = ids.clone(), w.clone()
        return partial(x, w, ids, slots)

    cache = SimpleNamespace(
        collect_stats=collect_stats, ensure_experts_hybrid=ensure,
        record_decode_stats_hybrid=lambda layer_id: events.append(("stats", layer_id)),
        copy_missing=lambda: events.append("copy"),
    )
    cpu = SimpleNamespace(decode_submit=submit, decode_sync=sync)
    executor = HybridDecodeExecutor(cache, cpu, overlap=overlap)
    ids = raw_ids.clone()
    result = executor.decode(HybridDecodeRequest(2, hidden, weights, ids), gpu)

    reference = torch.zeros_like(hidden)
    for token in range(2):
        for route in range(3):
            reference[token] += weights[token, route] * experts[raw_ids[token, route]] * hidden[token]
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(ids, rewritten)
    torch.testing.assert_close(captured["cpu_ids"], torch.where(resident, -1, raw_ids))
    torch.testing.assert_close(captured["gpu_ids"], rewritten)
    torch.testing.assert_close(captured["gpu_weights"], torch.where(resident, weights, 0.0))
    expected = ["ensure"] + ([("stats", 2)] if collect_stats else []) + ["submit"]
    expected += ["copy", "gpu", "sync"] if overlap else ["sync", "copy", "gpu"]
    assert events == expected


@pytest.mark.parametrize("supports_inactive", [None, False, True])
def test_layer_keeps_kernel_specific_inactive_slots(supports_inactive):
    layer = object.__new__(OffloadMoELayer)
    layer.layer_id = 2
    layer.quant_method = (
        None if supports_inactive is None else
        SimpleNamespace(kernel=SimpleNamespace(supports_inactive_slots=supports_inactive))
    )
    views, alphas = (object(),), (object(), object())
    layer.offload_cache = SimpleNamespace(
        bank_views=lambda: views, alphas_for_slots=lambda layer_id: alphas,
    )
    hidden, weights = torch.randn(2, 4), torch.tensor([[0.7, 0.0], [0.0, 0.6]])
    ids = torch.tensor([[7, -1], [-1, 9]], dtype=torch.int32)

    def gemm(cache, x, w, slots, **kwargs):
        assert cache is layer.offload_cache and x is hidden and w is weights
        assert kwargs == {"views": views, "n": None, "alphas": alphas, "is_prefill": False}
        torch.testing.assert_close(slots, ids if supports_inactive else ids.clamp_min(0))
        return hidden

    layer._expert_gemm = gemm
    assert layer._run_cached_decode_experts(hidden, weights, ids) is hidden
    assert ids.tolist() == [[7, -1], [-1, 9]]


@pytest.mark.parametrize("target", ["cpu", "hybrid"])
def test_reattach_cpu_executor_updates_hybrid_owner(target):
    cache = OffloadMoeCache(1, 4, 4, torch.device("cpu"), decode_target=target)
    for cpu in (object(), object()):
        cache.set_cpu_executor(cpu)
        assert cache.cpu_executor is cpu
        if target == "hybrid":
            assert cache.hybrid_decode_executor.cache.cpu_executor is cpu
            assert cache.hybrid_decode_executor.cpu_executor is cpu
        else:
            assert cache.hybrid_decode_executor is None


def test_hybrid_scheduler_does_not_keep_cache_alive():
    import weakref

    cache = OffloadMoeCache(1, 4, 4, torch.device("cpu"), decode_target="hybrid")
    cache.set_cpu_executor(object())
    cache_ref = weakref.ref(cache)
    scheduler_ref = weakref.ref(cache.hybrid_decode_executor)
    del cache
    assert cache_ref() is None
    assert scheduler_ref() is None


@pytest.mark.parametrize("overlap", [False, True])
def test_overlap_default_is_startup_setting(monkeypatch, overlap):
    import freetoken.moe.hybrid_decode as hybrid

    monkeypatch.setattr(hybrid, "_HYBRID_OVERLAP", overlap)
    executor = HybridDecodeExecutor(None, None)
    assert executor.overlap is overlap
    assert HybridDecodeExecutor(None, None, overlap=not overlap).overlap is not overlap


def test_cpu_only_layer_bypasses_hybrid_schedule():
    layer = object.__new__(OffloadMoELayer)
    layer.layer_id = 2
    hidden, weights, ids = torch.randn(1, 4), torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32)

    def decode(layer_id, x, w, expert_ids):
        assert layer_id == 2 and x is hidden and w is weights and expert_ids is ids
        return hidden

    layer.offload_cache = SimpleNamespace(
        is_cpu_layer=lambda layer_id: True, decode_target="hybrid",
        cpu_executor=SimpleNamespace(decode=decode),
    )
    assert layer._decode_routed(hidden, weights, ids) is hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or HIP GPU")
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("fetch", [0, 1, 8])
@pytest.mark.parametrize("bs", [1, 4])
def test_hybrid_layer_matches_bf16_reference(overlap, fetch, bs):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(491)
    layers, experts, hidden_size, intermediate, top_k = 2, 8, 256, 128, 2
    device = torch.device("cuda")
    cache = OffloadMoeCache(
        layers, experts, experts, device, decode_target="hybrid", hybrid_max_fetch=fetch,
    )
    banks = {}
    for name, shape in (("gate_up", (2 * intermediate, hidden_size)), ("down", (hidden_size, intermediate))):
        bank = alloc_pinned_tensor(layers * experts, *shape, dtype=torch.bfloat16)
        bank.copy_(torch.randn(bank.shape) * 0.05)
        banks[name] = list(bank.split(experts))
    cache.set_bank_sources(banks)
    cpu = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=bs, device=device,
    )
    cache.set_cpu_executor(cpu)
    cache.hybrid_decode_executor.overlap = overlap
    cache.collect_stats = True
    layer = OffloadMoELayer(0, experts, top_k, hidden_size, intermediate, quant_config=NoQuantConfig())
    layer.offload_cache = cache

    for step in range(8):
        layer.layer_id = step % layers
        hidden = torch.randn(bs, hidden_size, device=device, dtype=torch.bfloat16)
        raw_ids = (torch.arange(bs * top_k).reshape(bs, top_k) + step * 3).remainder(experts).int()
        weights = torch.rand(bs, top_k, dtype=torch.float32)
        weights[0, 0] = 0.0
        reference = torch.zeros(bs, hidden_size)
        for token in range(bs):
            for route in range(top_k):
                expert = raw_ids[token, route]
                gate, up = (banks["gate_up"][layer.layer_id][expert].float() @ hidden[token].float().cpu()).chunk(2)
                act = (torch.nn.functional.silu(gate) * up).bfloat16().float()
                reference[token] += weights[token, route] * (banks["down"][layer.layer_id][expert].float() @ act)
        ids = raw_ids.to(device)
        result = layer._decode_routed(hidden, weights.to(device), ids)
        torch.cuda.synchronize()
        error = (result.float().cpu() - reference).abs().max()
        # GPU gate/up stores bf16; the CPU GEMV keeps that intermediate in fp32.
        assert error < 0.03 * reference.abs().max() + 1e-5
        if fetch == 0:
            assert (ids == -1).all()
        if fetch == experts:
            assert (ids >= 0).all()
    assert cache.stat_calls.item() == 8
