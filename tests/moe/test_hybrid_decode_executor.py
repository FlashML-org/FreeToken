"""Platform-independent contract tests for Hybrid decode orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.moe.hybrid_decode import HybridDecodeExecutor, HybridDecodeRequest


class FakeCache:
    def __init__(
        self,
        rewritten_ids: torch.Tensor,
        events: list[str],
        *,
        stats: bool = True,
    ):
        self.rewritten_ids = rewritten_ids
        self.events = events
        self.collect_stats = stats
        self.stats_calls = 0

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        self.events.append(f"ensure:{layer_id}")
        expert_ids.copy_(self.rewritten_ids)

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        self.events.append(f"stats:{layer_id}")
        self.stats_calls += 1

    def copy_missing(self) -> None:
        self.events.append("copy")


class FakeCpuExecutor:
    def __init__(self, events: list[str], result: torch.Tensor):
        self.events = events
        self.result = result
        self.submissions = []
        self.pending = object()

    def decode_submit(self, layer_id, hidden_states, topk_weights, topk_ids):
        self.events.append(f"submit:{layer_id}")
        self.submissions.append(
            SimpleNamespace(
                layer_id=layer_id,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids.clone(),
            )
        )
        return self.pending

    def decode_sync(self, pending):
        assert pending is self.pending
        self.events.append("wait")
        return self.result


class RecordingGpuRunner:
    def __init__(self, events: list[str], result: torch.Tensor):
        self.events = events
        self.result = result
        self.calls = []

    def __call__(self, hidden_states, topk_weights, slot_ids):
        self.events.append("gpu")
        self.calls.append(
            SimpleNamespace(
                hidden_states=hidden_states,
                topk_weights=topk_weights.clone(),
                slot_ids=slot_ids.clone(),
            )
        )
        return self.result


def _run(*, overlap: bool, rewritten_ids=None, collect_stats: bool = True):
    events = []
    raw_ids = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    topk_ids = raw_ids.clone()
    topk_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    rewritten = (
        torch.tensor([[7, -1], [-1, 9]], dtype=torch.int32)
        if rewritten_ids is None
        else rewritten_ids
    )
    cpu_result = torch.full_like(hidden_states, 2.0)
    gpu_result = torch.full_like(hidden_states, 5.0)
    cache = FakeCache(rewritten, events, stats=collect_stats)
    cpu = FakeCpuExecutor(events, cpu_result)
    gpu = RecordingGpuRunner(events, gpu_result)
    executor = HybridDecodeExecutor(cache, cpu, overlap=overlap)

    output = executor.decode(
        HybridDecodeRequest(
            layer_id=2,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        ),
        gpu,
    )
    return SimpleNamespace(
        events=events,
        raw_ids=raw_ids,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        output=output,
        cpu_result=cpu_result,
        gpu_result=gpu_result,
        cache=cache,
        cpu=cpu,
        gpu=gpu,
    )


def test_overlap_submits_cpu_before_copy_and_waits_after_gpu():
    result = _run(overlap=True)

    assert result.events == ["ensure:2", "stats:2", "submit:2", "copy", "gpu", "wait"]
    torch.testing.assert_close(result.output, result.cpu_result + result.gpu_result)


def test_serial_mode_waits_before_copy_and_gpu():
    result = _run(overlap=False)

    assert result.events == ["ensure:2", "stats:2", "submit:2", "wait", "copy", "gpu"]
    torch.testing.assert_close(result.output, result.cpu_result + result.gpu_result)


def test_routes_are_partitioned_once_and_ids_keep_in_place_rewrite():
    result = _run(overlap=True)

    assert result.topk_ids.tolist() == [[7, -1], [-1, 9]]
    submission = result.cpu.submissions[0]
    assert submission.layer_id == 2
    assert submission.hidden_states is result.hidden_states
    assert submission.topk_weights is result.topk_weights
    assert submission.topk_ids.tolist() == [[-1, 4], [5, -1]]

    gpu_call = result.gpu.calls[0]
    assert gpu_call.hidden_states is result.hidden_states
    assert gpu_call.slot_ids.tolist() == [[7, 0], [0, 9]]
    torch.testing.assert_close(
        gpu_call.topk_weights,
        torch.tensor([[0.7, 0.0], [0.0, 0.6]], dtype=torch.float32),
    )

    cpu_mask = submission.topk_ids >= 0
    gpu_mask = gpu_call.topk_weights != 0
    assert torch.equal(cpu_mask, ~gpu_mask)


def test_all_gpu_still_uses_fixed_cpu_submit_wait_structure():
    result = _run(
        overlap=True,
        rewritten_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )

    assert result.cpu.submissions[0].topk_ids.tolist() == [[-1, -1], [-1, -1]]
    assert result.gpu.calls[0].slot_ids.tolist() == [[1, 2], [3, 4]]
    torch.testing.assert_close(result.gpu.calls[0].topk_weights, result.topk_weights)
    assert result.events[-1] == "wait"


def test_stats_follow_cache_flag_once_per_decode():
    enabled = _run(overlap=True, collect_stats=True)
    disabled = _run(overlap=True, collect_stats=False)

    assert enabled.cache.stats_calls == 1
    assert disabled.cache.stats_calls == 0
    assert "stats:2" not in disabled.events


def test_offload_layer_delegates_hybrid_decode_without_owning_the_schedule():
    from freetoken.layers.moe import OffloadMoELayer

    hidden_states = torch.randn(2, 4)
    topk_weights = torch.rand(2, 2)
    topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    expected = torch.randn_like(hidden_states)
    calls = []

    class RecordingHybridExecutor:
        def decode(self, request, gpu_expert_runner):
            calls.append((request, gpu_expert_runner))
            return expected

    cache = SimpleNamespace(
        is_cpu_layer=lambda _layer_id: False,
        decode_target="hybrid",
        hybrid_decode_executor=RecordingHybridExecutor(),
    )
    layer = object.__new__(OffloadMoELayer)
    layer.layer_id = 7
    layer.offload_cache = cache

    output = layer._decode_routed(hidden_states, topk_weights, topk_ids)

    assert output is expected
    assert len(calls) == 1
    request, runner = calls[0]
    assert request.layer_id == 7
    assert request.hidden_states is hidden_states
    assert request.topk_weights is topk_weights
    assert request.topk_ids is topk_ids
    assert runner.__self__ is layer
    assert runner.__func__ is OffloadMoELayer._run_cached_decode_experts


def test_cache_wires_hybrid_executor_when_cpu_executor_is_attached():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = object.__new__(OffloadMoeCache)
    cache.decode_target = "hybrid"
    cpu_executor = object()

    cache.set_cpu_executor(cpu_executor)

    assert cache.cpu_executor is cpu_executor
    assert isinstance(cache.hybrid_decode_executor, HybridDecodeExecutor)
    assert cache.hybrid_decode_executor.cache is cache
    assert cache.hybrid_decode_executor.cpu_executor is cpu_executor
