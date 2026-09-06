from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend", ["cpu", "hybrid", "offload"])
def test_rocm_unsafe_cpu_executor_disables_graph(monkeypatch, backend):
    """Same gate covers cpu/hybrid and offload + moe_cpu_layers."""
    import freetoken.engine.engine as engine

    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    config = SimpleNamespace(
        moe_backend=backend,
        moe_cpu_layers="0" if backend == "offload" else None,
        cuda_graph_bs=None,
        cuda_graph_max_bs=32,
    )
    executor = SimpleNamespace(graph_capture_safe=False)

    assert engine._disable_unsafe_rocm_cpu_moe_graph(config, executor)
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


def test_rocm_verified_cpu_executor_keeps_graph(monkeypatch):
    import freetoken.engine.engine as engine

    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    config = SimpleNamespace(cuda_graph_bs=[1, 2, 4], cuda_graph_max_bs=4)

    assert not engine._disable_unsafe_rocm_cpu_moe_graph(
        config, SimpleNamespace(graph_capture_safe=True)
    )
    assert config.cuda_graph_bs == [1, 2, 4]
    assert config.cuda_graph_max_bs == 4


def test_non_rocm_keeps_existing_cuda_host_callback_fallback(monkeypatch):
    import freetoken.engine.engine as engine

    monkeypatch.setattr(engine, "is_rocm", lambda: False)
    config = SimpleNamespace(cuda_graph_bs=None, cuda_graph_max_bs=16)

    assert not engine._disable_unsafe_rocm_cpu_moe_graph(
        config, SimpleNamespace(graph_capture_safe=False)
    )
    assert config.cuda_graph_bs is None
    assert config.cuda_graph_max_bs == 16


@pytest.mark.parametrize(
    ("cuda_graph_bs", "cuda_graph_max_bs", "expected"),
    [
        (None, 32, False),
        ([1, 2], 2, False),
        ([1], 0, False),
        ([], 32, True),
        (None, 0, True),
        (None, -1, True),
    ],
)
def test_rocm_auto_hybrid_requires_graph_to_be_disabled(
    monkeypatch, cuda_graph_bs, cuda_graph_max_bs, expected
):
    import freetoken.engine.engine as engine

    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    config = SimpleNamespace(
        cuda_graph_bs=cuda_graph_bs, cuda_graph_max_bs=cuda_graph_max_bs
    )

    assert engine._auto_hybrid_allowed_before_rocm_probe(config) is expected


def test_cpu_moe_flag_slots_cover_every_graph_batch_size():
    import freetoken.engine.engine as engine

    config = SimpleNamespace(cuda_graph_bs=None, cuda_graph_max_bs=128)

    # [1, 2, 4] plus 8..128 in steps of 8: 19 distinct captured sizes.
    assert engine._cpu_moe_flag_slots_per_layer(config, free_memory=0) == 19


def test_cpu_moe_flag_slots_keep_eager_headroom_for_small_graph_sets():
    import freetoken.engine.engine as engine

    config = SimpleNamespace(cuda_graph_bs=[1, 2, 2, 4], cuda_graph_max_bs=128)

    assert engine._cpu_moe_flag_slots_per_layer(config, free_memory=0) == 16


def test_cuda_flag_memops_keep_module_level_api(monkeypatch):
    """CUDA must not be rerouted through ROCm-only executor methods."""
    import freetoken.moe.cpu_executor as cpu_executor

    calls = []

    class TensorStub:
        shape = (1, 8)

        def copy_(self, *args, **kwargs):
            return self

        def to(self, *args, **kwargs):
            return self

    executor = object.__new__(cpu_executor.CpuMoeExecutor)
    executor._flag_sync = True
    executor._shared_flag_signal = False
    executor._gpu_prequant = False
    executor._flag_slots = {(0, 1): 3}
    executor._cpu_moe = SimpleNamespace(
        memop_submit=lambda *args: calls.append(("submit", args)),
        memop_sync=lambda *args: calls.append(("sync", args)),
    )
    executor._ext = SimpleNamespace(
        submit_flag_memop=lambda *args: pytest.fail("ROCm submit used on CUDA"),
        sync_flag_memop=lambda *args: pytest.fail("ROCm sync used on CUDA"),
    )
    executor._ready = SimpleNamespace(data_ptr=lambda: 101)
    executor._done = SimpleNamespace(data_ptr=lambda: 202)
    io = {name: TensorStub() for name in ("x", "ids", "w", "y")}
    executor._io_for = lambda bs: io
    executor._task_for = lambda layer_id, bs: 303
    executor._io = {1: io}

    monkeypatch.setattr(cpu_executor, "_IS_ROCM", False)
    monkeypatch.setattr(
        cpu_executor.torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=404),
    )
    monkeypatch.setattr(cpu_executor.torch, "empty_like", lambda value: TensorStub())

    tensor = TensorStub()
    pending = executor.decode_submit(0, tensor, tensor, tensor)
    executor.decode_sync(pending)

    assert calls == [
        ("submit", (404, 202, 101, 3)),
        ("sync", (404, 202, 3)),
    ]


def test_direct_rocm_capture_without_flag_sync_fails_closed(monkeypatch):
    import freetoken.moe.cpu_executor as cpu_executor

    monkeypatch.setattr(cpu_executor, "_IS_ROCM", True)
    monkeypatch.setattr(
        cpu_executor.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    executor = object.__new__(cpu_executor.CpuMoeExecutor)
    executor._flag_sync = False

    with pytest.raises(RuntimeError, match="not CUDA-graph safe on ROCm"):
        executor.decode_submit(0, None, None, None)
