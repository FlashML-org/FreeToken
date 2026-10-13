from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("strategy", ["cpu", "hybrid", "offload"])
def test_rocm_unsafe_cpu_executor_disables_graph(monkeypatch, strategy):
    """The same gate covers cpu/hybrid and offload + moe_cpu_layers."""
    import freetoken.engine.engine as engine

    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    config = SimpleNamespace(
        moe_strategy=strategy,
        moe_cpu_layers="0" if strategy == "offload" else None,
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


@pytest.mark.parametrize("graph_bs, max_bs, tokens, slots", [
    ([1, 129], 16, 129, 16), (None, 128, 128, 19),
])
def test_cpu_executor_capacity_and_checkpoint_block(monkeypatch, graph_bs, max_bs, tokens, slots):
    import torch
    import freetoken.engine.engine as engine
    import freetoken.moe.cpu_executor as cpu_executor

    constructed = {}
    executor = object()

    def build(cache, **kwargs):
        constructed.update(kwargs)
        return executor

    monkeypatch.setattr(cpu_executor, "CpuMoeExecutor", build)
    attached = []
    cache = SimpleNamespace(set_cpu_executor=attached.append)
    config = SimpleNamespace(
        cuda_graph_bs=graph_bs, cuda_graph_max_bs=max_bs,
        max_running_req=32, moe_cpu_threads=2,
    )
    layer = SimpleNamespace(
        top_k=2, activation="swiglu_clamp", apply_router_weight_on_input=False,
        alpha=1.0, limit=7.0,
        quant_method=SimpleNamespace(cpu_format="ds_fp4", cfg=SimpleNamespace(act_block=32)),
    )
    instance = SimpleNamespace(_init_free_memory=0, device=torch.device("cuda"))
    engine.Engine._init_cpu_moe_executor(instance, config, cache, [layer])

    assert constructed["max_tokens"] == tokens
    assert constructed["flag_slots_per_layer"] == slots
    assert constructed["fmt"] == "ds_fp4"
    assert constructed["act_block"] == 32
    assert attached == [executor]
    assert instance.cpu_moe_executor is executor


def test_cuda_flag_memops_keep_module_level_api(monkeypatch):
    """The ownership refactor must not reroute CUDA through the ROCm-only methods."""
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


def test_rocm_capture_with_exhausted_slots_does_not_use_host_callbacks(monkeypatch):
    import freetoken.moe.cpu_executor as cpu_executor

    monkeypatch.setattr(cpu_executor, "_IS_ROCM", True)
    monkeypatch.setattr(cpu_executor.torch.cuda, "is_current_stream_capturing", lambda: True)
    executor = object.__new__(cpu_executor.CpuMoeExecutor)
    executor._flag_sync = True
    executor._flag_slots = {}
    executor._io_for = lambda bs: {}
    executor._task_for = lambda layer, bs: 123
    with pytest.raises(RuntimeError, match="exhausted its graph-safe ROCm flag slots"):
        executor.decode_submit(0, SimpleNamespace(shape=(17, 64)), None, None)


def _check_rank_consensus(rank, init_method):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist
    import freetoken.engine.engine as engine

    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        with patch.object(engine, "is_rocm", return_value=True):
            for safe in ((True, True), (True, False), (False, True), (False, False)):
                config = SimpleNamespace(cuda_graph_bs=[1, 2], cuda_graph_max_bs=2)
                disabled = engine._disable_unsafe_rocm_cpu_moe_graph(
                    config, SimpleNamespace(graph_capture_safe=safe[rank]),
                    tp_cpu_group=dist.group.WORLD,
                )
                assert disabled == (not all(safe))
                assert config.cuda_graph_bs == ([1, 2] if all(safe) else [])
                assert config.cuda_graph_max_bs == (2 if all(safe) else 0)
    finally:
        dist.destroy_process_group()


def test_rocm_probe_failure_disables_graph_on_every_rank(tmp_path):
    """Exercise the startup consensus with two real CPU ranks, not two GPUs."""
    import torch.multiprocessing as mp

    mp.spawn(_check_rank_consensus, args=((tmp_path / "gloo-init").as_uri(),), nprocs=2)
