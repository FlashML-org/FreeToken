from __future__ import annotations

import multiprocessing as mp
import os
from queue import Empty
from types import SimpleNamespace

import pytest
import torch

import freetoken.distributed.impl as distributed_impl
import freetoken.engine.engine as engine_module
from freetoken.distributed import DistributedCommunicator, SingleRankProcessGroup
from freetoken.engine.engine import Engine
from freetoken.scheduler.io import SchedulerIOMixin


@pytest.fixture(autouse=True)
def _restore_distributed_plugins():
    original = list(DistributedCommunicator.plugins)
    yield
    DistributedCommunicator.plugins = original


def _config(*, size: int, use_pynccl: bool) -> SimpleNamespace:
    return SimpleNamespace(
        use_pynccl=use_pynccl,
        tp_info=SimpleNamespace(size=size, rank=0),
        distributed_timeout=10,
        distributed_addr="tcp://127.0.0.1:29500",
        max_forward_len=32,
        model_config=SimpleNamespace(hidden_size=64),
    )


def _reject_torch_distributed(*_args, **_kwargs):
    raise AssertionError("torch.distributed must not be used by the single-rank fallback")


def test_process_group_capability_requires_available_runtime(monkeypatch):
    monkeypatch.setattr(distributed_impl.dist, "is_available", lambda: False)

    assert distributed_impl.torch_distributed_process_group_available() is False


def test_single_rank_falls_back_without_torch_distributed(monkeypatch):
    monkeypatch.setattr(engine_module, "torch_distributed_process_group_available", lambda: False)
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "init_process_group",
        _reject_torch_distributed,
        raising=False,
    )

    group = Engine._init_communication(
        SimpleNamespace(dtype=torch.float16), _config(size=1, use_pynccl=True)
    )

    assert isinstance(group, SingleRankProcessGroup)
    value = torch.tensor([1.0])
    assert DistributedCommunicator().all_reduce(value) is value


def test_single_rank_keeps_gloo_when_torch_distributed_is_available(monkeypatch):
    calls = []
    world_group = object()
    monkeypatch.setattr(engine_module, "torch_distributed_process_group_available", lambda: True)
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "group",
        SimpleNamespace(WORLD=world_group),
        raising=False,
    )
    monkeypatch.setattr(
        engine_module,
        "enable_pynccl_distributed",
        lambda *args: calls.append(("pynccl", args)),
    )

    result = Engine._init_communication(
        SimpleNamespace(dtype=torch.float16), _config(size=1, use_pynccl=True)
    )

    assert result is world_group
    assert calls[0][1]["backend"] == "gloo"
    assert calls[1][0] == "pynccl"


def test_multi_rank_pynccl_route_is_unchanged(monkeypatch):
    calls = []
    world_group = object()
    monkeypatch.setattr(engine_module, "torch_distributed_process_group_available", lambda: False)
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "group",
        SimpleNamespace(WORLD=world_group),
        raising=False,
    )
    monkeypatch.setattr(
        engine_module,
        "enable_pynccl_distributed",
        lambda *args: calls.append(("pynccl", args)),
    )

    result = Engine._init_communication(
        SimpleNamespace(dtype=torch.float16), _config(size=2, use_pynccl=True)
    )

    assert result is world_group
    assert calls[0][1]["backend"] == "gloo"
    assert calls[1][0] == "pynccl"


def test_multi_rank_native_route_is_unchanged(monkeypatch):
    calls = []
    cpu_group = object()
    monkeypatch.setattr(engine_module, "torch_distributed_process_group_available", lambda: False)
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "new_group",
        lambda **kwargs: calls.append(("new", kwargs)) or cpu_group,
        raising=False,
    )

    result = Engine._init_communication(
        SimpleNamespace(dtype=torch.float16), _config(size=2, use_pynccl=False)
    )

    assert result is cpu_group
    assert calls == [
        (
            "init",
            {
                "backend": "nccl",
                "rank": 0,
                "world_size": 2,
                "timeout": engine_module.timedelta(seconds=10),
                "init_method": "tcp://127.0.0.1:29500",
            },
        ),
        ("new", {"backend": "gloo"}),
    ]


def test_single_rank_group_completes_barrier_and_broadcast():
    group = SingleRankProcessGroup()
    value = torch.tensor([7])

    assert group.barrier().wait() is True
    assert group.broadcast(value, root=0).wait() is True
    assert value.item() == 7
    with pytest.raises(ValueError, match="root must be 0"):
        group.broadcast(value, root=1)


def test_single_rank_layer_collectives_are_identity():
    distributed_impl.enable_single_rank_distributed()
    communicator = DistributedCommunicator()
    value = torch.tensor([[1.0, 2.0]])

    assert communicator.all_reduce(value) is value
    assert communicator.all_gather(value) is value


def test_single_rank_memory_sync_skips_process_group_collective(monkeypatch):
    monkeypatch.setattr(engine_module.torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(engine_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(engine_module.torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(engine_module, "get_free_memory", lambda _device: 123456)
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "all_reduce",
        _reject_torch_distributed,
        raising=False,
    )
    engine = SimpleNamespace(device="cuda:0", tp_cpu_group=SingleRankProcessGroup())

    assert Engine._sync_get_memory(engine) == (123456, 123456)


def test_scheduler_barrier_uses_local_group_contract():
    scheduler = SchedulerIOMixin.__new__(SchedulerIOMixin)
    scheduler.tp_cpu_group = SingleRankProcessGroup()

    SchedulerIOMixin.sync_all_ranks(scheduler)


def test_single_rank_shutdown_skips_torch_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "destroy_process_group",
        _reject_torch_distributed,
        raising=False,
    )
    monkeypatch.setattr(engine_module, "destroy_distributed", lambda: calls.append("plugins"))
    engine = SimpleNamespace(
        graph_runner=SimpleNamespace(destroy_cuda_graphs=lambda: calls.append("graphs")),
        tp_cpu_group=SingleRankProcessGroup(),
    )

    Engine.shutdown(engine)

    assert calls == ["graphs", "plugins"]


def test_multi_rank_shutdown_keeps_torch_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(
        engine_module.torch.distributed,
        "destroy_process_group",
        lambda: calls.append("process-group"),
        raising=False,
    )
    monkeypatch.setattr(engine_module, "destroy_distributed", lambda: calls.append("plugins"))
    engine = SimpleNamespace(
        graph_runner=SimpleNamespace(destroy_cuda_graphs=lambda: calls.append("graphs")),
        tp_cpu_group=object(),
    )

    Engine.shutdown(engine)

    assert calls == ["graphs", "process-group", "plugins"]


def _single_rank_spawn_worker(result_queue) -> None:
    try:
        import torch as child_torch

        import freetoken.engine.engine as child_engine_module
        from freetoken.distributed import DistributedCommunicator as ChildCommunicator
        from freetoken.distributed import SingleRankProcessGroup as ChildGroup
        from freetoken.engine.engine import Engine as ChildEngine
        from freetoken.scheduler.io import SchedulerIOMixin as ChildSchedulerIOMixin

        child_engine_module.torch_distributed_process_group_available = lambda: False
        child_engine_module.torch.distributed.init_process_group = _reject_torch_distributed
        child_engine_module.torch.distributed.destroy_process_group = _reject_torch_distributed
        group = ChildEngine._init_communication(
            SimpleNamespace(dtype=child_torch.float16), _config(size=1, use_pynccl=True)
        )
        value = child_torch.tensor([3.0])
        communicator = ChildCommunicator()
        scheduler = ChildSchedulerIOMixin.__new__(ChildSchedulerIOMixin)
        scheduler.tp_cpu_group = group
        scheduler.sync_all_ranks()
        group.broadcast(value, root=0).wait()
        all_reduce_identity = communicator.all_reduce(value) is value
        fake_engine = SimpleNamespace(
            graph_runner=SimpleNamespace(destroy_cuda_graphs=lambda: None),
            tp_cpu_group=group,
        )
        ChildEngine.shutdown(fake_engine)
        result_queue.put(
            {
                "ok": True,
                "group": type(group).__name__,
                "all_reduce_identity": all_reduce_identity,
                "value": value.item(),
                "is_local_group": isinstance(group, ChildGroup),
            }
        )
    except BaseException as exc:  # pragma: no cover - surfaced in parent assertion
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise


@pytest.mark.skipif(os.name != "nt", reason="native Windows spawn smoke")
def test_windows_spawn_smoke_without_torch_distributed():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_single_rank_spawn_worker, args=(result_queue,))
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("single-rank spawn smoke timed out")

    try:
        result = result_queue.get(timeout=5)
    except Empty:
        pytest.fail(f"spawn worker produced no result (exit code {process.exitcode})")
    finally:
        result_queue.close()
        result_queue.join_thread()

    assert process.exitcode == 0
    assert result == {
        "ok": True,
        "group": "SingleRankProcessGroup",
        "all_reduce_identity": True,
        "value": 3.0,
        "is_local_group": True,
    }
