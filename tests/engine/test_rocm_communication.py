from types import SimpleNamespace

import pytest
import torch

import freetoken.engine.engine as engine_module
import freetoken.kernel.backend as kernel_backend
import freetoken.kernel.pynccl as pynccl_module
from freetoken.engine.engine import Engine


def _config(
    *,
    use_pynccl: bool = True,
    rank: int = 0,
    distributed_addr: str = "tcp://127.0.0.1:29500",
):
    return SimpleNamespace(
        use_pynccl=use_pynccl,
        tp_info=SimpleNamespace(size=2, rank=rank),
        distributed_timeout=30,
        distributed_addr=distributed_addr,
        max_forward_len=32,
        model_config=SimpleNamespace(hidden_size=64),
    )


def test_rocm_routes_tensor_parallel_communication_to_rccl(monkeypatch):
    calls = []
    cpu_group = object()

    def reject_pynccl(*_args):
        raise AssertionError("PyNCCL selected on ROCm")

    monkeypatch.setattr(kernel_backend, "is_rocm", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        lambda **kwargs: calls.append(("new", kwargs)) or cpu_group,
    )
    monkeypatch.setattr(engine_module, "enable_pynccl_distributed", reject_pynccl)

    result = Engine._init_communication(SimpleNamespace(), _config())

    assert result is cpu_group
    assert calls[0][1]["backend"] == "nccl"
    assert calls[1] == ("new", {"backend": "gloo"})


def test_cuda_keeps_custom_pynccl_path(monkeypatch):
    calls = []
    world_group = object()

    def reject_new_group(**_kwargs):
        raise AssertionError("unexpected RCCL path")

    monkeypatch.setattr(kernel_backend, "is_rocm", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=world_group))
    monkeypatch.setattr(torch.distributed, "new_group", reject_new_group)
    monkeypatch.setattr(
        engine_module,
        "enable_pynccl_distributed",
        lambda *args: calls.append(("pynccl", args)),
    )

    engine = SimpleNamespace(dtype=torch.float16)
    result = Engine._init_communication(engine, _config())

    assert result is world_group
    assert calls[0][1]["backend"] == "gloo"
    assert calls[1][0] == "pynccl"


def test_rocm_pynccl_loader_fails_before_linking_nccl(monkeypatch):
    def reject_load(*_args, **_kwargs):
        raise AssertionError("ROCm must fail before attempting to link libnccl")

    monkeypatch.setattr(kernel_backend, "is_rocm", lambda: True)
    monkeypatch.setattr(pynccl_module, "load_aot", reject_load)
    pynccl_module._load_nccl_module.cache_clear()

    with pytest.raises(RuntimeError, match="PyNCCL is NVIDIA-only"):
        pynccl_module._load_nccl_module()

    pynccl_module._load_nccl_module.cache_clear()


def _run_engine_rccl_rank(rank: int, world_size: int, distributed_addr: str) -> None:
    assert world_size == 2
    torch.cuda.set_device(rank)
    engine = SimpleNamespace(dtype=torch.float32)
    cpu_group = None
    try:
        cpu_group = Engine._init_communication(
            engine,
            _config(rank=rank, distributed_addr=distributed_addr),
        )
        assert torch.distributed.get_backend() == "nccl"
        assert torch.distributed.get_backend(cpu_group) == "gloo"

        value = torch.tensor(float(rank + 1), device=f"cuda:{rank}")
        torch.distributed.all_reduce(value)
        assert value.item() == 3.0
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(
    torch.version.hip is None or torch.cuda.device_count() < 2,
    reason="two ROCm GPUs are required",
)
def test_engine_route_runs_two_rank_rccl_all_reduce(tmp_path):
    import torch.multiprocessing as mp

    distributed_addr = f"file://{tmp_path / 'rccl-rendezvous'}"
    mp.spawn(
        _run_engine_rccl_rank,
        args=(2, distributed_addr),
        nprocs=2,
        join=True,
    )
