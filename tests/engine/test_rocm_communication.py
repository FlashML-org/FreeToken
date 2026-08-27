from types import SimpleNamespace

import torch

import freetoken.engine.engine as engine_module
import freetoken.kernel.backend as kernel_backend
from freetoken.engine.engine import Engine


def _config(*, use_pynccl: bool = True):
    return SimpleNamespace(
        use_pynccl=use_pynccl,
        tp_info=SimpleNamespace(size=2, rank=0),
        distributed_timeout=10,
        distributed_addr="tcp://127.0.0.1:29500",
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
