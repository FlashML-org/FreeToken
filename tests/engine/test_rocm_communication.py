from __future__ import annotations

import sys
from types import SimpleNamespace
from enum import IntEnum

import pytest
import torch


def _engine_stub():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.dtype = torch.bfloat16
    return engine


def _config(*, tp_size: int, use_pynccl: bool):
    return SimpleNamespace(
        tp_info=SimpleNamespace(rank=0, size=tp_size),
        use_pynccl=use_pynccl,
        distributed_timeout=1.0,
        distributed_addr="tcp://127.0.0.1:2333",
        max_forward_len=4,
        model_config=SimpleNamespace(hidden_size=8),
    )


@pytest.fixture
def fake_process_groups(monkeypatch):
    # Keep this selection test independent of CUDA-only flashlib wheels. Engine
    # imports the offload modules at module load, but communication selection does
    # not execute those kernels.
    flashlib = SimpleNamespace()
    kernels = SimpleNamespace()

    class Stat(IntEnum):
        ACTIVE = 0
        MISS = 1
        CALLS = 2

    slot_cache = SimpleNamespace(N_STATS=3, Stat=Stat, lru_ensure=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "flashlib", flashlib)
    monkeypatch.setitem(sys.modules, "flashlib.kernels", kernels)
    monkeypatch.setitem(sys.modules, "flashlib.kernels.slot_cache", slot_cache)

    from freetoken.engine import engine as engine_module

    calls = []
    world = object()
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=world))
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        lambda **kwargs: calls.append(("new", kwargs)) or object(),
    )
    pynccl_calls = []
    monkeypatch.setattr(
        engine_module,
        "enable_pynccl_distributed",
        lambda *args: pynccl_calls.append(args),
    )
    return calls, pynccl_calls


def test_rocm_tp2_uses_rccl_via_pytorch_nccl(monkeypatch, fake_process_groups):
    from freetoken.engine import engine as engine_module

    calls, pynccl_calls = fake_process_groups
    monkeypatch.setattr(engine_module, "is_rocm", lambda: True, raising=False)

    group = _engine_stub()._init_communication(_config(tp_size=2, use_pynccl=True))

    assert group is not None
    assert calls[0][1]["backend"] == "nccl"
    assert calls[1] == ("new", {"backend": "gloo"})
    assert pynccl_calls == []


def test_cuda_tp2_retains_pynccl(monkeypatch, fake_process_groups):
    from freetoken.engine import engine as engine_module

    calls, pynccl_calls = fake_process_groups
    monkeypatch.setattr(engine_module, "is_rocm", lambda: False, raising=False)

    group = _engine_stub()._init_communication(_config(tp_size=2, use_pynccl=True))

    assert group is not None
    assert calls[0][1]["backend"] == "gloo"
    assert len(calls) == 1
    assert len(pynccl_calls) == 1
    assert pynccl_calls[0][2] == 64


@pytest.mark.parametrize("rocm", [False, True])
def test_tp1_does_not_create_device_communicator(monkeypatch, fake_process_groups, rocm):
    from freetoken.engine import engine as engine_module

    calls, pynccl_calls = fake_process_groups
    monkeypatch.setattr(engine_module, "is_rocm", lambda: rocm, raising=False)

    _engine_stub()._init_communication(_config(tp_size=1, use_pynccl=True))

    assert calls[0][1]["backend"] == "gloo"
    assert pynccl_calls == []


def test_pynccl_loader_rejects_rocm_before_jit(monkeypatch):
    from freetoken.kernel import pynccl

    loaded = []
    monkeypatch.setattr(pynccl, "is_rocm", lambda: True)
    monkeypatch.setattr(pynccl, "load_aot", lambda *args, **kwargs: loaded.append(1))
    pynccl._load_nccl_module.cache_clear()

    with pytest.raises(RuntimeError, match="PyNCCL is CUDA-only"):
        pynccl._load_nccl_module()

    assert loaded == []
    pynccl._load_nccl_module.cache_clear()


def test_pynccl_loader_retains_cuda_path(monkeypatch):
    from freetoken.kernel import pynccl

    sentinel = object()
    monkeypatch.setattr(pynccl, "is_rocm", lambda: False)
    monkeypatch.setattr(pynccl, "load_aot", lambda *args, **kwargs: sentinel)
    pynccl._load_nccl_module.cache_clear()

    assert pynccl._load_nccl_module() is sentinel
    pynccl._load_nccl_module.cache_clear()
