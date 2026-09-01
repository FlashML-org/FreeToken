"""CPU-only dual-card TP / RCCL wiring. No AMD GPU required."""

from __future__ import annotations

import pytest

from freetoken.distributed.collectives import (
    apply_hip_rccl_env,
    gpu_dist_backend,
    hip_tp_missing,
    pynccl_supported,
    require_pynccl_allowed,
    tp_comm_plan,
)
from freetoken.gpu_select import parse_gpu_spec
from freetoken.runtime import gpu
from freetoken.server.args import infer_tensor_parallel_size


@pytest.fixture(autouse=True)
def _clear_vendor_cache():
    gpu.vendor.cache_clear()
    yield
    gpu.vendor.cache_clear()


def test_gpu_0_1_parses_two_visible_indices():
    assert parse_gpu_spec("0,1") == ("0", "1")
    assert parse_gpu_spec("1,2") == ("1", "2")


def test_gpu_list_infers_tp_size():
    assert infer_tensor_parallel_size(("0", "1"), 1) == 2
    assert infer_tensor_parallel_size(("0", "1", "2"), 1) == 3
    assert infer_tensor_parallel_size(("0",), 1) == 1
    assert infer_tensor_parallel_size((), 1) == 1
    assert infer_tensor_parallel_size(("0", "1"), 2) == 2


def test_gpu_list_rejects_tp_mismatch():
    with pytest.raises(ValueError, match="2 entries"):
        infer_tensor_parallel_size(("0", "1"), 4)


def test_nvidia_tp_plan_unchanged():
    assert tp_comm_plan(1, use_pynccl=True, hip=False) == "gloo"
    assert tp_comm_plan(2, use_pynccl=True, hip=False) == "gloo+pynccl"
    assert tp_comm_plan(2, use_pynccl=False, hip=False) == "nccl+gloo"


def test_hip_tp_plan_is_rccl_never_pynccl():
    assert tp_comm_plan(2, use_pynccl=True, hip=True) == "rccl+gloo"
    assert tp_comm_plan(2, use_pynccl=False, hip=True) == "rccl+gloo"
    assert tp_comm_plan(1, use_pynccl=True, hip=True) == "gloo"


def test_hip_refuses_pynccl(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()
    assert pynccl_supported() is False
    with pytest.raises(RuntimeError, match="NVIDIA-only"):
        require_pynccl_allowed()
    with pytest.raises(RuntimeError, match="PyNCCL|NCCL 2.27"):
        require_pynccl_allowed()


def test_nvidia_allows_pynccl(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "nvidia")
    gpu.vendor.cache_clear()
    assert pynccl_supported() is True
    require_pynccl_allowed()


def test_hip_gpu_dist_backend_prefers_rccl(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()

    class _Dist:
        @staticmethod
        def is_rccl_available():
            return True

        @staticmethod
        def is_nccl_available():
            return True

    monkeypatch.setattr("torch.distributed.is_rccl_available", _Dist.is_rccl_available, raising=False)
    monkeypatch.setattr("torch.distributed.is_nccl_available", _Dist.is_nccl_available)
    assert gpu_dist_backend() == "rccl"


def test_hip_gpu_dist_backend_rocm_nccl_alias(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()

    class _Dist:
        @staticmethod
        def is_nccl_available():
            return True

    import torch.distributed as dist

    if hasattr(dist, "is_rccl_available"):
        monkeypatch.setattr(dist, "is_rccl_available", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", _Dist.is_nccl_available)
    assert gpu_dist_backend() == "nccl"


def test_hip_gpu_dist_backend_missing_is_loud(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()
    import torch.distributed as dist

    if hasattr(dist, "is_rccl_available"):
        monkeypatch.setattr(dist, "is_rccl_available", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: False)
    with pytest.raises(RuntimeError, match="RCCL"):
        gpu_dist_backend()


def test_hip_tp_missing_lists_rccl_and_cards(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()
    monkeypatch.setattr(gpu, "discrete_tp_devices", lambda devices=None: [])
    import torch.distributed as dist

    if hasattr(dist, "is_rccl_available"):
        monkeypatch.setattr(dist, "is_rccl_available", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: False)
    missing = hip_tp_missing(2)
    assert missing
    text = "\n".join(missing)
    assert "RCCL" in text
    assert "discrete GPUs" in text
    assert hip_tp_missing(1) == []


def test_apply_hip_rccl_env_sets_ib_disable(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)
    gpu.vendor.cache_clear()
    written = apply_hip_rccl_env()
    assert written["NCCL_IB_DISABLE"] == "1"
    assert os_env_ib() == "1"
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    written = apply_hip_rccl_env()
    assert "NCCL_IB_DISABLE" not in written
    assert os_env_ib() == "0"


def os_env_ib() -> str:
    import os

    return os.environ.get("NCCL_IB_DISABLE", "")


def test_tp_ranks_are_two_r9700s_not_igpu():
    physical = [
        {"index": 0, "name": "AMD Radeon Graphics", "arch": "gfx1036", "hidden_igpu": True},
        {"index": 1, "name": "AMD Radeon AI PRO R9700", "arch": "gfx1201", "hidden_igpu": False},
        {"index": 2, "name": "AMD Radeon AI PRO R9700", "arch": "gfx1201", "hidden_igpu": False},
    ]
    ranks = gpu.tp_rank_devices(2, physical)
    assert len(ranks) == 2
    assert all(d["arch"] == "gfx1201" for d in ranks)
    assert all("R9700" in d["name"] for d in ranks)
    assert gpu.discrete_tp_devices(physical)[0]["index"] == 1


def test_tp_ranks_too_few_discrete(monkeypatch):
    monkeypatch.delenv("FREETOKEN_INCLUDE_IGPU", raising=False)
    physical = [
        {"index": 0, "name": "AMD Radeon Graphics", "arch": "gfx1036", "hidden_igpu": True},
        {"index": 1, "name": "AMD Radeon AI PRO R9700", "arch": "gfx1201", "hidden_igpu": False},
    ]
    with pytest.raises(RuntimeError, match="TP=2"):
        gpu.tp_rank_devices(2, physical)


def test_init_pynccl_module_load_blocked_on_hip(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()
    from freetoken.kernel.pynccl import _load_nccl_module

    _load_nccl_module.cache_clear()
    with pytest.raises(RuntimeError, match="NVIDIA-only"):
        _load_nccl_module()
    _load_nccl_module.cache_clear()
