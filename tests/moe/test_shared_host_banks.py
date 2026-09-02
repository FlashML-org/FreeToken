"""shmem (memfd) shared HostBank pool: TP ranks map one inode, resident once.

CPU-only. torch.distributed is replaced by a recording fake so the rank-0 ->
rank-N handoff, the gloo-group requirement and the preflights are all
exercised without a process group or a GPU.
"""

from __future__ import annotations

import sys

import pytest
import torch

from freetoken.distributed import reset_tp_info, set_tp_info
from freetoken.distributed.info import reset_tp_cpu_group, set_tp_cpu_group
from freetoken.moe import host_banks as hb
from freetoken.moe.host_banks import (
    DEFAULT_BANK_SHARE_NEED_BYTES,
    HostBank,
    alloc_banks,
    alloc_layer_banks,
    bank_sharing_enabled,
    kfd_system_mem_limit_bytes,
    prepare_shared_banks,
    require_shared_pool_capacity,
)
from freetoken.runtime import gpu


@pytest.fixture(autouse=True)
def _clean_tp_state(monkeypatch):
    monkeypatch.delenv("FREETOKEN_BANK_SHARE", raising=False)
    reset_tp_info()
    reset_tp_cpu_group()
    yield
    reset_tp_info()
    reset_tp_cpu_group()


class _FakeDist:
    """torch.distributed stand-in: records broadcast calls; delivers a canned payload to receivers."""

    def __init__(self, backend="nccl", payload=None):
        self._backend = backend
        self.payload = payload
        self.calls: list[dict] = []

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_backend(self):
        return self._backend

    def broadcast_object_list(self, box, src=0, group=None):
        self.calls.append({"src": src, "group": group, "sent": box[0]})
        if box[0] is None:
            box[0] = self.payload


def _install_fake_dist(monkeypatch, fake):
    # ``import torch.distributed as dist`` resolves getattr(torch, "distributed"), so
    # the attribute must be patched, not only sys.modules.
    monkeypatch.setattr(torch, "distributed", fake)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake)


# --- HostBank shmem semantics ---------------------------------------------------


def test_shared_bank_opener_sees_creator_writes():
    a = HostBank((8,), torch.float32, share=True, name="w")
    assert a.share_path is not None and a.share_path.startswith("/proc/")
    a.tensor[:] = 3.5
    a.flush()
    b = HostBank((8,), torch.float32, share_path=a.share_path)
    assert float(b.tensor[0]) == 3.5
    b.tensor[7] = -1.0
    assert float(a.tensor[7]) == -1.0  # same inode, both directions
    assert b.share_path == a.share_path


def test_private_bank_has_no_share_path():
    assert HostBank((2,), torch.uint8).share_path is None


def test_share_and_share_path_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        HostBank((2,), torch.uint8, share=True, share_path="/proc/self/fd/0")


def test_opener_rejects_undersized_bank():
    small = HostBank((2,), torch.uint8, share=True, name="small")
    with pytest.raises(RuntimeError, match="need"):
        HostBank((1 << 20,), torch.uint8, share_path=small.share_path)


# --- sharing decision -------------------------------------------------------------


def test_sharing_only_for_tp_gt_1(monkeypatch):
    assert bank_sharing_enabled() is False
    set_tp_info(0, 2)
    assert bank_sharing_enabled() is True
    monkeypatch.setenv("FREETOKEN_BANK_SHARE", "0")
    assert bank_sharing_enabled() is False


def test_anonymous_mmap_when_tp1():
    banks = alloc_layer_banks({"w": ((2,), torch.uint8)}, 1)
    assert banks["w"][0].share_path is None


def test_prepare_shared_banks(monkeypatch):
    assert prepare_shared_banks(need_bytes=0) is None  # TP=1
    set_tp_info(0, 2)
    assert prepare_shared_banks(need_bytes=0) == "memfd"
    monkeypatch.setenv("FREETOKEN_BANK_SHARE", "0")
    assert prepare_shared_banks(need_bytes=0) is None
    assert DEFAULT_BANK_SHARE_NEED_BYTES == 43 * 256 * 13_369_344


# --- rank handoff over the gloo CPU group --------------------------------------------


def test_alloc_layer_banks_tp2_rank0_creates_and_broadcasts(monkeypatch):
    set_tp_info(0, 2)
    gloo = object()
    set_tp_cpu_group(gloo)
    fake = _FakeDist(backend="nccl")
    _install_fake_dist(monkeypatch, fake)
    banks = alloc_layer_banks({"gate": ((4, 8), torch.uint8), "down": ((4, 8), torch.uint8)}, 2)
    assert [c["group"] for c in fake.calls] == [gloo]  # CPU group, never the GPU world
    sent = fake.calls[0]["sent"]
    assert [(n, l) for n, l, _ in sent] == [("gate", 0), ("gate", 1), ("down", 0), ("down", 1)]
    assert all(p.startswith("/proc/") for _, _, p in sent)
    assert banks["gate"][1].share_path == sent[1][2]


def test_alloc_layer_banks_tp2_rank1_maps_rank0_pool(monkeypatch):
    # rank 0 side (this process) builds the pool ...
    set_tp_info(0, 2)
    set_tp_cpu_group(object())
    fake0 = _FakeDist(backend="nccl")
    _install_fake_dist(monkeypatch, fake0)
    rank0 = alloc_layer_banks({"gate": ((4, 8), torch.uint8)}, 2)
    rank0["gate"][1].tensor.fill_(9)
    rank0["gate"][1].flush()
    # ... and rank 1 receives its handoff and maps the same inodes.
    reset_tp_info()
    set_tp_info(1, 2)
    fake1 = _FakeDist(backend="nccl", payload=fake0.calls[0]["sent"])
    _install_fake_dist(monkeypatch, fake1)
    rank1 = alloc_layer_banks({"gate": ((4, 8), torch.uint8)}, 2)
    assert fake1.calls[0]["sent"] is None  # receiver contributes nothing
    assert rank1["gate"][1].share_path == rank0["gate"][1].share_path
    assert int(rank1["gate"][1].tensor[3, 7]) == 9
    assert int(rank1["gate"][0].tensor[0, 0]) == 0


def test_alloc_layer_banks_layout_mismatch_is_loud(monkeypatch):
    set_tp_info(1, 2)
    set_tp_cpu_group(object())
    fake = _FakeDist(backend="nccl", payload=[("gate", 0, "/proc/self/fd/0")])
    _install_fake_dist(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="disagree on the bank layout"):
        alloc_layer_banks({"gate": ((4, 8), torch.uint8)}, 2)


def test_alloc_banks_tp2_handoff(monkeypatch):
    set_tp_info(0, 2)
    set_tp_cpu_group(object())
    fake0 = _FakeDist(backend="nccl")
    _install_fake_dist(monkeypatch, fake0)
    rank0 = alloc_banks({"w": ((3,), torch.float32)})
    rank0["w"].tensor[:] = 2.5
    reset_tp_info()
    set_tp_info(1, 2)
    fake1 = _FakeDist(backend="nccl", payload=fake0.calls[0]["sent"])
    _install_fake_dist(monkeypatch, fake1)
    rank1 = alloc_banks({"w": ((3,), torch.float32)})
    assert float(rank1["w"].tensor[2]) == 2.5


def test_broadcast_refuses_gpu_world_without_cpu_group(monkeypatch):
    """Without a registered gloo group and with an NCCL/RCCL default group, the
    handoff must refuse rather than run a collective on the GPU world (the
    rank-guess hang class)."""
    fake = _FakeDist(backend="nccl")
    _install_fake_dist(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="no gloo CPU group"):
        hb._tp_cpu_broadcast(["x"])
    assert fake.calls == []


def test_broadcast_plain_on_gloo_world(monkeypatch):
    fake = _FakeDist(backend="gloo", payload=["from0"])
    _install_fake_dist(monkeypatch, fake)
    assert hb._tp_cpu_broadcast(None) == ["from0"]
    assert fake.calls == [{"src": 0, "group": None, "sent": None}]


# --- preflights -------------------------------------------------------------------


def _mem(monkeypatch, *, total_gib, avail_gib):
    vals = {"MemTotal": total_gib << 30, "MemAvailable": avail_gib << 30}
    monkeypatch.setattr(hb, "_meminfo_bytes", lambda key: vals[key])


def test_pool_must_fit_in_ram(monkeypatch):
    _mem(monkeypatch, total_gib=172, avail_gib=100)
    with pytest.raises(RuntimeError, match="MemAvailable is 100.0 GiB"):
        require_shared_pool_capacity(need_bytes=137 << 30, tp_size=2)


def test_kfd_limit_from_memtotal(monkeypatch):
    _mem(monkeypatch, total_gib=172, avail_gib=160)
    monkeypatch.setattr(hb, "_read_sysfs", lambda path: "N")
    total = 172 << 30
    assert kfd_system_mem_limit_bytes() == total - (total >> 6)
    monkeypatch.setattr(hb, "_read_sysfs", lambda path: "Y")
    assert kfd_system_mem_limit_bytes() is None
    monkeypatch.setattr(hb, "_read_sysfs", lambda path: None)  # not amdgpu
    assert kfd_system_mem_limit_bytes() is None


def test_kfd_budget_refuses_double_counted_pool(monkeypatch):
    """2 ranks x 137 GiB > ~169 GiB: the second rank's pins would fail mid-load."""
    _mem(monkeypatch, total_gib=172, avail_gib=160)
    monkeypatch.setattr(hb, "_read_sysfs", lambda path: "N")
    monkeypatch.setattr(gpu, "is_hip", lambda: True)
    with pytest.raises(RuntimeError, match="no_system_mem_limit"):
        require_shared_pool_capacity(need_bytes=137 << 30, tp_size=2)
    # limit disabled by the operator -> passes
    monkeypatch.setattr(hb, "_read_sysfs", lambda path: "Y")
    require_shared_pool_capacity(need_bytes=137 << 30, tp_size=2)


def test_kfd_budget_only_on_hip_tp_gt_1(monkeypatch):
    _mem(monkeypatch, total_gib=172, avail_gib=160)
    monkeypatch.setattr(hb, "_read_sysfs", lambda path: "N")
    monkeypatch.setattr(gpu, "is_hip", lambda: False)
    require_shared_pool_capacity(need_bytes=137 << 30, tp_size=2)  # NVIDIA: no KFD budget
    monkeypatch.setattr(gpu, "is_hip", lambda: True)
    require_shared_pool_capacity(need_bytes=137 << 30, tp_size=1)  # one registration fits
    require_shared_pool_capacity(need_bytes=0, tp_size=2)  # unit-test skip


def test_kfd_budget_passes_when_it_fits(monkeypatch):
    _mem(monkeypatch, total_gib=172, avail_gib=160)
    monkeypatch.setattr(hb, "_read_sysfs", lambda path: "N")
    monkeypatch.setattr(gpu, "is_hip", lambda: True)
    require_shared_pool_capacity(need_bytes=60 << 30, tp_size=2)  # 120 < 169
