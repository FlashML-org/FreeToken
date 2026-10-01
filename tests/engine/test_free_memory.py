"""The free device memory the engine sizes its pools against.

Under WDDM (Windows) cudaMemGetInfo leaves out what other processes hold, so a card the desktop uses
reads as free and a pool sized to it pages into shared memory. NVML counts every process.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from freetoken import gpu_select
from freetoken.engine import graph

GIB = 1 << 30
DEVICE_UUID = "80472e39-ad2b-441d-fcff-0f3cc6ab9db2"


class FakeNvml:
    """NVML with one device and ``free`` bytes free on it; records the UUIDs it is asked about."""

    def __init__(self, free: int):
        self.free = free
        self.asked: list[bytes] = []

    def nvmlInit(self) -> int:
        return 0

    def nvmlShutdown(self) -> int:
        return 0

    def nvmlDeviceGetHandleByUUID(self, uuid: bytes, handle) -> int:
        self.asked.append(uuid)
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle, memory) -> int:
        memory._obj.free = self.free
        return 0


def _machine(monkeypatch, *, cuda_free: int, nvml: FakeNvml | None, platform: str) -> None:
    """Fakes CUDA's reading, the device's UUID and the NVML library the engine would load."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (cuda_free, 24 * GIB))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=DEVICE_UUID))
    monkeypatch.setattr(gpu_select, "_load_nvml", lambda: nvml)
    monkeypatch.setattr(sys, "platform", platform)


def test_windows_sizes_against_what_other_processes_leave(monkeypatch):
    # the reading that spilled: CUDA says 23 GiB free while the desktop holds more than 2 GiB of 24
    nvml = FakeNvml(free=21 * GIB)
    _machine(monkeypatch, cuda_free=23 * GIB, nvml=nvml, platform="win32")
    assert graph.get_free_memory(torch.device("cuda", 0)) == 21 * GIB
    assert nvml.asked == [f"GPU-{DEVICE_UUID}".encode("ascii")]


def test_windows_keeps_cudas_figure_when_it_is_the_smaller(monkeypatch):
    _machine(monkeypatch, cuda_free=20 * GIB, nvml=FakeNvml(free=21 * GIB), platform="win32")
    assert graph.get_free_memory(torch.device("cuda", 0)) == 20 * GIB


def test_windows_without_nvml_keeps_cudas_figure(monkeypatch):
    _machine(monkeypatch, cuda_free=23 * GIB, nvml=None, platform="win32")
    assert graph.get_free_memory(torch.device("cuda", 0)) == 23 * GIB


def test_linux_reads_cuda_alone(monkeypatch):
    # cudaMemGetInfo already counts every process there, so NVML is not asked
    nvml = FakeNvml(free=21 * GIB)
    _machine(monkeypatch, cuda_free=23 * GIB, nvml=nvml, platform="linux")
    assert graph.get_free_memory(torch.device("cuda", 0)) == 23 * GIB
    assert nvml.asked == []
