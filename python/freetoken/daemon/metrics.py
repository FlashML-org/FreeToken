"""The engine's OWN footprint. Boundary: only the serve tree's RAM/VRAM — system-wide host
telemetry is not this daemon's job.

RAM = summed PSS across the serve process group (shared pages counted once, the honest number).
VRAM = per-process GPU memory for those pids, via ``pynvml`` if importable (optional), else
parsed from ``nvidia-smi``, else the kernel's DRM fdinfo (AMD), else 0. All best-effort and off
the event loop — a missing GPU or absent NVML returns 0, never an error."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Callable

from . import osproc


def engine_footprint(pid: int | None) -> dict:
    if pid is None:
        return {"ramBytes": 0, "vramBytes": 0, "pids": []}
    pids = osproc.tree_pids(pid)
    ram = sum(osproc.read_pss_bytes(p) for p in pids)
    vram = vram_bytes_for_pids(pids)
    return {"ramBytes": ram, "vramBytes": vram, "pids": pids}


class FootprintCache:
    """Single-flight + short-TTL cache over ``engine_footprint``. Clients poll metrics frequently
    and an NVML/nvidia-smi probe can take seconds on a busy GPU; without this, every poll pays
    that cost and can back up the proxy executor. Concurrent callers within the TTL collapse to
    one probe."""

    def __init__(self, *, ttl_s: float = 2.0, now: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl_s
        self._now = now
        self._lock = threading.Lock()
        self._cache: dict[int | None, tuple[float, dict]] = {}

    def get(self, pid: int | None) -> dict:
        with self._lock:
            hit = self._cache.get(pid)
            if hit is not None and (self._now() - hit[0]) < self._ttl:
                return hit[1]
            val = engine_footprint(pid)
            self._cache[pid] = (self._now(), val)
            return val


def vram_bytes_for_pids(pids: list[int]) -> int:
    want = set(pids)
    if not want:
        return 0
    usage = _nvml_process_vram()
    if usage is None:
        usage = _smi_process_vram()
    if not usage:
        return _drm_fdinfo_vram(want)
    return sum(nbytes for p, nbytes in usage.items() if p in want)


_PROC = "/proc"


def _drm_fdinfo_vram(pids: set[int]) -> int:
    """VRAM held by ``pids`` per the kernel's DRM fdinfo, which amdgpu fills in for ROCm processes."""
    clients: dict[tuple[str, str], int] = {}
    for pid in pids:
        fdinfo = os.path.join(_PROC, str(pid), "fdinfo")
        try:
            fds = os.listdir(fdinfo)
        except OSError:
            continue
        for fd in fds:
            try:
                with open(os.path.join(fdinfo, fd), encoding="utf-8", errors="replace") as fh:
                    fields = dict(line.split(":", 1) for line in fh if ":" in line)
            except OSError:
                continue
            # drm-memory-<region> is the older spelling of drm-resident-<region>
            vram = (fields.get("drm-resident-vram") or fields.get("drm-memory-vram") or "").split()
            if "drm-client-id" not in fields or not vram or not vram[0].isdigit():
                continue
            unit = {"KiB": 1 << 10, "MiB": 1 << 20}.get(vram[1], 1) if len(vram) > 1 else 1
            # forked children and dup'ed fds repeat one client; count it once per device
            key = (fields.get("drm-pdev", "").strip(), fields["drm-client-id"].strip())
            clients[key] = int(vram[0]) * unit
    return sum(clients.values())


# NVML is initialized ONCE and held for the daemon's life — nvmlInit()+nvmlShutdown() on every
# call costs seconds on a busy GPU. None = not yet tried, True = ready, False = unavailable.
_NVML = {"ready": None}
_NVML_LOCK = threading.Lock()


def _nvml_ready():
    with _NVML_LOCK:
        if _NVML["ready"] is None:
            try:
                import pynvml  # optional; not a hard dep

                pynvml.nvmlInit()
                _NVML["ready"] = pynvml
            except Exception:  # noqa: BLE001
                _NVML["ready"] = False
        return _NVML["ready"]


def _nvml_process_vram() -> dict[int, int] | None:
    pynvml = _nvml_ready()
    if not pynvml:
        return None
    out: dict[int, int] = {}
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for getter in (
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
            ):
                if getter is None:
                    continue
                try:
                    for proc in getter(handle):
                        used = getattr(proc, "usedGpuMemory", None)
                        if used:  # None == "not available", per NVML
                            out[int(proc.pid)] = out.get(int(proc.pid), 0) + int(used)
                    break
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return out or None
    # Empty → NVML enumeration gave nothing usable (e.g. every process getter raised on a
    # driver/MIG mismatch); signal that with None so the nvidia-smi fallback still runs, matching
    # the error path above.
    return out or None


def _smi_process_vram() -> dict[int, int]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    usage: dict[int, int] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        usage[int(parts[0])] = usage.get(int(parts[0]), 0) + int(parts[1]) * 1024 * 1024  # MiB
    return usage
