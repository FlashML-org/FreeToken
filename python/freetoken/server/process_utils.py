"""Torch-free process helpers shared by native and Metal server lifecycles."""

from __future__ import annotations

import subprocess
from typing import Any, Iterable


def process_is_alive(process: Any) -> bool:
    """Normalize multiprocessing.Process and subprocess.Popen liveness."""
    if hasattr(process, "is_alive"):
        return bool(process.is_alive())
    return process.poll() is None


def terminate_backend_workers(processes: Iterable[Any]) -> None:
    """Best-effort, nonblocking SIGTERM for every live backend worker."""
    for process in processes or []:
        try:
            if process_is_alive(process):
                process.terminate()
        except Exception:  # noqa: BLE001 -- already gone or unqueryable
            continue


def reap_backend_workers(processes: Iterable[Any], timeout: float = 5.0) -> None:
    """Wait for SIGTERM and kill workers that remain alive after the timeout."""
    for process in processes or []:
        try:
            if hasattr(process, "join"):
                process.join(timeout=timeout)
            else:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass
            if process_is_alive(process):
                process.kill()
                if hasattr(process, "join"):
                    process.join(timeout=timeout)
                else:
                    process.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 -- already gone or unqueryable
            continue
