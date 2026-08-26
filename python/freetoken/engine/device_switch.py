"""Device resolution with an OR-switch fallback.

FreeToken's engine (`engine.py`) historically hard-codes a CUDA device at init:

    self.device = torch.device(f"cuda:{config.tp_info.rank}")
    torch.cuda.set_device(self.device)
    self.stream = torch.cuda.Stream()

That makes the whole server require an NVIDIA GPU even for the parts that could run
on CPU (the `--moe-backend cpu` decode fallback already exists in C++). This module
introduces the OR-switch Peter asked for: **prefer CUDA, fall back to CPU**.

The idiom is deliberately a plain `or`-style resolver so it reads as the requested
"or function / switching type":

    resolve_device(rank) -> cuda:rank   if torch.cuda.is_available()
                          else cpu

Everything that touches the device in the engine should route through `resolve_device()`
and `guard_cuda()` so the same code runs on a GPU box *and* a CPU-only box (e.g. a
free x86-Linux CI runner, or any consumer machine without an NVIDIA card).
"""

from __future__ import annotations

import torch
from torch import device as TorchDevice


def has_cuda() -> bool:
    """True iff a CUDA device is actually usable right now."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_device(rank: int = 0) -> TorchDevice:
    """OR-switch: CUDA when present, otherwise CPU.

    Args:
        rank: tensor-parallel rank; mapped to ``cuda:{rank}`` on GPU, ignored on CPU.

    Returns:
        ``torch.device("cuda", rank)`` when CUDA is available, else
        ``torch.device("cpu")``. This is the single switch point the engine uses
        instead of the old hard-coded ``cuda:{rank}``.
    """
    if has_cuda():
        return TorchDevice("cuda", rank)
    return TorchDevice("cpu")


def guard_cuda() -> str:
    """Return the backend tag the engine should report.

    Used for logging / config resolution so the rest of the stack knows whether
    it is running on the CUDA path or the CPU-fallback path.
    """
    return "cuda" if has_cuda() else "cpu"


def make_stream(dev: TorchDevice):
    """OR-switch for the decode stream.

    CUDA path uses ``torch.cuda.Stream()``; CPU path uses ``None`` (the executor
    is driven by the in-process worker pool, not a GPU stream).
    """
    if dev.type == "cuda":
        return torch.cuda.Stream()
    return None
