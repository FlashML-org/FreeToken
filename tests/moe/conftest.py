"""Keep pure GGUF MoE contract tests importable without CUDA-only flashlib wheels."""

from enum import IntEnum
import sys
from types import SimpleNamespace


if "flashlib.kernels.slot_cache" not in sys.modules:
    class Stat(IntEnum):
        ACTIVE = 0
        MISS = 1
        CALLS = 2

    flashlib = SimpleNamespace()
    kernels = SimpleNamespace()
    slot_cache = SimpleNamespace(N_STATS=3, Stat=Stat, lru_ensure=lambda *args, **kwargs: None)
    sys.modules.setdefault("flashlib", flashlib)
    sys.modules.setdefault("flashlib.kernels", kernels)
    sys.modules.setdefault("flashlib.kernels.slot_cache", slot_cache)
