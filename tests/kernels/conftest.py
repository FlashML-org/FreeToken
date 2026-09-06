"""Allow kernel contract tests to run without optional CUDA-only flashlib."""

from enum import IntEnum
import sys
from types import SimpleNamespace


if "flashlib.kernels.slot_cache" not in sys.modules:
    class Stat(IntEnum):
        ACTIVE = 0
        MISS = 1
        CALLS = 2

    slot_cache = SimpleNamespace(
        N_STATS=3,
        Stat=Stat,
        lru_ensure=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("flashlib", SimpleNamespace())
    sys.modules.setdefault("flashlib.kernels", SimpleNamespace())
    sys.modules.setdefault("flashlib.kernels.slot_cache", slot_cache)
