"""ggml-cpu row kernels for GGUF experts on the CPU MoE executor.

setup.py builds one module per x86 variant of llama.cpp's GGML_CPU_ALL_VARIANTS; select() picks the one this CPU runs best.
"""

from __future__ import annotations

import functools
import importlib
import os
import pkgutil
from typing import NamedTuple


class Variant(NamedTuple):
    name: str
    api: int  # address of the variant's kernel table, for _cpu_moe.set_ggml_cpu


@functools.cache
def select() -> Variant | None:
    """The highest-scoring variant this CPU runs (upstream's rule), or the one FREETOKEN_GGML_CPU_VARIANT names; None when the build has none."""
    names = sorted(m.name[1:] for m in pkgutil.iter_modules(__path__) if m.name.startswith("_"))
    forced = os.environ.get("FREETOKEN_GGML_CPU_VARIANT")
    if forced:
        if forced not in names:
            raise RuntimeError(f"FREETOKEN_GGML_CPU_VARIANT={forced}: this build has the ggml-cpu variants {names}")
        names = [forced]
    best, best_score = None, 0
    for name in names:
        # importing a variant runs no code built for its ISA extensions
        mod = importlib.import_module(f"{__name__}._{name}")
        if (score := mod.score()) > best_score:
            best, best_score = Variant(name, mod.api()), score
    if forced and best is None:
        raise RuntimeError(f"FREETOKEN_GGML_CPU_VARIANT={forced}: this CPU lacks an ISA extension the variant is built for")
    return best
