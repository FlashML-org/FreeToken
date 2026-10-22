"""Torch-free reader for the ``ft bench bw`` hardware profile (``benchbw/<gpu-uuid>.json``).

The profile holds one row per MoE layer kind (``layer_kind_key``): the CPU MoE and PCIe gather
bandwidths benched for that kind on this GPU and host. The engine reads it to set each hybrid
layer's fetch split, without importing the (torch-heavy) benchmark. ``benchbw.py`` writes it.
"""

from __future__ import annotations

import json
import os

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Bump when the row key or the measurements change: rows of another version are not comparable.
PROFILE_VERSION = 6


def _cache_dir() -> str:
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "freetoken")


def default_profile_path(gpu_uuid: str | None = None) -> str:
    """``$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json``, or the legacy ``benchbw.json`` without a uuid.

    One file per GPU: bandwidth differs between slots.
    """
    if gpu_uuid:
        return os.path.join(_cache_dir(), "benchbw", f"{gpu_uuid}.json")
    return os.path.join(_cache_dir(), "benchbw.json")


def latest_profile_path() -> str | None:
    """Newest ``benchbw/*.json``, else the legacy ``benchbw.json``, else None."""
    per_gpu = os.path.join(_cache_dir(), "benchbw")
    newest: tuple[float, str] | None = None
    try:
        for name in os.listdir(per_gpu):
            if not name.endswith(".json"):
                continue
            path = os.path.join(per_gpu, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if newest is None or mtime > newest[0]:
                newest = (mtime, path)
    except OSError:
        pass
    if newest is not None:
        return newest[1]
    legacy = default_profile_path()
    return legacy if os.path.isfile(legacy) else None


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _usable_profile(
    gpu_name: str | None, path: str | None, gpu_uuid: str | None = None
) -> dict | None:
    """The cached profile, or ``None`` when there is no file / it was benched on another GPU
    (bandwidths are hardware-specific, so a mismatch is ignored rather than trusted).

    Lookup: explicit ``path`` (else ``FREETOKEN_BENCHBW_PATH``) -> ``benchbw/<gpu_uuid>.json`` -> legacy ``benchbw.json``.
    """
    explicit = path or os.environ.get("FREETOKEN_BENCHBW_PATH")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [default_profile_path(gpu_uuid)] if gpu_uuid else []
        candidates.append(default_profile_path())
    prof = None
    for src in candidates:
        prof = _load(src)
        if isinstance(prof, dict):
            break
        if os.path.exists(src):
            # unreadable profile for this card: stay on the safe default, do not borrow the legacy file
            return None
    if not isinstance(prof, dict):
        return None
    if prof.get("version") != PROFILE_VERSION:
        logger.warning(
            f"benchbw profile {src} has version {prof.get('version')}, this engine reads "
            f"{PROFILE_VERSION}; ignoring it, rerun `ft bench bw`"
        )
        return None
    prof_gpu = (prof.get("gpu") or {}).get("name")
    if gpu_name and prof_gpu and prof_gpu != gpu_name:
        logger.warning(
            f"benchbw profile {src} was measured on {prof_gpu!r}, not this GPU "
            f"({gpu_name!r}); ignoring it"
        )
        return None
    return prof


def layer_kind_key(fmt: str, options: dict[str, int], hidden: int, inter: int, top_k: int, cpu_threads: int) -> str:
    """Row key of one MoE layer kind, e.g. ``gguf[down_type=8,gate_up_type=8]:2816x704:top8:t12``.

    ``cpu_threads`` is the CPU executor's worker count: the per-step pool overhead makes the CPU bandwidth depend on it and on ``top_k``."""
    opts = ",".join(f"{k}={v}" for k, v in sorted(options.items()))
    fmt = f"{fmt}[{opts}]" if opts else fmt
    return f"{fmt}:{hidden}x{inter}:top{top_k}:t{cpu_threads}"


def fetch_fraction(row: dict | None) -> float | None:
    """The hybrid fetch split of a row: of a decode step's expert misses, the fraction to fetch over PCIe so the fetch and the CPU compute finish together.

    From the overlapped pair (both sides measured while running concurrently): pcie_ov / (pcie_ov + cpu_ov); without it, pcie / cpu. Clamped to 1."""
    if not isinstance(row, dict):
        return None
    cpu_ov, pcie_ov = row.get("cpu_moe_overlap_gbs"), row.get("pcie_gather_overlap_gbs")
    if cpu_ov and pcie_ov:
        return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
    cpu, pcie = row.get("cpu_moe_gbs"), row.get("pcie_gather_gbs")
    if cpu and pcie:
        return min(1.0, pcie / cpu)
    return None


def locked_fetch_fraction(row: dict | None) -> float | None:
    """The fetch split of a locked layer of this row's kind: p / (2p + c) from the same pair as ``fetch_fraction``.

    A locked fetch is a CPU memcpy into staging, then the PCIe gather; taking the memcpy rate as the CPU MoE rate c, the two steps in series run at p*c / (p + c), which against the CPU compute gives p / (2p + c)."""
    if not isinstance(row, dict):
        return None
    for cpu_key, pcie_key in (("cpu_moe_overlap_gbs", "pcie_gather_overlap_gbs"), ("cpu_moe_gbs", "pcie_gather_gbs")):
        cpu, pcie = row.get(cpu_key), row.get(pcie_key)
        if cpu and pcie:
            return pcie / (2 * pcie + cpu)
    return None


def load_layer_kinds(gpu_name: str | None = None, path: str | None = None, gpu_uuid: str | None = None) -> dict | None:
    """The rows of the usable profile for this GPU (key -> row), or None without one."""
    prof = _usable_profile(gpu_name, path, gpu_uuid)
    if prof is None:
        return None
    rows = prof.get("layer_kinds")
    return rows if isinstance(rows, dict) else {}
