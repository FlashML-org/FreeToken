"""Torch-free reader for the ``ft bench bw`` hardware profile (``benchbw/<gpu-uuid>.json``).

The engine consults this at MoE-backend *auto* resolution (``engine.py``) to make the
offload-vs-hybrid choice hardware-adaptive without importing the (torch-heavy) benchmark
itself. ``benchbw.py`` writes the profile; this module only reads it.

The join key is the expert *format*: the CPU-MoE-vs-PCIe-gather bandwidth ratio the choice
rides on is dominated by ``(format, hardware)``, not by the exact model, so a profile benched
on one workload transfers to any model with the same expert format on the same GPU.
"""

from __future__ import annotations

import json
import os

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Engine ``expert_quant`` (models/config.py) -> benchbw format key (offload_cache._BANK_SCHEMAS
# / benchbw._offload_bank_specs). Only the offload-family formats with a CPU MoE weight path can
# ever resolve to hybrid; anything not listed falls through unmapped and finds no profile entry
# (-> None -> offload), which is the safe default.
_QUANT_TO_BENCH_FORMAT = {
    "nvfp4": "nvfp4",
    "ds_fp4": "ds_fp4",
    "mxfp4": "mxfp4_triton",
    "bf16": "bf16",
    "fp8_block": "fp8_block",
}


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
    prof_gpu = (prof.get("gpu") or {}).get("name")
    if gpu_name and prof_gpu and prof_gpu != gpu_name:
        logger.warning(
            f"benchbw profile {src} was measured on {prof_gpu!r}, not this GPU "
            f"({gpu_name!r}); ignoring it"
        )
        return None
    return prof


def load_backend_recommendation(
    quant_format: str,
    gpu_name: str | None = None,
    path: str | None = None,
    gpu_uuid: str | None = None,
) -> str | None:
    """Bench-recommended offload-family backend for ``quant_format`` on this GPU, or ``None``.

    Returns ``"hybrid"`` only when *every* benched workload sharing this expert format
    recommended hybrid (CPU MoE BW > threshold x PCIe gather BW); a mixed verdict (a
    near-threshold format) resolves conservatively to ``"offload"``. ``None`` means "no usable
    profile" (see ``_usable_profile``) or no entry for this format. The caller keeps its own
    default (offload) on ``None``.
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof = _usable_profile(gpu_name, path, gpu_uuid)
    if prof is None:
        return None

    # Preferred: the per-dtype tuning verdicts (`ft bench bw --dtype`), a direct format->backend
    # map -- the axis the backend pick is meant to key on.
    dtypes = prof.get("dtypes")
    if isinstance(dtypes, dict) and dtypes.get(fmt) in ("hybrid", "offload"):
        return dtypes[fmt]

    # Fallback: a per-model profile (`ft bench bw --model`). Aggregate the workloads sharing this
    # format -- unanimous hybrid -> hybrid; any offload (a near-threshold split) -> offload.
    workloads = prof.get("workloads")
    if not isinstance(workloads, dict):
        return None
    picks = [
        entry["recommended"]
        for wl in workloads.values()
        if isinstance(wl, dict)
        for entry in [(wl.get("kernels") or {}).get(fmt)]
        if isinstance(entry, dict) and entry.get("recommended")
    ]
    if not picks:
        return None
    return "hybrid" if all(p == "hybrid" for p in picks) else "offload"


def load_hybrid_fetch_fraction(
    quant_format: str,
    gpu_name: str | None = None,
    path: str | None = None,
    gpu_uuid: str | None = None,
) -> float | None:
    """Benched hybrid fetch fraction for ``quant_format``, or ``None``.

    The hybrid backend's bandwidth-matched fetch split: of a decode step's expert misses,
    fetch this fraction over PCIe and compute the rest on the CPU, so both finish together.
    Preferred source is the *overlapped* pair (CPU MoE and PCIe gather measured while
    running concurrently -- the real contention regime): fetched/misses = pcie_ov /
    (pcie_ov + cpu_ov). Older profiles without it fall back to the standalone bandwidths
    under a full-DRAM-contention assumption (cpu keeps cpu - pcie under DMA), which
    reduces to pcie/cpu. Per-dtype entry first, then any per-model entry with this format.
    ``None`` = no usable profile; clamped to [0, 1].
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof = _usable_profile(gpu_name, path, gpu_uuid)
    if prof is None:
        return None
    entries = [(prof.get("dtype_kernels") or {}).get(fmt)] + [
        (wl.get("kernels") or {}).get(fmt)
        for wl in (prof.get("workloads") or {}).values()
        if isinstance(wl, dict)
    ]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
        if cpu_ov and pcie_ov:
            return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
        cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
        if cpu and pcie:
            return min(1.0, pcie / cpu)
    return None
