"""Torch-free reader for the ``ft bench bw`` hardware profile (``benchbw/<gpu-uuid>.json``).

The engine consults this at MoE-backend *auto* resolution (``engine.py``) to make the
offload-vs-hybrid choice hardware-adaptive without importing the (torch-heavy) benchmark
itself. ``benchbw.py`` writes the profile; this module only reads it.

The join key is the expert *format*: the CPU-MoE-vs-PCIe-gather bandwidth ratio the choice
rides on is dominated by ``(format, hardware)``, so a profile benched on one workload transfers
to other models with the same expert format on the same GPU -- as long as their experts are of
comparable size. The CPU side has a per-expert fixed cost (thread fan-out over a short GEMV),
so a verdict measured on much larger experts overstates it for a small-expert model: Qwen3.6-
35B-A3B-NVFP4 (1.7 MB experts) benches at 32 GB/s where the nvfp4 dtype geometry (8.0 MB)
gives 90 GB/s on the same box, and hybrid then decodes 3x slower than offload. So a per-model
bench of the served geometry wins, and a dtype verdict is only applied when its expert size is
within :data:`EXPERT_BYTES_TOLERANCE` of the model's.
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


# A dtype verdict / fetch fraction transfers to a model whose experts are within this factor
# of the benched ones (either way). Beyond it the entry is skipped and the caller falls back to
# offload; `ft bench bw --model` for the model's geometry is the remedy.
EXPERT_BYTES_TOLERANCE = 2.0
_GEOMETRY_KEYS = ("hidden", "inter", "experts", "top_k")
_warned: set = set()


def _geometry_matches(entry_model, geometry) -> bool:
    if not isinstance(entry_model, dict) or not geometry:
        return False
    return all(entry_model.get(k) == geometry.get(k) for k in _GEOMETRY_KEYS)


def _model_entry(prof: dict, fmt: str, geometry) -> dict | None:
    """``kernels[fmt]`` of the per-model workload benched at exactly ``geometry``, else None."""
    for wl in (prof.get("workloads") or {}).values():
        if isinstance(wl, dict) and _geometry_matches(wl.get("model"), geometry):
            entry = (wl.get("kernels") or {}).get(fmt)
            if isinstance(entry, dict):
                return entry
    return None


def _comparable(entry, fmt: str, expert_bytes) -> bool:
    """Whether ``entry`` (a dtype_kernels / workload kernels entry) was benched on experts of a
    size comparable to ``expert_bytes``. Unknown on either side -> assumed comparable (older
    profiles carry no ``expert_bytes``; callers that pass none keep the format-only join)."""
    benched = entry.get("expert_bytes") if isinstance(entry, dict) else None
    if not benched or not expert_bytes:
        return True
    ratio = max(benched, expert_bytes) / min(benched, expert_bytes)
    if ratio <= EXPERT_BYTES_TOLERANCE:
        return True
    key = (fmt, benched, expert_bytes)
    if key not in _warned:
        _warned.add(key)
        logger.warning(
            f"benchbw profile: the {fmt!r} entry was benched on {benched / 2**20:.2f} MB experts, "
            f"this model's are {expert_bytes / 2**20:.2f} MB ({ratio:.1f}x apart); not applying "
            f"its verdict. Run `ft bench bw --model <preset>` for this model's geometry "
            f"(see `ft bench bw --help`)"
        )
    return False


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
    *,
    expert_bytes: int | None = None,
    geometry: dict | None = None,
) -> str | None:
    """Bench-recommended offload-family backend for ``quant_format`` on this GPU, or ``None``.

    Returns ``"hybrid"`` only when *every* benched workload sharing this expert format
    recommended hybrid (CPU MoE BW > threshold x PCIe gather BW); a mixed verdict (a
    near-threshold format) resolves conservatively to ``"offload"``. ``None`` means "no usable
    profile" (see ``_usable_profile``) or no entry for this format. The caller keeps its own
    default (offload) on ``None``.

    ``geometry`` (``{hidden, inter, experts, top_k}``) selects a per-model bench of exactly the
    served model first; ``expert_bytes`` (the model's per-expert bank bytes) then gates every
    other entry through :func:`_comparable`.
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof = _usable_profile(gpu_name, path, gpu_uuid)
    if prof is None:
        return None

    # Best: this exact geometry was benched (`ft bench bw --model`).
    entry = _model_entry(prof, fmt, geometry)
    if entry is not None and entry.get("recommended") in ("hybrid", "offload"):
        return entry["recommended"]

    # Next: the per-dtype tuning verdicts (`ft bench bw --dtype`), a direct format->backend
    # map -- the axis the backend pick keys on -- when benched on comparable experts.
    dtypes = prof.get("dtypes")
    if isinstance(dtypes, dict) and dtypes.get(fmt) in ("hybrid", "offload"):
        if _comparable((prof.get("dtype_kernels") or {}).get(fmt), fmt, expert_bytes):
            return dtypes[fmt]

    # Fallback: other per-model workloads sharing this format, comparable experts only --
    # unanimous hybrid -> hybrid; any offload (a near-threshold split) -> offload.
    workloads = prof.get("workloads")
    if not isinstance(workloads, dict):
        return None
    picks = [
        entry["recommended"]
        for wl in workloads.values()
        if isinstance(wl, dict)
        for entry in [(wl.get("kernels") or {}).get(fmt)]
        if isinstance(entry, dict) and entry.get("recommended")
        and _comparable(entry, fmt, expert_bytes)
    ]
    if not picks:
        return None
    return "hybrid" if all(p == "hybrid" for p in picks) else "offload"


def load_hybrid_fetch_fraction(
    quant_format: str,
    gpu_name: str | None = None,
    path: str | None = None,
    gpu_uuid: str | None = None,
    *,
    expert_bytes: int | None = None,
    geometry: dict | None = None,
) -> float | None:
    """Benched hybrid fetch fraction for ``quant_format``, or ``None``.

    The hybrid backend's bandwidth-matched fetch split: of a decode step's expert misses,
    fetch this fraction over PCIe and compute the rest on the CPU, so both finish together.
    Preferred source is the *overlapped* pair (CPU MoE and PCIe gather measured while
    running concurrently -- the real contention regime): fetched/misses = pcie_ov /
    (pcie_ov + cpu_ov). Older profiles without it fall back to the standalone bandwidths
    under a full-DRAM-contention assumption (cpu keeps cpu - pcie under DMA), which
    reduces to pcie/cpu. A per-model entry of the served ``geometry`` first, then the
    per-dtype entry, then any per-model entry with this format -- each only when benched on
    experts comparable to ``expert_bytes``. ``None`` = no usable profile; clamped to [0, 1].
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof = _usable_profile(gpu_name, path, gpu_uuid)
    if prof is None:
        return None
    entries = [_model_entry(prof, fmt, geometry), (prof.get("dtype_kernels") or {}).get(fmt)] + [
        (wl.get("kernels") or {}).get(fmt)
        for wl in (prof.get("workloads") or {}).values()
        if isinstance(wl, dict)
    ]
    for entry in entries:
        if not isinstance(entry, dict) or not _comparable(entry, fmt, expert_bytes):
            continue
        cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
        if cpu_ov and pcie_ov:
            return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
        cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
        if cpu and pcie:
            return min(1.0, pcie / cpu)
    return None
