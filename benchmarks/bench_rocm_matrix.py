"""Machine-readable ROCm benchmark identity and timing helpers.

This module does not decide whether an optimization is good. It makes missing identity,
route, completion, and timing evidence impossible to mistake for a passing run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA = "freetoken-rocm-manifest-v1"
LANES = frozenset({"sampled_absolute", "greedy_correctness", "teacher_forced_replay"})
_HEX64 = set("0123456789abcdefABCDEF")


def sha256_path(path: str | os.PathLike[str]) -> str:
    """Hash file bytes or sorted directory-relative file names and bytes; ignore mtimes."""
    root = Path(path).expanduser()
    if root.is_file():
        digest = hashlib.sha256()
        with root.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for child in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = child.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _full_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def git_identity(repo: str | os.PathLike[str] = ".") -> dict[str, str | bool]:
    """Capture commit and diff identity without mutating the checkout."""
    root = str(repo)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"], cwd=root, stderr=subprocess.STDOUT
        )
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot capture git identity: {exc}") from exc
    return {
        "commit": commit,
        "dirty_diff": sha256_bytes(diff),
        "dirty": dirty,
    }


def runtime_identity() -> dict[str, Any]:
    """Capture runtime identity; unavailable optional probes are explicit strings."""
    import torch

    try:
        triton = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton = "unavailable"
    gpu = "none"
    driver = "unavailable"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        try:
            driver = str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName)
        except (AttributeError, RuntimeError):
            driver = "unavailable"
    env = {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith(("FREETOKEN_", "PYTORCH_ROCM_ARCH", "TORCH_EXTENSIONS_DIR"))
    }
    return {
        **git_identity(),
        "gpu": gpu,
        "driver": driver,
        "torch": str(torch.__version__),
        "rocm": str(torch.version.hip or "none"),
        "hip": str(torch.version.hip or "none"),
        "triton": triton,
        "jit_sha": os.environ.get("FREETOKEN_GGUF_SOURCE_VERSION", "unknown"),
        "env_digest": sha256_text(json.dumps(env, sort_keys=True)),
    }


def summarize_timings(values: Iterable[float], *, warmup: int = 0) -> dict[str, Any]:
    raw = [float(value) for value in values]
    measured = raw[max(0, int(warmup)) :]
    if not measured or any(not math.isfinite(value) or value <= 0 for value in raw):
        raise ValueError("timing values must be finite, positive, and non-empty")
    return {
        "warmup": max(0, int(warmup)),
        "raw_tok_s": raw,
        "median_tok_s": statistics.median(measured),
        "spread": max(measured) - min(measured),
        "repeats": len(measured),
    }


def build_manifest(
    *,
    model: str | os.PathLike[str],
    prompt: str,
    token_count: int,
    mtp: str = "off",
    flags: dict[str, Any] | None = None,
    backend: str,
    quant: str,
    graph_mode: str,
    route: str | dict[str, Any],
    cache_hits: int,
    fetches: int,
    fallbacks: int,
    finite_logits: bool,
    completion_count: int,
    lane: str,
    timings_tok_s: Iterable[float],
    warmup: int = 0,
    runtime: dict[str, Any] | None = None,
    repo: str | os.PathLike[str] = ".",
) -> dict[str, Any]:
    model_value = str(model)
    model_sha = model_value if _full_sha(model_value) else sha256_path(model_value)
    if runtime is None:
        captured_runtime = runtime_identity()
        if str(repo) != ".":
            captured_runtime.update(git_identity(repo))
    else:
        captured_runtime = dict(runtime)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "accepted",
        "workload": {
            "model_sha256": model_sha,
            "prompt_sha256": sha256_text(prompt),
            "token_count": int(token_count),
            "mtp": mtp,
            "flags": dict(flags or {}),
        },
        "runtime": captured_runtime,
        "observed": {
            "backend": backend,
            "quant": quant,
            "graph_mode": graph_mode,
            "route": route,
            "cache_hits": int(cache_hits),
            "fetches": int(fetches),
            "fallbacks": int(fallbacks),
            "finite_logits": bool(finite_logits),
            "completion_count": int(completion_count),
        },
        "timing": {
            "lane": lane,
            **summarize_timings(timings_tok_s, warmup=warmup),
        },
    }
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("invalid benchmark manifest: " + "; ".join(problems))
    return manifest


def validate_manifest(value: object) -> list[str]:
    """Return rejection reasons; empty means complete provenance/acceptance evidence."""
    if not isinstance(value, dict):
        return ["manifest is not an object"]
    problems: list[str] = []
    if value.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema must be {MANIFEST_SCHEMA!r}")
    workload = value.get("workload")
    runtime = value.get("runtime")
    observed = value.get("observed")
    timing = value.get("timing")
    if not isinstance(workload, dict):
        problems.append("workload is missing")
    else:
        for key in ("model_sha256", "prompt_sha256"):
            if not _full_sha(workload.get(key)):
                problems.append(f"workload.{key} must be a full SHA-256")
        if not isinstance(workload.get("token_count"), int) or workload["token_count"] < 1:
            problems.append("workload.token_count must be positive")
        if workload.get("mtp") not in ("off", False, 0):
            problems.append("workload.mtp must be off")
        if not isinstance(workload.get("flags"), dict):
            problems.append("workload.flags must be an object")
    if not isinstance(runtime, dict):
        problems.append("runtime is missing")
    else:
        for key in ("commit", "dirty_diff", "gpu", "driver", "torch", "rocm", "hip", "triton", "jit_sha", "env_digest"):
            if not isinstance(runtime.get(key), (str, bool)) or runtime.get(key) in (None, ""):
                problems.append(f"runtime.{key} is missing")
    if not isinstance(observed, dict):
        problems.append("observed is missing")
    else:
        for key in ("backend", "quant", "graph_mode", "route"):
            if not observed.get(key):
                problems.append(f"observed.{key} is missing")
        if observed.get("graph_mode") not in {"eager", "replay", "disabled"}:
            problems.append("observed.graph_mode must be eager, replay, or disabled")
        for key in ("cache_hits", "fetches", "fallbacks", "completion_count"):
            if not isinstance(observed.get(key), int) or observed[key] < 0:
                problems.append(f"observed.{key} must be a non-negative integer")
        if observed.get("fallbacks", 0) > 0:
            problems.append("observed.fallbacks must be zero for promotion evidence")
        if observed.get("finite_logits") is not True:
            problems.append("observed.finite_logits must be true")
        if isinstance(workload, dict) and observed.get("completion_count") != workload.get("token_count"):
            problems.append("observed.completion_count must equal workload.token_count")
    if not isinstance(timing, dict):
        problems.append("timing is missing")
    else:
        if timing.get("lane") not in LANES:
            problems.append("timing.lane is unknown")
        if not isinstance(timing.get("repeats"), int) or timing["repeats"] < 1:
            problems.append("timing.repeats must be positive")
        if (
            not isinstance(timing.get("median_tok_s"), (int, float))
            or not math.isfinite(timing["median_tok_s"])
            or timing["median_tok_s"] <= 0
        ):
            problems.append("timing.median_tok_s must be positive")
        if (
            not isinstance(timing.get("spread"), (int, float))
            or not math.isfinite(timing["spread"])
            or timing["spread"] < 0
        ):
            problems.append("timing.spread must be non-negative")
        raw_timings = timing.get("raw_tok_s")
        if raw_timings is not None and (
            not isinstance(raw_timings, list)
            or not raw_timings
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in raw_timings
            )
        ):
            problems.append("timing.raw_tok_s must contain finite positive values")
    return problems


def write_manifest(path: str | os.PathLike[str], manifest: dict[str, Any]) -> None:
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("manifest rejected: " + "; ".join(problems))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, sort_keys=True)
    if target.suffix == ".jsonl":
        with target.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifests(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        values = value if isinstance(value, list) else [value]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{target}: every manifest must be an object")
    return values
