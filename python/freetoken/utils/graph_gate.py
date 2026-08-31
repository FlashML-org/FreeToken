"""HIP/CUDA graph-capture parity probe.

Whether ``torch.cuda.graph`` graph capture works on the target GPU is the single
highest-informational-risk assumption for AMD (ROCm) support. This module probes it
once and records a PASS/FAIL + device result that the engine reads when deciding
whether to use CUDA-graph decode. On CUDA it is expected to PASS; on ROCm it may
fail on some consumer cards, in which case decode must use the kernel-launch path.

The result is cached to disk under the user cache dir so it survives across runs,
and keyed by device kind + device name so a change of GPU invalidates it.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

_CACHE_FILE = "freetoken_graph_gate.json"


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    path = os.path.join(base, "freetoken")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path() -> str:
    return os.path.join(_cache_dir(), _CACHE_FILE)


def _device_kind() -> str:
    from freetoken.utils.arch import device_kind

    return device_kind()


@lru_cache(maxsize=1)
def _device_name() -> str | None:
    """Best-effort current device name via torch.cuda, or None when no device / torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return None


def probe_graph_capture(variants: tuple[tuple[str, dict[str, str], str], ...] | None = None) -> dict:
    """Run the capture probes on the current device across the variants.

    Returns ``{"device_kind", "device", "ok", "detail", "variant", "env"}`` where
    ``variant`` is the first passing entry (or the last failing one) and ``env`` is
    the environment the engine worker must be spawned with for capture to work
    (empty for the default variant). See :data:`_GRAPH_VARIANTS`.

    Each variant runs in a **fresh subprocess**: on some ROCm builds a
    hipBLASLt/capture failure raises an uncatchable fatal HIP error (error 900) that
    aborts the whole process, so running it inline would crash the caller (and, worse,
    a decode path that attempted graph capture would die). A subprocess lets a fatal
    failure surface as a clean ``ok:false`` result instead.
    """
    device = _device_name()
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "device_kind": _device_kind(),
                "device": device,
                "ok": False,
                "detail": "no CUDA-capable device available",
            }
    except Exception as exc:
        detail = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
        return {
            "device_kind": _device_kind(),
            "device": device,
            "ok": False,
            "detail": f"torch unavailable: {type(exc).__name__}: {detail}",
        }

    # Each variant probes both an elementwise op (capturable on both backends) and a
    # GEMM (hipBLASLt on ROCm), which is what a real decode forward would run. The GEMM
    # is the discriminating case: on this ROCm build it fatally aborts -> child exit != 0.
    import subprocess as _subprocess
    import sys as _sys

    variants = variants if variants is not None else _GRAPH_VARIANTS
    last_fail: dict | None = None
    for name, extra_env, mode in variants:
        env = {**os.environ, **extra_env}
        child = _subprocess.run(
            [_sys.executable, "-c", _CAPTURE_CHILD, mode],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if child.returncode != 0:
            detail = next(
                (l.strip() for l in child.stderr.splitlines() if l.strip()),
                f"graph capture subprocess aborted (rc={child.returncode})",
            )
            last_fail = {
                "device_kind": _device_kind(),
                "device": device,
                "ok": False,
                "detail": f"fatal during capture: {detail[:240]}",
            }
            continue
        try:
            data = json.loads(child.stdout)
        except Exception:
            last_fail = {
                "device_kind": _device_kind(),
                "device": device,
                "ok": False,
                "detail": f"unparseable probe output: {child.stdout[:120]}",
            }
            continue
        data.setdefault("device_kind", _device_kind())
        data.setdefault("device", device)
        data["variant"] = name
        data["env"] = dict(extra_env)
        if data.get("ok"):
            return data
        last_fail = data
    assert last_fail is not None
    return last_fail


@lru_cache(maxsize=1)
def graph_capture_env() -> dict[str, str]:
    """Env vars the engine worker must be spawned with for graph capture to work.

    Empty when the default variant passes (or the gate failed / no device); otherwise
    the winning variant's extra env (e.g. ``TORCH_BLAS_PREFER_HIPBLASLT=0``). Applied
    by the server supervisor BEFORE the engine worker is spawned — never late at
    capture time, where a late env write may no-op.
    """
    try:
        result = run_graph_gate()
        env = result.get("env") or {}
        return env if result.get("ok") else {}
    except Exception:
        return {}


#: Child body for the graph-capture probe (see :func:`probe_graph_capture`). Prints a
#: JSON line ``{"ok": true/false, "detail": ...}`` and exits nonzero on a fatal abort.
#: Modes: "plain" = capture a GEMM with default BLAS/library state; "prewarm" = run the
#: same-shaped GEMM once before capture (hipBLASLt lazily allocates its workspace on
#: the first GEMM, and a capture that includes that allocation is what aborts on some
#: ROCm builds).
_CAPTURE_CHILD = r"""
import json, sys
mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
import torch
try:
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    x = torch.randn(8, 8, device='cuda')
    if mode == "prewarm":
        a0 = torch.randn(64, 64, device='cuda')
        torch.mm(a0, a0)  # pre-allocate the BLAS workspace OUTSIDE any capture
    # elementwise (capturable) first, then a GEMM (hipBLASLt on ROCm)
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.add(x, x)
        s.synchronize()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        a = torch.randn(64, 64, device='cuda')
        with torch.cuda.graph(g):
            torch.mm(a, a)
        s.synchronize()
    torch.cuda.synchronize()
    print(json.dumps({"ok": True, "detail": f"{mode}: elementwise+GEMM capture/replay succeeded"}))
except Exception as e:
    print(json.dumps({"ok": False, "detail": f"{type(e).__name__}: {str(e)[:160]}"}))
"""


# Capture-probe variants, tried in order (Inc 6, .plans/rocm-perf-parity). Each entry
# is (name, extra_env, child_mode). "rocblas" forces the rocBLAS GEMM path via
# TORCH_BLAS_PREFER_HIPBLASLT=0 — hipBLASLt is the capture abort's usual suspect on
# gfx1100. "prewarm" pre-runs the GEMM outside capture (workspace pre-alloc). The
# winning variant's env must be applied to the engine worker process at SPAWN time
# (server/launch.py) — never late at capture time, where a late env write may no-op.
_GRAPH_VARIANTS: tuple[tuple[str, dict[str, str], str], ...] = (
    ("default", {}, "plain"),
    ("rocblas", {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}, "plain"),
    ("prewarm", {}, "prewarm"),
    ("rocblas_prewarm", {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}, "prewarm"),
)


def _load_cached() -> dict | None:
    try:
        with open(_cache_path()) as f:
            data = json.load(f)
        if (
            data.get("device_kind") == _device_kind()
            and data.get("device") == _device_name()
            # v2 cache format: carries the winning "variant"/"env". An old all-FAIL
            # record (no variant) must not mask a possible variant PASS after a
            # ROCm/driver upgrade.
            and "variant" in data
        ):
            return data
    except Exception:
        pass
    return None


def run_graph_gate() -> dict:
    """Run (or reuse the cached result for the current device of) the capture probe."""
    cached = _load_cached()
    if cached is not None:
        return cached
    result = probe_graph_capture()
    try:
        with open(_cache_path(), "w") as f:
            json.dump(result, f)
    except Exception:
        pass
    return result


@lru_cache(maxsize=1)
def graph_capture_status() -> str:
    """Cached graph-capture status: ``"pass"``, ``"fail"``, or ``"unknown"`` (no device /
    probe unavailable). The graph runner reads this to pick HIP-graph vs kernel-launch decode."""
    try:
        result = run_graph_gate()
        if result["ok"]:
            return "pass"
        if result.get("device_kind") and result["device_kind"] != "cpu":
            return "fail"
        return "unknown"
    except Exception:
        return "unknown"
