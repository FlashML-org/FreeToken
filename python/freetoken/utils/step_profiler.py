"""Env-gated per-step profiler and optional ROCm marker wrapper.

``FREETOKEN_TORCH_PROFILE="<warm>:<steps>:<out>"``, unset/empty = no-op (the only
hot-path cost is one cached check). Skip the first ``warm`` call(s) of the wrapped
region, then profile the next ``steps`` calls in one ``torch.profiler`` window and
export on the window's exit:

- a chrome trace to ``<out>`` and
- a top-kernels table to ``<out minus extension>-kernels.log``.

``FREETOKEN_ROCTX_MARKERS=1`` emits low-overhead ROCTX ranges without torch
profiling. A non-boolean value also retains marker records in the profiler's
``-markers.json`` sidecar.

This is the Inc 1 instrument of .plans/rocm-parity-next: NVTX is a no-op on ROCm,
so the trace segments by the explicit ``torch.profiler.record_function`` range
names (moe_router, moe_gate_up, moe_down, attn, Sampler) that appear as table row
names on both backends.
"""

from __future__ import annotations

import atexit
import os
import ctypes
import ctypes.util
import json
import statistics
import time

_NO_SPEC = object()  # "not parsed yet" sentinel


def parse_spec(raw: str) -> tuple[int, int, str] | None:
    """Parse '<warm>:<steps>:<out>' -> (warm, steps, out); None when empty."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        warm_s, steps_s, out = raw.split(":")
        warm, steps = int(warm_s), int(steps_s)
        if warm < 0 or steps <= 0 or not out:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "FREETOKEN_TORCH_PROFILE must be '<warm>:<steps>:<out.json>' "
            f"(warm>=0, steps>0), got {raw!r}"
        ) from exc
    return warm, steps, out


def _read_spec() -> tuple[int, int, str] | None:
    return parse_spec(os.environ.get("FREETOKEN_TORCH_PROFILE", ""))


class _State:
    __slots__ = (
        "spec",
        "calls",
        "done",
        "prof",
        "markers_enabled",
        "marker_output",
        "markers",
        "roctx",
        "roctx_error",
    )

    def __init__(self) -> None:
        self.spec: tuple[int, int, str] | None = _NO_SPEC  # type: ignore[assignment]
        self.calls = 0
        self.done = False
        self.prof = None
        self.markers_enabled: bool | None = None
        self.marker_output: str | None = None
        self.markers: list[dict] = []
        self.roctx = None
        self.roctx_error = None


_state = _State()


class _NullCtx:
    """No-op context manager (profiler disabled or outside the profiled window)."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ProfilerCtx:
    """One wrapped step inside the profiled window; exports on the last step's exit."""

    __slots__ = ("prof", "final", "step", "phase", "stream_id")

    def __init__(self, prof, final: bool, step: int, phase: str, stream_id: str | None) -> None:
        self.prof = prof
        self.final = final
        self.step = step
        self.phase = phase
        self.stream_id = stream_id
        if final:
            _state.done = True

    def __enter__(self):
        _mark(self.step, self.phase, "begin", self.stream_id)
        return self

    def __exit__(self, exc_type=None, exc=None, tb=None) -> bool:  # noqa: ANN001
        _mark(self.step, self.phase, "end", self.stream_id)
        if self.final:
            if self.prof is not None:
                self.prof.__exit__(exc_type, exc, tb)
            warm, steps, out = _state.spec or (0, 0, "")
            if self.prof is not None:
                _export(self.prof, out, steps)
        return False


class _PhaseCtx:
    """Low-overhead phase range nested inside a scheduler step."""

    __slots__ = ("phase", "step", "stream_id", "record")

    def __init__(self, phase: str, step: int, stream_id: str | None) -> None:
        self.phase = phase
        self.step = step
        self.stream_id = stream_id
        self.record = None

    def __enter__(self):
        _mark(self.step, self.phase, "begin", self.stream_id)
        # record_function is useful in torch traces even when ROCTX is unavailable. Import
        # lazily so disabled profiling has no torch dependency in this helper.
        try:
            import torch.profiler

            self.record = torch.profiler.record_function(self.phase)
            self.record.__enter__()
        except Exception:
            self.record = None
        return self

    def __exit__(self, exc_type=None, exc=None, tb=None) -> bool:  # noqa: ANN001
        if self.record is not None:
            self.record.__exit__(exc_type, exc, tb)
        _mark(self.step, self.phase, "end", self.stream_id)
        return False


_NULL = _NullCtx()


def _configure_markers() -> None:
    raw = os.environ.get("FREETOKEN_ROCTX_MARKERS", "").strip()
    _state.markers_enabled = bool(raw and raw.lower() not in {"0", "false", "off", "no"})
    if _state.markers_enabled and raw.lower() not in {"1", "true", "yes", "on"}:
        _state.marker_output = raw
    if not _state.markers_enabled:
        return
    if _state.marker_output:
        # Launch-mode rocprof has no torch-profiler exit hook. Persist sidecar
        # ranges at process exit so GPU trace and CPU phase ledger share a key.
        atexit.register(_export_marker_sidecar)
    library = ctypes.util.find_library("roctx64") or "libroctx64.so"
    try:
        _state.roctx = ctypes.CDLL(library)
        _state.roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
        _state.roctx.roctxRangePushA.restype = ctypes.c_int
        _state.roctx.roctxRangePop.argtypes = []
        _state.roctx.roctxRangePop.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        _state.roctx = None
        _state.roctx_error = f"{type(exc).__name__}: {exc}"


def _current_stream_id() -> str | None:
    try:
        import torch

        stream = torch.cuda.current_stream()
        value = getattr(stream, "cuda_stream", None)
        return str(value) if value is not None else None
    except Exception:
        return None


def _mark(step: int, phase: str, event: str, stream_id: str | None) -> None:
    if not _state.markers_enabled:
        return
    monotonic_ns = time.monotonic_ns()
    marker = {
        "step": step,
        "phase": phase,
        "event": event,
        "monotonic_ns": monotonic_ns,
        "stream_id": stream_id,
    }
    if _state.marker_output:
        _state.markers.append(marker)
    if _state.roctx is not None:
        label = f"freetoken step={step} phase={phase}"
        try:
            if event == "begin":
                _state.roctx.roctxRangePushA(label.encode())
            else:
                _state.roctx.roctxRangePop()
        except Exception as exc:  # marker failure must never stop serving
            _state.roctx_error = f"{type(exc).__name__}: {exc}"


def step_profiler(phase: str = "scheduler", stream_id: str | None = None):
    """Wrap one scheduler step. No-op unless FREETOKEN_TORCH_PROFILE is set (parsed
    on the first call); the window covers `steps` calls after the first `warm`."""
    if _state.markers_enabled is None:
        _configure_markers()
    if _state.done:
        return _NULL
    if _state.spec is _NO_SPEC:  # type: ignore[comparison-overlap]
        spec = _read_spec()
        if spec is None:
            if not _state.markers_enabled:
                _state.done = True
                return _NULL
        _state.spec = spec
    _state.calls += 1
    marker_step = _state.calls
    marker_stream = stream_id if stream_id is not None else _current_stream_id()
    if _state.spec is None:
        return _ProfilerCtx(None, final=False, step=marker_step, phase=phase, stream_id=marker_stream)
    warm, steps, _ = _state.spec
    if _state.calls <= warm:
        if _state.markers_enabled:
            return _ProfilerCtx(None, final=False, step=marker_step, phase=phase, stream_id=marker_stream)
        return _NULL

    import torch.profiler

    if _state.prof is None:
        _state.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
            acc_events=True,
        )
        _state.prof.__enter__()
    return _ProfilerCtx(
        _state.prof,
        final=_state.calls >= warm + steps,
        step=marker_step,
        phase=phase,
        stream_id=marker_stream,
    )


def profiler_phase(phase: str, stream_id: str | None = None):
    """Mark graph/sampler subranges without making missing profiler libs fatal."""
    if _state.markers_enabled is None:
        _configure_markers()
    if not _state.markers_enabled and _state.spec is _NO_SPEC:
        # Keep record_function available for an explicitly active torch profiler, but avoid
        # importing torch on the normal serving path when no instrumentation is configured.
        if not os.environ.get("FREETOKEN_TORCH_PROFILE"):
            return _NULL
    marker_step = _state.calls or 0
    marker_stream = stream_id if stream_id is not None else _current_stream_id()
    return _PhaseCtx(phase, marker_step, marker_stream)


def _export(prof, out: str, steps: int) -> None:  # pragma: no cover - heavy
    import os as _os

    _os.makedirs(_os.path.dirname(out) or ".", exist_ok=True)
    prof.export_chrome_trace(out)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    table_path = out.rsplit(".", 1)[0] + "-kernels.log"
    with open(table_path, "w") as f:
        f.write(f"torch.profiler key_averages over {steps} profiled step(s)\n")
        f.write(table)
        f.write("\n")
    _export_marker_sidecar(out)


def _marker_path(out: str) -> str:
    return out.rsplit(".", 1)[0] + "-markers.json"


def _export_marker_sidecar(out: str | None = None) -> None:
    path = out or _state.marker_output
    if not path or not _state.markers:
        return
    marker_path = _marker_path(path)
    os.makedirs(os.path.dirname(marker_path) or ".", exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump(
            {
                "schema": "freetoken-roctx-markers-v1",
                "roctx_available": _state.roctx is not None,
                "roctx_error": _state.roctx_error,
                "markers": _state.markers,
            },
            f,
            sort_keys=True,
            indent=2,
        )


def summarize_markers(markers: list[dict]) -> dict:
    """Summarize paired ROCTX/sidecar ranges without summing overlapping stages.

    Durations are host monotonic-clock intervals. ``critical_step`` reports the
    outer scheduler envelope; phase totals remain attribution data and are never
    presented as additive wall time.
    """
    open_ranges: dict[tuple[object, object, object], list[int]] = {}
    phase_durations: dict[str, list[int]] = {}
    step_durations: list[int] = []
    errors: list[str] = []
    for marker in markers:
        if not isinstance(marker, dict):
            errors.append("marker is not an object")
            continue
        try:
            key = (marker["step"], marker["phase"], marker.get("stream_id"))
            stamp = int(marker["monotonic_ns"])
            event = marker["event"]
            phase = str(marker["phase"])
        except (KeyError, TypeError, ValueError):
            errors.append("malformed marker")
            continue
        if event == "begin":
            open_ranges.setdefault(key, []).append(stamp)
        elif event == "end":
            starts = open_ranges.get(key)
            if not starts:
                errors.append(f"unmatched end for step={key[0]} phase={phase}")
                continue
            duration = stamp - starts.pop()
            if duration < 0:
                errors.append(f"negative duration for step={key[0]} phase={phase}")
                continue
            phase_durations.setdefault(phase, []).append(duration)
            if phase == "scheduler":
                step_durations.append(duration)
        else:
            errors.append(f"unknown marker event {event!r}")
    for step, phase, _stream in open_ranges:
        if open_ranges[(step, phase, _stream)]:
            errors.append(f"unmatched begin for step={step} phase={phase}")

    def stats(values: list[int]) -> dict[str, int | float]:
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "total_ns": sum(ordered),
            "median_ns": statistics.median(ordered),
            "p95_ns": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        }

    return {
        "schema": "freetoken-stage-summary-v1",
        "complete": not errors,
        "errors": errors,
        "phases": {phase: stats(values) for phase, values in sorted(phase_durations.items())},
        "critical_step": stats(step_durations) if step_durations else None,
        "note": "phase totals overlap; critical_step is the wall-clock envelope",
    }


def summarize_marker_file(path: str) -> dict:
    """Load a profiler marker sidecar and return a machine-readable stage summary."""
    with open(path) as marker_file:
        payload = json.load(marker_file)
    markers = payload.get("markers") if isinstance(payload, dict) else None
    summary = summarize_markers(markers if isinstance(markers, list) else [])
    summary["source"] = path
    if isinstance(payload, dict):
        summary["roctx_available"] = payload.get("roctx_available")
        summary["roctx_error"] = payload.get("roctx_error")
    return summary


if __name__ == "__main__":  # pragma: no cover - command-line artifact helper
    import argparse

    parser = argparse.ArgumentParser(description="Summarize FreeToken profiler markers")
    parser.add_argument("markers")
    parser.add_argument("--out")
    args = parser.parse_args()
    summary = json.dumps(summarize_marker_file(args.markers), indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as summary_file:
            summary_file.write(summary + "\n")
    else:
        print(summary)
