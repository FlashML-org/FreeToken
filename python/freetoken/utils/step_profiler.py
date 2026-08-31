"""Env-gated per-step torch.profiler wrapper (stage-level decode breakdowns).

``FREETOKEN_TORCH_PROFILE="<warm>:<steps>:<out>"``, unset/empty = no-op (the only
hot-path cost is one cached check). Skip the first ``warm`` call(s) of the wrapped
region, then profile the next ``steps`` calls in one ``torch.profiler`` window and
export on the window's exit:

- a chrome trace to ``<out>`` and
- a top-kernels table to ``<out minus extension>-kernels.log``.

This is the Inc 2 instrument of .plans/rocm-perf-parity: NVTX is a no-op on ROCm,
so the trace segments by the explicit ``torch.profiler.record_function`` range
names (moe_router, moe_gate_up, moe_down, attn, Sampler) that appear as table row
names on both backends.
"""

from __future__ import annotations

import os

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
    __slots__ = ("spec", "calls", "done")

    def __init__(self) -> None:
        self.spec: tuple[int, int, str] | None = _NO_SPEC  # type: ignore[assignment]
        self.calls = 0
        self.done = False


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

    __slots__ = ("prof", "final")

    def __init__(self, prof, final: bool) -> None:
        self.prof = prof
        self.final = final
        if final:
            _state.done = True

    def __enter__(self):
        self.prof.__enter__()

    def __exit__(self, exc_type=None, exc=None, tb=None) -> bool:  # noqa: ANN001
        self.prof.__exit__(exc_type, exc, tb)
        if self.final:
            warm, steps, out = _state.spec or (0, 0, "")
            _export(self.prof, out, steps)
        return False


_NULL = _NullCtx()


def step_profiler():
    """Wrap one scheduler step. No-op unless FREETOKEN_TORCH_PROFILE is set (parsed
    on the first call); the window covers `steps` calls after the first `warm`."""
    if _state.done:
        return _NULL
    if _state.spec is _NO_SPEC:  # type: ignore[comparison-overlap]
        spec = _read_spec()
        if spec is None:
            _state.done = True
            return _NULL
        _state.spec = spec
    warm, steps, _ = _state.spec  # type: ignore[misc]
    _state.calls += 1
    if _state.calls <= warm:
        return _NULL

    import torch.profiler

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    )
    return _ProfilerCtx(prof, final=_state.calls >= warm + steps)


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