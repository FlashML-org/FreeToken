"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _pin_budget_bytes as pin_budget
from freetoken.engine.engine import _resolve_cpu_layers as resolve
from freetoken.engine import engine as engine_module

L = 40


def test_explicit_list():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})  # whitespace + trailing comma
    assert parse("5,5,5", L) == frozenset({5})  # dups collapse


def test_count_evenly_strided():
    assert parse("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse("1", L) == frozenset({0})
    assert len(parse(str(L), L)) == L  # all layers
    assert parse("0", L) == frozenset()


def test_fraction():
    assert len(parse("0.5", L)) == L // 2
    assert len(parse("1.0", L)) == L
    assert parse("0.0", L) == frozenset()


def test_empty():
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5"])
def test_out_of_range_raises(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def _cfg(backend, spec=None):
    return SimpleNamespace(moe_backend=backend, moe_cpu_layers=spec)


def test_resolve_backend_dispatch():
    # cpu backend -> every layer, ignoring any spec
    assert resolve(_cfg("cpu"), L) == frozenset(range(L))
    assert resolve(_cfg("cpu", "8"), L) == frozenset(range(L))
    # offload + spec -> parsed subset
    assert len(resolve(_cfg("offload", "8"), L)) == 8
    # offload, no spec -> none (plain offload)
    assert resolve(_cfg("offload", None), L) == frozenset()
    # non-offload backend ignores the spec (validation lives in _adjust_config)
    assert resolve(_cfg("fused", "8"), L) == frozenset()


def test_pin_budget_native_windows(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    monkeypatch.setattr(engine_module.os, "name", "nt")
    monkeypatch.setattr(engine_module, "_windows_total_physical_memory", lambda: 128 << 30)
    assert pin_budget() == int((128 << 30) * 0.4)


def test_pin_budget_env_overrides_platform(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "47.5")
    monkeypatch.setattr(engine_module.os, "name", "nt")
    monkeypatch.setattr(engine_module, "_windows_total_physical_memory", lambda: 1)
    assert pin_budget() == int(47.5 * 2**30)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
