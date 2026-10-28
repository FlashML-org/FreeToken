"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.engine import Engine
from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _resolve_cpu_layers as resolve

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
    return SimpleNamespace(moe_strategy=backend, moe_cpu_layers=spec)


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


def test_residency_split_returns_decode_target_and_lock_plan():
    engine = Engine.__new__(Engine)
    engine._host_tables_bytes = 0
    config = SimpleNamespace(
        model_config=SimpleNamespace(decode_target="gpu"),
        moe_strategy="offload",
        moe_prefill_overlap=True,
    )

    result = engine._residency_split(config, frozenset({2}), None, capped=True)

    assert result == ("gpu", True)
    assert config.moe_prefill_overlap is False


def test_engine_moe_pool_helpers_preserve_per_layout_pool_constraints():
    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=8, num_experts_per_tok=2)
    )
    engine._moe_pool_groups = [[0, 1], [2]]
    engine._moe_layer_slot_bytes = [10, 10, 30]
    engine._moe_slot_limit = 12

    pools = engine._moe_pools(batch=3)

    assert [(pool.layers, pool.min_num_slots, pool.max_num_slots) for pool in pools] == [
        ((0, 1), 6, 12),
        ((2,), 6, 8),
    ]
    assert engine._moe_pool_bytes([10, 7]) == 310


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
