"""Tests for pin-exempt layer spec parser (``--pin-exempt-layers``).

CPU-only: exercises ``parse_layer_subset_spec`` / ``resolve_pin_exempt_layers``
without a GPU.
"""

from __future__ import annotations

import pytest

from freetoken.moe.pin_policy import parse_layer_subset_spec, resolve_pin_exempt_layers

L = 40


def test_explicit_list():
    assert parse_layer_subset_spec("3,7,11", L) == frozenset({3, 7, 11})
    assert parse_layer_subset_spec("3, 7 ,11,", L) == frozenset({3, 7, 11})
    assert parse_layer_subset_spec("5,5,5", L) == frozenset({5})


def test_count_exact_sets():
    assert parse_layer_subset_spec("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse_layer_subset_spec("1", L) == frozenset({0})
    assert len(parse_layer_subset_spec(str(L), L)) == L
    assert parse_layer_subset_spec("0", L) == frozenset()


def test_fraction():
    assert len(parse_layer_subset_spec("0.5", L)) == 20
    assert len(parse_layer_subset_spec("1.0", L)) == 40
    assert parse_layer_subset_spec("0.0", L) == frozenset()


def test_empty():
    assert parse_layer_subset_spec("", L) == frozenset()
    assert parse_layer_subset_spec("   ", L) == frozenset()
    assert resolve_pin_exempt_layers(None, L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5", "abc"])
def test_out_of_range_raises(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_layer_subset_spec(spec, L)


def test_error_message_contains_flag_name() -> None:
    # Default flag name appears in the message
    with pytest.raises(ValueError, match="--pin-exempt-layers"):
        parse_layer_subset_spec("99", L)
    # Custom flag name overrides
    with pytest.raises(ValueError, match="--moe-cpu-layers"):
        parse_layer_subset_spec("99", L, flag="--moe-cpu-layers")


def test_auto_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        resolve_pin_exempt_layers("auto", L)
    with pytest.raises(NotImplementedError):
        resolve_pin_exempt_layers("  AUTO ", L)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
