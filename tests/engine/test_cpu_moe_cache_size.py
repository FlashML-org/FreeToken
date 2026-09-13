"""Resolver for the cpu MoE strategy's GPU prefill slot cache (--moe-cache-size).

CPU-only: exercises _resolve_cpu_moe_cache without a GPU.
"""

from __future__ import annotations

import pytest

from freetoken.engine.engine import _resolve_cpu_moe_cache as resolve

E = 256  # num_experts of Qwen3.6-35B-A3B


def test_explicit_size_is_authoritative():
    # The pre-0.1.3 behavior doubled this to 2*E = 512.
    size, overlap = resolve(256, True, E)
    assert size == 256
    assert overlap is False  # 256 < 2*E cannot feed the two-buffer overlap


def test_explicit_size_at_or_above_double_keeps_overlap():
    size, overlap = resolve(512, True, E)
    assert size == 512
    assert overlap is True
    size, overlap = resolve(1024, True, E)
    assert size == 1024
    assert overlap is True


def test_explicit_size_respects_disabled_overlap():
    size, overlap = resolve(256, False, E)
    assert size == 256
    assert overlap is False


def test_default_is_two_layer_double_buffer():
    size, overlap = resolve(0, True, E)
    assert size == 2 * E
    assert overlap is True
    # even if the user passed --disable-moe-prefill-overlap without a size,
    # the default two-layer buffer keeps overlap semantics
    size, overlap = resolve(0, False, E)
    assert size == 2 * E
    assert overlap is True


def test_negative_size_treated_as_default():
    # argparse-level guard aside, be robust: anything not > 0 means "not explicit"
    size, overlap = resolve(-1, True, E)
    assert size == 2 * E
    assert overlap is True


def test_small_expert_models():
    # 8-expert model: explicit 16 == 2*E boundary keeps overlap
    size, overlap = resolve(16, True, 8)
    assert size == 16
    assert overlap is True
    # explicit 15 < 2*E disables it
    size, overlap = resolve(15, True, 8)
    assert size == 15
    assert overlap is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
