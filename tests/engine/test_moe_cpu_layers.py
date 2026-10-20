"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("local_safe,remote_safe", [(True, True), (False, True), (True, False)])
def test_cpu_moe_graph_fallback_agrees_across_tp_ranks(monkeypatch, local_safe, remote_safe):
    import torch
    from freetoken.engine.engine import _disable_unsafe_cpu_moe_graphs

    config = SimpleNamespace(tp_info=SimpleNamespace(size=2), cuda_graph_bs=[1, 4], cuda_graph_max_bs=4)
    executor = SimpleNamespace(graph_capture_safe=lambda dtype: local_safe)
    group = object()
    calls = []

    def reduce(tensor, *, op, group):
        calls.append((op, group))
        tensor.fill_(bool(tensor.item()) and remote_safe)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    _disable_unsafe_cpu_moe_graphs(config, executor, torch.bfloat16, group)
    assert calls == [(torch.distributed.ReduceOp.MIN, group)]
    assert config.cuda_graph_bs == ([1, 4] if local_safe and remote_safe else [])
    assert config.cuda_graph_max_bs == (4 if local_safe and remote_safe else 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
