"""Tests for OffloadMoeCache.set_bank_sources residency propagation.

Covers the PARTIAL-PIN contract: non-pinned residency is accepted only when
every non-pinned layer id is in cache.cpu_layer_ids and the cache was built
with prefill_overlap=True.  Previously (before the partial-pin feature) this
suite exercised an all-pinned-only contract where every non-pinned class
raised NotImplementedError.

CPU-only: imports torch and freetoken, runs inside the project venv.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.host_banks import HostResidency
from freetoken.moe.offload_cache import OffloadMoeCache

L, E = 4, 8


def _cache() -> OffloadMoeCache:
    return OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=E,
        device=torch.device("cpu"),
        quant_format="bf16",
    )


def _sources() -> dict[str, list[torch.Tensor]]:
    # bf16 bank schema is ("gate_up", "down"); one contiguous [E, ...] tensor
    # per layer per bank, uniform shape and dtype across layers.
    return {
        "gate_up": [torch.zeros(E, 6, 4) for _ in range(L)],
        "down": [torch.zeros(E, 4, 3) for _ in range(L)],
    }


def test_default_residency_is_all_pinned():
    """set_bank_sources with no layer_residency succeeds and defaults to all PINNED."""
    cache = _cache()
    cache.set_bank_sources(_sources())
    assert cache.layer_residency == [HostResidency.PINNED.value] * L


def test_explicit_all_pinned_accepted():
    """Passing all-pinned layer_residency is accepted and stored as a copy."""
    cache = _cache()
    residency = [HostResidency.PINNED.value] * L
    cache.set_bank_sources(_sources(), layer_residency=residency)
    assert cache.layer_residency == residency
    # Mutating the passed list must not affect the stored copy.
    residency[0] = "changed"
    assert cache.layer_residency[0] == HostResidency.PINNED.value


def test_pageable_not_cpu_layer_raises():
    """A PAGEABLE entry whose index is not in cpu_layer_ids raises ValueError."""
    cache = _cache()  # default: no cpu_layer_ids
    residency = [HostResidency.PINNED.value] * L
    residency[1] = HostResidency.PAGEABLE.value
    with pytest.raises(ValueError, match="are not CPU-decode layers"):
        cache.set_bank_sources(_sources(), layer_residency=residency)


def test_locked_not_cpu_layer_raises():
    """A LOCKED entry whose index is not in cpu_layer_ids raises ValueError."""
    cache = _cache()  # default: no cpu_layer_ids
    residency = [HostResidency.PINNED.value] * L
    residency[2] = HostResidency.LOCKED.value
    with pytest.raises(ValueError, match="are not CPU-decode layers"):
        cache.set_bank_sources(_sources(), layer_residency=residency)


def test_pageable_accepted_for_cpu_layer_with_overlap():
    """PAGEABLE layer in cpu_layer_ids with prefill_overlap=True is accepted."""
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=2 * E,
        device=torch.device("cpu"),
        quant_format="bf16",
        prefill_overlap=True,
        cpu_layer_ids=frozenset({1}),
    )
    residency = [HostResidency.PINNED.value] * L
    residency[1] = HostResidency.PAGEABLE.value
    cache.set_bank_sources(_sources(), layer_residency=residency)
    assert cache.layer_residency == residency
    assert cache._exempt_layers == frozenset({1})


def test_pageable_without_overlap_raises():
    """PAGEABLE layer in cpu_layer_ids but prefill_overlap=False raises ValueError."""
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=E,
        device=torch.device("cpu"),
        quant_format="bf16",
        cpu_layer_ids=frozenset({1}),
    )
    # prefill_overlap stays False (default for cache_size=E, not 2*E)
    residency = [HostResidency.PINNED.value] * L
    residency[1] = HostResidency.PAGEABLE.value
    with pytest.raises(ValueError, match="prefill_overlap"):
        cache.set_bank_sources(_sources(), layer_residency=residency)


def test_wrong_length_asserts():
    """A residency list shorter than num_layers raises AssertionError."""
    cache = _cache()
    residency = [HostResidency.PINNED.value] * (L - 1)
    with pytest.raises(AssertionError):
        cache.set_bank_sources(_sources(), layer_residency=residency)


def test_schema_mismatch_asserts():
    """Sources missing the 'down' bank key raises AssertionError."""
    cache = _cache()
    partial = {"gate_up": _sources()["gate_up"]}
    with pytest.raises(AssertionError):
        cache.set_bank_sources(partial)


def test_expert_banks_residency_defaults_to_none():
    """ExpertBanks layer_residency is None when omitted from the constructor."""
    from freetoken.moe.expert_banks import ExpertBanks

    banks = ExpertBanks("bf16", {})
    assert banks.layer_residency is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
