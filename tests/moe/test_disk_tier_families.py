"""Every NVFP4 family must hand the disk tier a source spec.

The loader releases expert rows ``[K, E)`` for every family, so a family that reaches
that release without an index would serve zeroed experts silently. These tests pin the
hook (``nvfp4_expert_spec``, resolved by ``moe.expert_pieces.nvfp4_expert_spec_of``)
that keeps the disk index and the loader reading the same rows.
"""
from __future__ import annotations

import importlib

import pytest

# Families whose NVFP4 experts load through the shared reader (nvfp4_expert_spec).
NVFP4_FAMILIES = [
    "qwen3_5_moe", "qwen4_exp", "glm4_moe", "glm5_next", "gemma4", "minimax_m2", "minimax_m3",
]


@pytest.mark.parametrize("family", NVFP4_FAMILIES)
def test_family_exposes_a_source_spec_hook(family):
    mod = importlib.import_module(f"freetoken.models.{family}.weight")
    getter = getattr(mod, "nvfp4_expert_spec", None)
    assert callable(getter), (
        f"{family} defines _NVFP4_SOURCE_SPEC but exposes no nvfp4_expert_spec, so "
        "the disk tier cannot build a disk index for it")


@pytest.mark.parametrize("family", [f for f in NVFP4_FAMILIES if f != "glm5_next"])
def test_hook_returns_the_spec_the_loader_uses(family):
    # glm5_next is excluded here only because its hook reads the checkpoint config to pick
    # between the compressed-tensors and modelopt namings; it is covered by the test below.
    mod = importlib.import_module(f"freetoken.models.{family}.weight")
    spec = mod.nvfp4_expert_spec("unused/for/these/families", None)
    assert spec is mod._NVFP4_SOURCE_SPEC
    assert spec.key_pattern.groupindex.keys() >= {"layer", "expert", "proj", "kind"}
    assert set(spec.proj_to_role.values()) == {"gate", "up", "down"}


def test_glm5_next_hook_follows_the_checkpoint_quant_method(monkeypatch):
    mod = importlib.import_module("freetoken.models.glm5_next.weight")

    class _Cfg:
        def __init__(self, method):
            self.quantization_config = {"quant_method": method}

    monkeypatch.setattr(mod, "cached_load_hf_config", lambda path: _Cfg("compressed-tensors"))
    assert mod.nvfp4_expert_spec("p", None) is mod._NVFP4_CT_SOURCE_SPEC
    monkeypatch.setattr(mod, "cached_load_hf_config", lambda path: _Cfg("modelopt"))
    assert mod.nvfp4_expert_spec("p", None) is mod._NVFP4_SOURCE_SPEC


def test_provider_refuses_a_family_without_a_spec(monkeypatch):
    """A family with no hook must fail at load, not release rows and serve zeros."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe import expert_banks
    from freetoken.moe.disk_tier import DiskTierSpec

    monkeypatch.setattr("freetoken.moe.expert_pieces.nvfp4_expert_spec_of",
                        lambda path, config: None)

    class _Kernel:
        name = "triton"

    class _Method:
        kind = QuantKind.NVFP4
        kernel = _Kernel()

    class _Cfg:
        num_moe_layers = 1
        architectures = ["SomeMoEForCausalLM"]

    with pytest.raises(NotImplementedError, match="nvfp4_expert_spec"):
        expert_banks._method_expert_banks(
            "does/not/matter", _Cfg(), _Method(), None, False,
            False, 8, 8 << 20, disk_tier=DiskTierSpec(ram_experts=1))
