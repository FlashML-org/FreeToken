from __future__ import annotations

import importlib


def test_registered_model_specs_resolve_required_package_exports():
    """Every registry entry must name things its package actually exports.

    Driven by the live registry, and checked against the modules themselves rather than a
    second copy of the table -- a typo'd class name or a loader that was renamed out from
    under an entry fails here, at import, instead of at model construction.
    """
    from freetoken.models.register import _MODEL_REGISTRY

    for arch, spec in _MODEL_REGISTRY.items():
        module = importlib.import_module(spec.module)
        # Resolve the model class and the per-spec config/weight loaders (honoring overrides:
        # GGUF specs carry parse_gguf_config / iter_gguf_weights in these same fields).
        assert callable(getattr(module, spec.model_cls)), arch
        assert callable(getattr(module, spec.parse_config)), arch
        assert callable(getattr(module, spec.iter_weights)), arch


def test_gguf_qwen_architectures_are_registered():
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY

    assert GGUF_ARCH_TO_REGISTRY["qwen3"] == "Qwen3ForCausalLM"
    assert GGUF_ARCH_TO_REGISTRY["qwen3_moe"] == "Qwen3MoeForCausalLM"
    assert GGUF_ARCH_TO_REGISTRY["qwen3_5_moe"] == "Qwen3_5MoeForCausalLM"

    for module_name in ("freetoken.models.qwen3", "freetoken.models.qwen3_moe", "freetoken.models.qwen3_5_moe"):
        module = importlib.import_module(module_name)
        assert callable(module.parse_gguf_config)
