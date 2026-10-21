"""Optional measured gfx1201 FP8 decode kernels, disabled by default."""

import os


def enabled():
    return os.name == "posix" and os.environ.get("FREETOKEN_USE_ARK", "0") == "1"


def initialize():
    """Build/load outside HIP Graph capture, into a writable source-keyed cache."""
    from .common.native import load

    load()


def candidate_kernels(method, candidates):
    if not enabled():
        return candidates
    from .common.policy import MODE, current_environment, environment_matches

    if current_environment().get("arch") != "gfx1201":
        return candidates
    if MODE != "native" and not environment_matches():
        return candidates
    from freetoken.layers.quantization.registry import LayerKind
    from freetoken.layers.quantization.scheme import QuantKind
    from .kernels import ArkFp8LinearKernel, ArkFp8MoEKernel

    cls = {
        (QuantKind.FP8_BLOCK, LayerKind.MOE): ArkFp8MoEKernel,
        (QuantKind.FP8_BLOCK, LayerKind.LINEAR): ArkFp8LinearKernel,
    }.get((method.kind, method.layer_kind))
    if cls is None or cls().unusable_reason(method.cfg):
        return candidates
    initialize()
    return (cls, *candidates)
