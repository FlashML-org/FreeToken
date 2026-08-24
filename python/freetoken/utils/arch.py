from __future__ import annotations

import functools
import os
import re
from typing import Tuple


_GFX_ARCH_RE = re.compile(r"gfx\d+[a-z]?")


def _gfx_arch_from(value: object) -> str | None:
    match = _GFX_ARCH_RE.search(str(value).lower())
    return match.group(0) if match else None


@functools.cache
def is_rocm() -> bool:
    """True when torch is built for ROCm (AMD GPU) instead of CUDA."""
    import torch
    return getattr(torch.version, "hip", None) is not None


@functools.cache
def get_rocm_gfx_arch() -> str | None:
    """Return the current AMD GPU target (for example ``gfx1201``).

    Prefer the runtime device because build variables may contain multiple
    semicolon-separated targets. Environment variables remain useful for
    cross-compilation and systems where no GPU is currently visible.
    """
    if not is_rocm():
        return None

    import torch

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            for attr in ("gcnArchName", "arch"):
                arch = _gfx_arch_from(getattr(props, attr, ""))
                if arch:
                    return arch
        except (AttributeError, RuntimeError):
            pass

    for env_var in ("FREETOKEN_ROCM_ARCH", "PYTORCH_ROCM_ARCH", "HCC_AMDGPU_TARGET"):
        arch = _gfx_arch_from(os.getenv(env_var, ""))
        if arch:
            return arch
    return None


@functools.cache
def is_gfx11xx_family() -> bool:
    """True when the current AMD GPU is RDNA3 (gfx110x)."""
    arch = get_rocm_gfx_arch()
    return arch is not None and arch.startswith("gfx110")


@functools.cache
def is_gfx12xx_family() -> bool:
    """True when the current AMD GPU is RDNA4 (gfx120x)."""
    arch = get_rocm_gfx_arch()
    return arch is not None and arch.startswith("gfx120")


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if is_rocm():
        return None
    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """capability >= (major, minor). Open-ended: newer archs also pass. Only use this
    for family-portable features (e.g. PDL); arch-specific kernels (sm_90a/sm_100a
    cubins) need the closed is_smXX_family checks below."""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def _is_arch_family(major: int) -> bool:
    arch = _get_torch_cuda_version()
    return arch is not None and arch[0] == major


def is_sm90_family() -> bool:
    """Exactly major 9 (Hopper). For sm_90a-only kernels (e.g. FA3)."""
    return _is_arch_family(9)


def is_sm100_family() -> bool:
    """Exactly major 10 (datacenter Blackwell). For sm_100a/103a-only kernels
    (e.g. trtllm-gen) that consumer Blackwell (sm_120/121) cannot run."""
    return _is_arch_family(10)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
