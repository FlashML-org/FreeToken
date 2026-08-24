from __future__ import annotations

import functools
from typing import Tuple

import torch


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

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


def default_compute_dtype(device: torch.device | None = None) -> torch.dtype:
    """Return the preferred floating-point dtype for ``device``.

    sm_80+ (Ampere and newer): BF16 — native ALUs, stable training range.
    sm_75 (Turing) and below: FP16 — BF16 has no hardware ALUs on Turing and
    runs in software emulation (~1.5-2× slower with identical accuracy).
    Falls back to float16 when no CUDA device is available.

    This matches the behaviour of vLLM-2080Ti-Definitive's CudaPlatformBase
    ``supported_dtypes``: ``has_device_capability(80)`` gates BF16.
    """
    if device is None:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None or device.type != "cuda":
        return torch.float16
    cc = torch.cuda.get_device_capability(device)
    if cc >= (8, 0):
        return torch.bfloat16
    # sm_75 / Turing and below: FP16 only
    return torch.float16
