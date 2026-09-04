from __future__ import annotations

import functools
from typing import Tuple


def device_kind() -> str:
    """Compute backend the current torch build targets: ``"cuda"`` (NVIDIA), ``"rocm"``
    (AMD/HIP) or ``"cpu"``. Keyed on the *build* (torch.version.hip vs torch.version.cuda),
    independent of whether a GPU is present, so feature-gating and graceful-degradation
    decisions can be made before any device is available. On ROCm torch, torch.version.cuda
    is None and torch.version.hip is set; on CUDA torch the inverse holds."""
    try:
        import torch.version
    except Exception:
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


def is_rocm() -> bool:
    """True when the installed torch is a ROCm (AMD) build."""
    return device_kind() == "rocm"


def is_cuda() -> bool:
    """True when the installed torch is a CUDA (NVIDIA) build."""
    return device_kind() == "cuda"


@functools.cache
def current_gpu_name() -> str | None:
    """Device name of the current CUDA-capable device (``torch.cuda.get_device_name``), or
    None if torch is unavailable / no device. On ROCm this returns the AMD card name through
    the torch.cuda compat layer."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return None


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """Compute capability ``(major, minor)`` of the current CUDA device, or None when it
    cannot be determined. Returns None on ROCm torch (gfx archs are not a CUDA compute
    capability), when no CUDA device is present, and when torch itself is unavailable --
    so every ``is_sm*``/``is_arch_supported`` gate degrades to the portable path."""
    try:
        import torch
        import torch.version

        if not torch.cuda.is_available() or not torch.version.cuda:
            return None
        return torch.cuda.get_device_capability()
    except Exception:
        return None


@functools.cache
def _get_gfx_arch() -> int | None:
    """Numeric gfx arch of the current device (e.g. 1100 for ``gfx1100``) on ROCm, or
    None when torch is unavailable / not ROCm / no device present. Used by
    :func:`is_gfx_arch_ge` for AMD feature gating."""
    try:
        import torch
        import torch.version

        if not torch.version.hip or not torch.cuda.is_available():
            return None
        import re

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # ROCm torch exposes the exact gfx string (e.g. ``gfx1100``) as gcnArchName,
        # which is the reliable field; the marketing device name is often just
        # ``Radeon RX 7900 XTX`` and carries no gfx marker.
        gcn = getattr(props, "gcnArchName", None)
        if gcn:
            m = re.search(r"gfx(\d{3,4})", str(gcn))
            if m:
                return int(m.group(1))
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        if name:
            m = re.search(r"gfx(\d{3,4})", name)
            if m:
                return int(m.group(1))
        return None
    except Exception:
        return None


def is_gfx_arch_ge(arch_int: int) -> bool:
    """True on ROCm when the current gfx arch number is >= ``arch_int`` (e.g.
    ``is_gfx_arch_ge(1100)`` for RDNA 3 / RX 7000). Parses the full gfx string
    (``gfx1100`` -> 1100) rather than a CUDA-style ``(major, minor)`` tuple. Returns
    False on CUDA and CPU builds, so every gfx gate degrades to the portable path."""
    gfx = _get_gfx_arch()
    if gfx is None:
        return False
    return gfx >= arch_int


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
