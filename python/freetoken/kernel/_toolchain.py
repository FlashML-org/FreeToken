"""CUDA/HIP toolchain/torch consistency checks.

Standalone on purpose: setup.py and the kernel-cache build backend load this
file by path, so it must not import the freetoken package.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess

ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_CUDA_MISMATCH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_rocm() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(getattr(torch.version, "hip", None))


def _hipcc_path() -> str | None:
    """Locate hipcc from explicit toolkit roots or PATH."""
    for env in ("ROCM_HOME", "HIP_PATH"):
        root = os.getenv(env)
        if root:
            candidate = os.path.join(root, "bin", "hipcc")
            if os.path.isfile(candidate):
                return candidate
    default = "/opt/rocm/bin/hipcc"
    return default if os.path.isfile(default) else shutil.which("hipcc")


def hip_hip_version(hipcc: str) -> tuple[int, int] | None:
    """Return HIP toolkit major/minor reported by hipcc."""
    try:
        proc = subprocess.run([hipcc, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"HIP version[:\s]+(\d+)\.(\d+)", proc.stdout)
    if match is None:
        match = re.search(r"(\d+)\.(\d+)\.\d+", proc.stdout)
    return (int(match.group(1)), int(match.group(2))) if match else None


def torch_hip_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return getattr(torch.version, "hip", None)


def torch_hip_major() -> int | None:
    hip = torch_hip_version()
    match = re.match(r"(\d+)", hip or "")
    return int(match.group(1)) if match else None


def is_rocm_torch() -> bool:
    return bool(torch_hip_version())


def check_hip_matches_torch() -> None:
    """Reject hipcc builds from a different HIP major than torch."""
    if os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES:
        return
    if not is_rocm_torch():
        return
    torch_major = torch_hip_major()
    hipcc = _hipcc_path()
    if hipcc is None:
        raise RuntimeError(
            "ROCm torch detected but no hipcc found. Install a matching ROCm toolkit "
            f"or set {ALLOW_MISMATCH_ENV}=1 to override."
        )
    release = hip_hip_version(hipcc)
    if release is None or torch_major is None or release[0] == torch_major:
        return
    raise RuntimeError(
        f"hipcc {release[0]}.{release[1]} does not match torch HIP {torch_hip_version()}; "
        f"install ROCm {torch_major}.x or set {ALLOW_MISMATCH_ENV}=1 to override."
    )


def check_toolchain_matches_torch() -> None:
    if is_rocm_torch():
        check_hip_matches_torch()
    else:
        check_nvcc_matches_torch()


def _nvcc_path() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME:
        return os.path.join(CUDA_HOME, "bin", "nvcc")
    return shutil.which("nvcc")


def nvcc_release(nvcc: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", proc.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def torch_cuda_major() -> int | None:
    import torch

    cuda = getattr(torch.version, "cuda", None)
    return int(cuda.split(".")[0]) if cuda else None


@functools.cache
def check_nvcc_matches_torch() -> None:
    """Refuse to nvcc-compile kernels across CUDA majors.

    nvcc-built binaries link libcudart.so.<nvcc major>; at runtime only the
    torch wheel's own CUDA runtime is guaranteed to be loadable.
    """
    if _is_rocm():
        return  # ROCm uses hipcc, not nvcc
    if os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES:
        return
    torch_major = torch_cuda_major()
    if torch_major is None:
        return
    nvcc = _nvcc_path()
    if nvcc is None:
        return
    release = nvcc_release(nvcc)
    if release is None:
        return
    if release[0] != torch_major:
        import torch

        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} would build kernels linking "
            f"libcudart.so.{release[0]}, but torch {torch.__version__} ships CUDA "
            f"{torch.version.cuda} (libcudart.so.{torch_major}). Install a CUDA "
            f"{torch_major}.x toolkit, or set {ALLOW_MISMATCH_ENV}=1 to override."
        )
