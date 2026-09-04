"""CUDA toolchain/torch consistency checks.

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


def _hipcc_path() -> str | None:
    """Locate hipcc: $ROCM_HOME/bin/hipcc, $HIP_PATH/bin/hipcc, /opt/rocm/bin/hipcc,
    then PATH."""
    for env in ("ROCM_HOME", "HIP_PATH"):
        root = os.getenv(env)
        if root:
            candidate = os.path.join(root, "bin", "hipcc")
            if os.path.isfile(candidate):
                return candidate
    default = "/opt/rocm/bin/hipcc"
    if os.path.isfile(default):
        return default
    return shutil.which("hipcc")


def hip_hip_version(hipcc: str) -> tuple[int, int] | None:
    """HIP toolkit version from ``hipcc --version`` (e.g. (6, 2)), or None."""
    try:
        proc = subprocess.run(
            [hipcc, "--version"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # hipcc --version prints e.g. "HIP version: 6.2.41000" (or a clang version line).
    m = re.search(r"HIP version[:\s]+(\d+)\.(\d+)", proc.stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\.(\d+)\.\d+", proc.stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def torch_hip_version() -> str | None:
    """The ``torch.version.hip`` string (e.g. "6.2.4100000"), or None on non-ROCm torch."""
    try:
        import torch

        return getattr(torch.version, "hip", None)
    except Exception:
        return None


def is_rocm_torch() -> bool:
    """True when the installed torch is a ROCm (AMD) build."""
    return bool(torch_hip_version())


def torch_hip_major() -> int | None:
    hip = torch_hip_version()
    if not hip:
        return None
    m = re.match(r"(\d+)", hip)
    return int(m.group(1)) if m else None


def check_hip_matches_torch() -> None:
    """Refuse to hipcc-compile kernels across HIP major versions.

    Mirrors check_nvcc_matches_torch: hipcc-built kernels link against the HIP runtime
    major they were built with; at runtime only the torch wheel's own HIP runtime is
    guaranteed to be loadable. No-op when torch is not ROCm.
    """
    if os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES:
        return
    if not is_rocm_torch():
        return
    torch_major = torch_hip_major()
    hipcc = _hipcc_path()
    if hipcc is None:
        raise RuntimeError(
            "ROCm torch detected but no hipcc found. Install a ROCm/HIP toolkit "
            "(e.g. via /opt/rocm) matching torch's HIP version, or set "
            f"{ALLOW_MISMATCH_ENV}=1 to override."
        )
    release = hip_hip_version(hipcc)
    if release is None:
        return
    if release[0] != torch_major:
        raise RuntimeError(
            f"hipcc {release[0]}.{release[1]} would build kernels linking HIP "
            f"{release[0]}.x, but torch ships HIP {torch_hip_version()}. Install a "
            f"ROCm {torch_major}.x toolkit, or set {ALLOW_MISMATCH_ENV}=1 to override."
        )


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
