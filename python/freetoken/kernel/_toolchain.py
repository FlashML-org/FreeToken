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
    import torch

    return getattr(torch.version, "hip", None) is not None


@functools.cache
def ensure_rocm_env() -> str | None:
    """Expose a wheel-provided ROCm SDK to JIT builders when the shell did not.

    Native Windows ROCm installs commonly live in ``_rocm_sdk_core`` rather than a
    system-wide ``C:\\Program Files\\AMD\\ROCm`` tree.  TVM-FFI keys its Windows HIP
    toolchain selection off ``HIP_PATH``; without it an otherwise healthy source run can
    fall back to MSVC/CUDA assumptions and fail only when the first uncached kernel builds.

    Existing user configuration wins.  ``ROCM_PATH`` is intentionally not synthesized:
    some builders interpret it as a traditional SDK layout and then look for device
    bitcode in a directory the modular wheel does not use.
    """
    if not _is_rocm():
        return None

    existing = os.environ.get("HIP_PATH")
    if existing:
        os.environ.setdefault("ROCM_HOME", existing)
        _ensure_rocm_arch()
        return existing

    try:
        import _rocm_sdk_core
    except ImportError:
        return None

    root = os.path.dirname(_rocm_sdk_core.__file__)
    clang_names = (
        os.path.join(root, "lib", "llvm", "bin", "clang.exe"),
        os.path.join(root, "lib", "llvm", "bin", "clang"),
    )
    if not any(os.path.isfile(path) for path in clang_names):
        return None

    os.environ["HIP_PATH"] = root
    os.environ.setdefault("ROCM_HOME", root)
    _ensure_rocm_arch()
    return root


def _ensure_rocm_arch() -> None:
    """Populate the arch variables used by the modular ROCm JIT toolchains."""
    names = (
        "TVM_FFI_ROCM_ARCH_LIST",
        "PYTORCH_ROCM_ARCH",
        "TRITON_OVERRIDE_ARCH",
        "ROCM_SDK_TARGET_FAMILY",
    )
    if all(os.environ.get(name) for name in names):
        return

    import torch

    try:
        if not torch.cuda.is_available():
            return
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0].strip()
    except Exception:
        return
    if not arch:
        return
    for name in names:
        os.environ.setdefault(name, arch)


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
