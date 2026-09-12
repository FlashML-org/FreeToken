"""CUDA toolchain/torch consistency checks.

Standalone on purpose: setup.py and the kernel-cache build backend load this
file by path, so it must not import the freetoken package.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_CUDA_MISMATCH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _mismatch_allowed() -> bool:
    return os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES


def _torch_cuda_release() -> tuple[int, int] | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None

    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return None
    match = re.match(r"^(\d+)\.(\d+)", str(cuda))
    if match is None:
        raise RuntimeError(f"torch reports an invalid CUDA release: {cuda!r}")
    return int(match.group(1)), int(match.group(2))


def _prepend_path(directory: str) -> None:
    entries = [item for item in os.getenv("PATH", "").split(os.pathsep) if item]
    entries = [item for item in entries if os.path.abspath(item) != directory]
    os.environ["PATH"] = os.pathsep.join([directory, *entries])


def _publish_cuda_home(cuda_home: Path) -> str:
    home = str(cuda_home)
    os.environ["CUDA_HOME"] = home
    _prepend_path(str(cuda_home / "bin"))

    # Normally configure_cuda_toolchain() runs before cpp_extension is imported.
    # Keep direct library/JIT callers correct too if another dependency imported it first.
    cpp_extension = sys.modules.get("torch.utils.cpp_extension")
    if cpp_extension is not None:
        cpp_extension.CUDA_HOME = home
    return home


def _release_error(
    nvcc: Path,
    expected: tuple[int, int],
    actual: tuple[int, int] | None,
) -> str:
    wanted = f"{expected[0]}.{expected[1]}"
    if actual is None:
        return f"{nvcc} does not report a valid CUDA release; torch requires CUDA {wanted}"
    return (
        f"nvcc {actual[0]}.{actual[1]} at {nvcc} does not match "
        f"torch CUDA {wanted}"
    )


def configure_cuda_toolchain(*, reject_path_mismatch: bool = True) -> str | None:
    """Select torch's exact CUDA toolkit in this process before any JIT/build.

    An explicit absolute ``CUDA_HOME`` is authoritative.  Otherwise discovery is
    deliberately bounded to the exact versioned ``/usr/local`` toolkit and the
    current ``PATH``.  ``reject_path_mismatch=False`` lets a prebuilt-only server
    ignore an unrelated compiler on ``PATH``; actual JIT/build checks stay strict.
    We never rewrite system alternatives or shell state.
    """
    expected = _torch_cuda_release()
    if expected is None:
        return None

    explicit = os.getenv("CUDA_HOME", "").strip()
    if explicit:
        cuda_home = Path(explicit)
        if not cuda_home.is_absolute():
            raise RuntimeError(f"CUDA_HOME must be absolute, got {explicit!r}")
        nvcc = cuda_home / "bin" / "nvcc"
        actual = nvcc_release(str(nvcc))
        if actual is None:
            raise RuntimeError(_release_error(nvcc, expected, actual))
        if actual != expected and not _mismatch_allowed():
            raise RuntimeError(
                f"{_release_error(nvcc, expected, actual)}; set "
                f"{ALLOW_MISMATCH_ENV}=1 to override the explicit CUDA_HOME check"
            )
        return _publish_cuda_home(cuda_home)

    exact_home = Path(f"/usr/local/cuda-{expected[0]}.{expected[1]}")
    exact_nvcc = exact_home / "bin" / "nvcc"
    if nvcc_release(str(exact_nvcc)) == expected:
        return _publish_cuda_home(exact_home)

    path_nvcc_raw = shutil.which("nvcc")
    if path_nvcc_raw is None:
        return None
    # Derive the toolkit root from the PATH entry, not its resolved implementation.
    # Distro packages commonly expose /usr/bin/nvcc as a symlink into /usr/lib while
    # keeping their public CUDA headers and libraries rooted at /usr.
    path_nvcc = Path(os.path.abspath(path_nvcc_raw))
    actual = nvcc_release(str(path_nvcc))
    if actual is None:
        if not reject_path_mismatch:
            return None
        raise RuntimeError(_release_error(path_nvcc, expected, actual))
    if actual != expected and not _mismatch_allowed():
        if not reject_path_mismatch:
            return None
        raise RuntimeError(
            f"{_release_error(path_nvcc, expected, actual)}; install "
            f"/usr/local/cuda-{expected[0]}.{expected[1]} or set an absolute CUDA_HOME"
        )
    return _publish_cuda_home(path_nvcc.parent.parent)


def _nvcc_path() -> str | None:
    cuda_home = os.getenv("CUDA_HOME", "").strip()
    if cuda_home:
        return os.path.join(cuda_home, "bin", "nvcc")
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
    release = _torch_cuda_release()
    return release[0] if release is not None else None


def check_nvcc_matches_torch() -> None:
    """Refuse to nvcc-compile kernels across CUDA release families.

    configure_cuda_toolchain() also publishes the selected toolkit before the
    downstream builder imports or spawns its compiler machinery.
    """
    configure_cuda_toolchain()
    torch_release = _torch_cuda_release()
    if torch_release is None:
        return
    nvcc = _nvcc_path()
    if nvcc is None:
        raise RuntimeError(
            f"no nvcc matching torch CUDA {torch_release[0]}.{torch_release[1]} was found; "
            "install the exact versioned toolkit or set an absolute CUDA_HOME"
        )
    release = nvcc_release(nvcc)
    if release is None:
        raise RuntimeError(
            f"{nvcc} does not report a valid CUDA release for torch CUDA "
            f"{torch_release[0]}.{torch_release[1]}"
        )
    if release != torch_release and not _mismatch_allowed():
        import torch

        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} would build kernels for a different "
            f"CUDA release family, but torch {torch.__version__} ships CUDA "
            f"{torch.version.cuda}. Set an exact absolute CUDA_HOME, or set "
            f"{ALLOW_MISMATCH_ENV}=1 to override."
        )
