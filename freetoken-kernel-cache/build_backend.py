from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import build_meta as _build_meta

PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parent
FREETOKEN_PROJECT_ROOT = PACKAGE_PROJECT_ROOT.parent
PACKAGE_ROOT = PACKAGE_PROJECT_ROOT / "freetoken_kernel_cache"
BUILD_META = PACKAGE_ROOT / "_build_meta.py"


def _ensure_freetoken_importable() -> None:
    source_dir = FREETOKEN_PROJECT_ROOT / "python"
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)


def _check_toolchain() -> None:
    import importlib.util

    path = FREETOKEN_PROJECT_ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_version_suffix() -> str:
    override = os.getenv("FREETOKEN_KERNEL_CACHE_VERSION_SUFFIX")
    if override:
        return override if override.startswith("+") else f"+{override}"

    try:
        import torch
    except Exception:
        return ""

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        return ""
    # The tag advertises torch's CUDA; the cache .so link nvcc's libcudart.
    # Only a matching major makes both statements true at once.
    _check_toolchain()
    return f"+cu{cuda_version.replace('.', '')}"


def _write_build_meta() -> None:
    _ensure_freetoken_importable()
    from freetoken.version import __version__ as freetoken_version

    # The runtime version may carry its own local segment (`0.1.1+g<sha>` from a
    # stamped release build, see scripts/build-release-wheels.sh). PEP 440 allows a
    # single `+`, so merge instead of concatenating -- cu first, so every existing
    # `+cu` matcher keeps working (kernel/utils.py's regex, install.sh's
    # wheel_cuda_major): 0.1.1+g<sha> and +cu130 -> 0.1.1+cu130.g<sha>.
    base, _, local = freetoken_version.partition("+")
    suffix = _cuda_version_suffix()  # "+cu130" or ""
    if suffix and local:
        version = f"{base}{suffix}.{local}"
    else:
        version = f"{freetoken_version}{suffix}"
    BUILD_META.write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def _selected_specs() -> list[str] | None:
    raw = os.getenv("FREETOKEN_KERNEL_CACHE_SPECS", "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_jit_cache() -> None:
    _ensure_freetoken_importable()
    from freetoken.kernel.aot import compile_and_package_kernels

    out_dir = PACKAGE_ROOT / "jit_cache"
    build_dir = Path(
        os.getenv(
            "FREETOKEN_KERNEL_CACHE_BUILD_DIR",
            str(FREETOKEN_PROJECT_ROOT / "build" / "freetoken-kernel-cache"),
        )
    )
    verbose = os.getenv("FREETOKEN_KERNEL_CACHE_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # FREETOKEN_KERNEL_CACHE_CLEAN=0 keeps per-spec build directories so ninja can skip
    # unchanged kernels on a rebuild. Dev-loop only: a spec removed from the default list
    # would leave its stale directory in jit_cache and get packaged, so release builds
    # must keep the default (clean).
    clean = os.getenv("FREETOKEN_KERNEL_CACHE_CLEAN", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Multi-arch fatbin: one SASS cubin per listed arch, plus the PTX of the highest one.
    # A GPU whose arch is not listed and is below the highest one has no usable image.
    #   8.0  -> A100, A800, A30                      (Ampere, datacenter)
    #   8.6  -> RTX 30 series, A10, A40              (Ampere, consumer / workstation)
    #   8.9  -> RTX 40 series, L4, L40, RTX 6000 Ada (Ada Lovelace)
    #   9.0  -> H100, H800, H20                      (Hopper)
    #   10.0 -> B200, B100, GB200                    (Blackwell, datacenter)
    #   12.0 -> RTX 50 series, RTX PRO 6000 Blackwell (Blackwell, consumer / workstation)
    # Override with FREETOKEN_KERNEL_CACHE_ARCHES (space-separated maj.min) or
    # TVM_FFI_CUDA_ARCH_LIST directly. Needs an nvcc that supports every listed arch.
    if "TVM_FFI_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TVM_FFI_CUDA_ARCH_LIST"] = os.getenv(
            "FREETOKEN_KERNEL_CACHE_ARCHES", "8.0 8.6 8.9 9.0 10.0 12.0"
        )
    compile_and_package_kernels(
        out_dir=out_dir,
        build_dir=build_dir,
        specs=_selected_specs(),
        clean=clean,
        verbose=verbose,
    )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _write_build_meta()
    return _build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _write_build_meta()
    _build_jit_cache()
    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _write_build_meta()
    return _build_meta.build_sdist(sdist_directory, config_settings)


get_requires_for_build_wheel = _build_meta.get_requires_for_build_wheel
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist
