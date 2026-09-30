from __future__ import annotations

import importlib.util
from importlib import metadata
import os
from pathlib import Path
import re

import sys

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, ROCM_HOME, CppExtension


ROOT = Path(__file__).parent
# The compiled PyTorch runtime selects which host runtime library the extension uses.
IS_ROCM = torch.version.hip is not None
GPU_RUNTIME_MACROS = [("FREETOKEN_USE_ROCM", "1")] if IS_ROCM else []
KERNEL_INCLUDE = str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include")
ROCM_ENV_REQUESTED = any(os.environ.get(name) for name in ("ROCM_HOME", "ROCM_PATH", "HIP_PATH"))  # Treat an explicit AMD root as build intent.
if ROCM_ENV_REQUESTED and not IS_ROCM:  # Reject combined ROCm selection with CUDA or CPU-only Torch.
    raise RuntimeError("ROCm environment variables require a ROCm 10 PyTorch build; remove CUDA/generic Torch packages")  # Fail before CUDA fallback.


def _numeric_version(value: str) -> tuple[int, ...]:  # Convert release text into comparable numeric components.
    match = re.search(r"\d+(?:\.\d+)+", value)  # Ignore distribution-specific prefixes and local suffixes.
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()  # Return an empty tuple for unparseable evidence.


def _distribution_version(name: str) -> str | None:  # Read installed-package evidence without making it a setup dependency.
    try:  # Package metadata may be absent in an incomplete environment.
        return metadata.version(name)  # Preserve the complete local-version suffix for ABI checks.
    except metadata.PackageNotFoundError:  # Turn absence into explicit preflight evidence.
        return None  # Let the caller produce one actionable contract error.


def _check_rocm10_environment(rocm_home: Path) -> None:  # Reject every accelerator environment outside the approved ROCm 10 contract.
    torch_build = str(getattr(torch, "__version__", ""))  # PyTorch's local suffix identifies the wheel's ROCm release.
    triton_build = _distribution_version("triton") or ""  # AMD Triton must carry the same ROCm release provenance.
    sdk_build = _distribution_version("rocm-sdk-core") or ""  # The SDK package version identifies the system distribution release.
    errors: list[str] = []  # Accumulate all mismatches so one build attempt reports the complete repair set.
    if not re.search(r"(?:^|[+.-])rocm10(?:\.|$)", torch_build):  # Reject CUDA, CPU-only, generic, and older ROCm Torch wheels.
        errors.append(f"torch {torch_build or '<unknown>'} is not a ROCm 10 build")  # Preserve the observed wheel identity.
    if not re.search(r"(?:^|[+.-])rocm10(?:\.|$)", triton_build):  # Reject generic or mismatched Triton distributions.
        errors.append(f"triton {triton_build or '<missing>'} is not a ROCm 10 build")  # Explain why native kernels are unsafe.
    if not _numeric_version(sdk_build) or _numeric_version(sdk_build)[0] != 10:  # Require ROCm SDK major version 10 exactly.
        errors.append(f"rocm-sdk-core {sdk_build or '<missing>'} is not release 10.x")  # Distinguish SDK release from HIP component metadata.
    for forbidden in ("flashinfer-python", "sglang-kernel"):  # CUDA-only extras must never leak into the HIP environment.
        forbidden_version = _distribution_version(forbidden)  # Probe without importing CUDA extension modules.
        if forbidden_version is not None:  # Treat any installed version as an ABI contamination.
            errors.append(f"CUDA-only package {forbidden} {forbidden_version} is installed")  # Name the exact package to remove.
    selected_root = rocm_home.resolve()  # Normalize symlinks before comparing the three toolkit selectors.
    for variable in ("ROCM_HOME", "ROCM_PATH", "HIP_PATH"):  # Require every explicitly supplied root to identify one SDK.
        configured = os.environ.get(variable)  # Ignore selectors the caller did not set.
        if configured and Path(configured).resolve() != selected_root:  # Detect mixed headers and libraries before compilation.
            errors.append(f"{variable}={configured} does not match ROCm root {selected_root}")  # Report both conflicting origins.
    if errors:  # Fail before defining extensions against an unsupported ABI.
        raise RuntimeError("ROCm 10 environment preflight failed: " + "; ".join(errors))  # Provide one deterministic repair message.


def _check_toolchain() -> None:
    if IS_ROCM:
        # nvcc/CUDA-major checks below are meaningless on a ROCm torch build
        # (torch.version.cuda is None there), so _toolchain.py's check is a no-op.
        return
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _gpu_runtime_paths() -> tuple[list[str], list[str], list[str], list[str]]:
    """Returns (include_dirs, library_dirs, libraries, extra_link_args)."""
    if IS_ROCM:
        rocm_root = os.environ.get("ROCM_HOME") or str(ROCM_HOME or "")
        if not rocm_root:
            raise RuntimeError(
                "A HIP PyTorch build requires the ROCm 10 SDK; set ROCM_HOME, ROCM_PATH, "
                "and HIP_PATH to the same SDK root."
            )
        rocm_home = Path(rocm_root)
        _check_rocm10_environment(rocm_home)  # Verify the externally provisioned runtime before locating headers or libraries.
        library_dirs = [d for d in (rocm_home / "lib64", rocm_home / "lib") if d.exists()]
        if not (rocm_home / "include" / "hip" / "hip_runtime.h").exists():
            raise RuntimeError(f"HIP headers not found under ROCM_HOME={rocm_home}")
        # Select the toolkit's linkable HIP runtime library without assuming a soname suffix.
        hip_candidates = [f for d in library_dirs for f in d.glob("libamdhip64.so*")]  # Collect every linker name from the selected root.
        hip_lib = max(hip_candidates, key=lambda path: _numeric_version(path.name), default=None)  # Prefer the greatest numeric SONAME.
        if hip_lib is None:
            raise RuntimeError(f"libamdhip64.so* not found under {library_dirs}")
        return (
            [KERNEL_INCLUDE, str(rocm_home / "include")],
            [str(d) for d in library_dirs],
            [f":{hip_lib.name}"],
            [f"-Wl,-rpath,{hip_lib.parent}"],
        )
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME (or ROCM_HOME) is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA/HIP runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [KERNEL_INCLUDE, str(cuda_home / "include")], library_dirs, ["cudart"], []


cuda_include_dirs, cuda_library_dirs, cuda_libraries, cuda_extra_link_args = _gpu_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=cuda_libraries,
            extra_link_args=cuda_extra_link_args,
            extra_compile_args=["-O3", "-std=c++17"],
            define_macros=GPU_RUNTIME_MACROS,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart/amdhip64 for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=cuda_libraries,
            extra_link_args=cuda_extra_link_args,
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
            define_macros=GPU_RUNTIME_MACROS,
        ),
        # disk-backed row store (PLE / Engram tables); Linux-only until the TableFile/BatchReader seams grow Windows bodies
        *([
            CppExtension(
                name="freetoken.kernel._row_store",
                sources=[
                    "python/freetoken/kernel/csrc/row_store/row_store_ext.cpp",
                ],
                extra_compile_args=["-O3", "-std=c++17"],
            )
        ] if sys.platform == "linux" else []),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
