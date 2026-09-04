from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent


def _load_toolchain():
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_toolchain() -> None:
    module = _load_toolchain()
    if module.is_rocm_torch():
        module.check_hip_matches_torch()
    else:
        module.check_nvcc_matches_torch()


def _rocm_home() -> Path | None:
    for env in ("ROCM_HOME", "HIP_PATH"):
        root = os.getenv(env)
        if root and (Path(root) / "include").exists():
            return Path(root)
    default = Path("/opt/rocm")
    return default if (default / "include").exists() else None


def _cuda_runtime_paths() -> tuple[list[str], list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs, ["cudart"]


def _rocm_runtime_paths() -> tuple[list[str], list[str], list[str]]:
    home = _rocm_home()
    if home is None:
        raise RuntimeError(
            "ROCm torch detected but no ROCm toolkit found. Install ROCm (e.g. /opt/rocm) "
            "matching torch's HIP version to build freetoken.kernel._pinned_tensor "
            "(it links the HIP runtime API)."
        )
    include_dirs = [str(home / "include")]
    library_dirs = []
    for sub in ("lib", "lib64"):
        if (home / sub).exists():
            library_dirs.append(str(home / sub))
    # HIP host APIs (hipHostMalloc/hipHostRegister/hipHostGetDevicePointer) and HIP
    # graph nodes all live in the HIP runtime, amdhip64. (hiprt is a separate optional
    # library not present on all ROCm installs; linking it would break the build.)
    libraries = ["amdhip64"]
    return include_dirs, library_dirs, libraries


def _runtime_paths() -> tuple[list[str], list[str], list[str], list[str]]:
    """Returns (include_dirs, library_dirs, libraries, compile_defs) for the active backend."""
    module = _load_toolchain()
    if module.is_rocm_torch():
        include_dirs, library_dirs, libraries = _rocm_runtime_paths()
        return include_dirs, library_dirs, libraries, ["-DUSE_HIP=1", "-DUSE_ROCM=1"]
    include_dirs, library_dirs, libraries = _cuda_runtime_paths()
    return include_dirs, library_dirs, libraries, []


include_dirs, library_dirs, libraries, compile_defs = _runtime_paths()
_check_toolchain()

_extra_compile_args = ["-O3", "-std=c++17", *compile_defs]

setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=_extra_compile_args,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. On CUDA it links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; on ROCm those become HIP graph nodes
        # (hipLaunchHostFunc) and we link the HIP runtime instead. The bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=_extra_compile_args + ["-pthread"],
        ),
        # --ple-backend disk row store; Linux-only until the TableFile/BatchReader seams grow Windows bodies
        *([
            CppExtension(
                name="freetoken.kernel._ple_store",
                sources=[
                    "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp",
                ],
                extra_compile_args=["-O3", "-std=c++17"],
            )
        ] if sys.platform == "linux" else []),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
