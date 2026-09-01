from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent
SKIP_CUDA_EXT = os.environ.get("FREETOKEN_SKIP_CUDA_EXT", "").strip() in {"1", "true", "yes", "on"}


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _is_hip_torch() -> bool:
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs


def _hip_runtime_paths() -> tuple[list[str], list[str], list[str]]:
    """(include_dirs, library_dirs, libraries) for the HIP runtime."""
    candidates = []
    for key in ("ROCM_HOME", "ROCM_PATH", "HIP_PATH"):
        val = os.environ.get(key)
        if val:
            candidates.append(Path(val))
    try:
        from torch.utils.cpp_extension import ROCM_HOME

        if ROCM_HOME:
            candidates.append(Path(ROCM_HOME))
    except Exception:
        pass
    candidates.append(Path("/opt/rocm"))

    home = next((p for p in candidates if (p / "include").exists()), None)
    if home is None:
        raise RuntimeError(
            "ROCM_HOME / HIP_PATH / /opt/rocm is required to build the HIP "
            "pinned-tensor extension. See docs/amd.md."
        )
    include_dirs = [str(home / "include")]
    library_dirs = [str(d) for d in (home / "lib", home / "lib64") if d.exists()]
    return include_dirs, library_dirs, ["amdhip64"]


def _ext_modules() -> list:
    # FREETOKEN_SKIP_CUDA_EXT skips the install-time C++ extensions on both
    # CUDA and HIP. The HIP host_register/device_ptr ctypes fallback in
    # kernel/pinned.py still works (see docs/amd.md).
    if SKIP_CUDA_EXT:
        return []

    compile_args = ["-O3", "-std=c++17"]
    if _is_hip_torch():
        include_dirs, library_dirs, libraries = _hip_runtime_paths()
        compile_args = compile_args + ["-DUSE_ROCM", "-D__HIP_PLATFORM_AMD__=1"]
        extra_link = []
    else:
        if CUDA_HOME is None:
            if SKIP_CUDA_EXT:
                return []
            # Preserve the historical hard failure for a CUDA-wheel install
            # without a toolkit — that is still the NVIDIA default path.
            include_dirs, library_dirs = _cuda_runtime_paths()
        else:
            _check_toolchain()
            include_dirs, library_dirs = _cuda_runtime_paths()
        libraries = ["cudart"]
        extra_link = []

    pinned = CppExtension(
        name="freetoken.kernel._pinned_tensor",
        sources=["python/freetoken/kernel/csrc/pinned_tensor.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=extra_link,
    )
    cpu_moe = CppExtension(
        name="freetoken.kernel._cpu_moe",
        sources=["python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args + ["-pthread"],
        extra_link_args=extra_link,
    )
    return [pinned, cpu_moe]


setup(
    ext_modules=_ext_modules(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
