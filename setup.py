from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).parent
# All extension sources MUST be relative to ROOT. setuptools>=77's build_py runs
# assert_relative() on every source path; an absolute path (str(ROOT / "..."))
# raises DistutilsSetupError: "setup script specifies an absolute path". Relative
# paths are the correct, portable form and keep the CPU build reproducible on
# modern setuptools (incl. GitHub's free ubuntu-latest runners).
SRC = ROOT / "python" / "freetoken" / "kernel" / "csrc"
STUB_DIR = str(SRC / "cpu_moe")


# --- CPU-only build path ---------------------------------------------------
# When FREETOKEN_CPU_ONLY=1 we build the pure-C++ extensions WITHOUT the CUDA
# toolkit: we supply a stub <cuda_runtime_api.h> (no-op symbols) and link no
# cudart. This is the OR-switch's CPU branch -- free, open-source, reproducible
# on any x86-64 Linux box (e.g. a free GitHub Actions runner, no NVIDIA GPU).
CPU_ONLY = os.environ.get("FREETOKEN_CPU_ONLY", "0") == "1"


def _toolchain_ok() -> bool:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    if not path.exists():
        return True
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.check_nvcc_matches_torch()  # CUDA-only check; skip on CPU build
        return True
    except Exception:
        return CPU_ONLY  # tolerate nvcc mismatch only on the CPU-only path


def _cpu_ext(name: str, rel_source: str, extra: list[str]) -> CppExtension:
    """Build a C++ extension with the stub cuda header, no cudart, no nvcc.

    `rel_source` is relative to setup.py (e.g. "python/freetoken/.../x.cpp") so
    setuptools never sees an absolute path (assert_relative would reject it).
    """
    return CppExtension(
        name=name,
        sources=[rel_source],
        include_dirs=[STUB_DIR],
        extra_compile_args=["-O3", "-std=c++17", "-pthread", "-DFREETOKEN_CPU_ONLY"]
        + extra,
        # No libraries= (no cudart); the stub header provides the symbols.
    )


if CPU_ONLY:
    _toolchain_ok()  # validated, but nvcc check is bypassed
    ext_modules = [
        _cpu_ext(
            "freetoken.kernel._pinned_tensor",
            "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            extra=[],
        ),
        _cpu_ext(
            "freetoken.kernel._cpu_moe",
            "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            extra=[],
        ),
    ]
    print("[setup.py] CPU-ONLY build: building C++ extensions with stub cuda runtime (no cudart/nvcc).")
else:
    from torch.utils.cpp_extension import CUDA_HOME

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

    _cuda_include_dirs, _cuda_library_dirs = _cuda_runtime_paths()
    _toolchain_ok()

    ext_modules = [
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=["python/freetoken/kernel/csrc/pinned_tensor.cpp"],
            include_dirs=_cuda_include_dirs,
            library_dirs=_cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=["python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"],
            include_dirs=_cuda_include_dirs,
            library_dirs=_cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
    ]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
