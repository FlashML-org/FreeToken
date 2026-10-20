from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path

import sys

from setuptools import Extension, setup
from setuptools.command.build_clib import build_clib
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension, ROCM_HOME


ROOT = Path(__file__).parent
KERNEL_INCLUDE = str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include")


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _is_rocm() -> bool:
    import torch
    return getattr(torch.version, "hip", None) is not None


def _rocm_paths() -> tuple[list[str], list[str], str]:
    candidates: list[Path] = []
    if os.getenv("ROCM_HOME"):
        candidates.append(Path(os.environ["ROCM_HOME"]))
    if ROCM_HOME:
        candidates.append(Path(ROCM_HOME))

    # ROCm 7.14 PyTorch images ship the SDK as a Python package instead of
    # installing it at /opt/rocm.
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec and spec.submodule_search_locations:
        candidates.append(Path(next(iter(spec.submodule_search_locations))))
    candidates.append(Path("/opt/rocm"))

    for rocm_home in dict.fromkeys(candidates):
        include_dir = rocm_home / "include"
        library_dir = rocm_home / "lib"
        if not (include_dir / "hip" / "hip_runtime.h").exists():
            continue
        if (library_dir / "libamdhip64.so").exists():
            return [str(include_dir)], [str(library_dir)], "amdhip64"
        versioned = sorted(library_dir.glob("libamdhip64.so.*"))
        if versioned:
            return [str(include_dir)], [str(library_dir)], f":{versioned[-1].name}"

    searched = ", ".join(str(path) for path in dict.fromkeys(candidates))
    raise RuntimeError(
        "A ROCm SDK with HIP headers and libamdhip64 is required to build on ROCm; "
        f"searched: {searched}. Set ROCM_HOME to override."
    )


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


IS_ROCM = _is_rocm()

if IS_ROCM:
    runtime_include_dirs, runtime_library_dirs, runtime_lib = _rocm_paths()
    runtime_link_args = [f"-Wl,-rpath,{runtime_library_dirs[0]}"]
    # Host sources. BuildExtension supplies the ROCm
    # platform defines to the C++ compiler; offload architecture flags belong on
    # HIP device sources and would be rejected by the host compiler here.
    extra_compile = ["-O3", "-std=c++20"]
    # Device sources (.cu, built by hipcc); BuildExtension adds the offload architectures.
    device_compile = ["-O3", "-std=c++20"]
else:
    runtime_include_dirs, runtime_library_dirs = _cuda_runtime_paths()
    runtime_lib = "cudart"
    runtime_link_args = []
    extra_compile = ["-O3", "-std=c++20"]
    # Device sources (.cu, built by nvcc): SASS for each major arch, which also runs on the newer minors of that major,
    # plus PTX for the oldest arch nvcc 13 supports, which the driver builds for any other GPU at load.
    # Naming archs also stops torch from picking them, which fails on a build machine without a GPU.
    device_compile = ["-O3", "-std=c++20"] + [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (75, 80, 90, 100, 120)]
    device_compile.append("-gencode=arch=compute_75,code=compute_75")

_check_toolchain()

# ggml-cpu's row kernels for GGUF experts on the CPU MoE executor: one module per x86 variant of llama.cpp's GGML_CPU_ALL_VARIANTS (ggml/src/CMakeLists.txt), freetoken.kernel.ggml_cpu._<variant>.
# freetoken.kernel.ggml_cpu.select() imports the best one this CPU runs; each module's entry is built without the variant's ISA flags so that check is safe on any CPU.
GGML_DIR = "python/freetoken/kernel/csrc/ggml"
GGML_CPU = platform.machine().lower() in ("x86_64", "amd64")
_HASWELL = ("SSE42", "AVX", "F16C", "FMA", "AVX2", "BMI2")
GGML_CPU_VARIANTS = {
    "x64": (),
    "sse42": ("SSE42",),
    "sandybridge": ("SSE42", "AVX"),
    "ivybridge": ("SSE42", "AVX", "F16C"),
    "piledriver": ("SSE42", "AVX", "F16C", "FMA"),
    "haswell": _HASWELL,
    "skylakex": (*_HASWELL, "AVX512"),
    "cannonlake": (*_HASWELL, "AVX512", "AVX512_VBMI"),
    "cascadelake": (*_HASWELL, "AVX512", "AVX512_VNNI"),
    "icelake": (*_HASWELL, "AVX512", "AVX512_VBMI", "AVX512_VNNI"),
    "cooperlake": (*_HASWELL, "AVX512", "AVX512_VNNI", "AVX512_BF16"),
    "zen4": (*_HASWELL, "AVX512", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16"),
    "alderlake": (*_HASWELL, "AVX_VNNI"),
    "sapphirerapids": (*_HASWELL, "AVX512", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16", "AMX_TILE", "AMX_INT8"),
}
_GGML_GCC_FLAGS = {
    "SSE42": ["-msse4.2"], "F16C": ["-mf16c"], "FMA": ["-mfma"], "BMI2": ["-mbmi2"], "AVX": ["-mavx"], "AVX2": ["-mavx2"],
    "AVX_VNNI": ["-mavxvnni"], "AVX512": ["-mavx512f", "-mavx512cd", "-mavx512vl", "-mavx512dq", "-mavx512bw"],
    "AVX512_VBMI": ["-mavx512vbmi"], "AVX512_VNNI": ["-mavx512vnni"], "AVX512_BF16": ["-mavx512bf16"],
    "AMX_TILE": ["-mamx-tile"], "AMX_INT8": ["-mamx-int8"],
}


def _ggml_variant_flags(feats: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """(ISA flags, macros) of one variant, as upstream's ggml-cpu/CMakeLists.txt sets them; the macros also tell cpu-feats.cpp what to check."""
    if sys.platform != "win32":
        return [f for feat in feats for f in _GGML_GCC_FLAGS[feat]], [f"GGML_{feat}" for feat in feats]
    # cl takes one /arch: level and does not define the macros of the extensions above it
    if "AVX512" in feats:
        flags, macros = ["/arch:AVX512"], ["GGML_AVX512"]
        if "AVX512_VBMI" in feats:
            macros.append("__AVX512VBMI__")
        if "AVX512_VNNI" in feats:
            macros += ["__AVX512VNNI__", "GGML_AVX512_VNNI"]
    elif "AVX2" in feats:
        flags, macros = ["/arch:AVX2"], ["GGML_AVX2", "GGML_FMA", "GGML_F16C"]
    elif "AVX" in feats:
        flags, macros = ["/arch:AVX"], ["GGML_AVX"]
    elif "SSE42" in feats:
        flags, macros = ["/arch:SSE4.2"], ["GGML_SSE42"]
    else:
        flags, macros = [], []
    if "AVX_VNNI" in feats:
        macros += ["__AVXVNNI__", "GGML_AVX_VNNI"]
    if "BMI2" in feats:
        macros += ["__BMI2__", "GGML_BMI2"]
    return flags, macros


def _ggml_cpu_targets() -> tuple[list, list, dict[str, list[str]]]:
    """(build_clib libraries, variant modules, the static libraries each module links in link order)."""
    if not GGML_CPU:
        return [], [], {}
    if sys.platform == "win32":
        cflags, c_std, cxx_std, define = ["/O2", "/DNDEBUG", "/utf-8", "/D_CRT_SECURE_NO_WARNINGS"], "/std:c11", "/std:c++20", "/D"
        # upstream builds these variants only with GCC or Clang
        variants = {k: v for k, v in GGML_CPU_VARIANTS.items() if k not in ("ivybridge", "piledriver", "cooperlake", "zen4", "sapphirerapids")}
    else:
        cflags, c_std, cxx_std, define = ["-O3", "-fPIC", "-fvisibility=hidden", "-DNDEBUG"], "-std=c11", "-std=c++20", "-D"
        variants = GGML_CPU_VARIANTS
    build = {
        "include_dirs": [f"{GGML_DIR}/include", f"{GGML_DIR}/src", f"{GGML_DIR}/ggml-cpu", GGML_DIR],
        "obj_deps": {"": sorted(str(p.relative_to(ROOT)) for d in ("include", "src", "ggml-cpu", "freetoken") for p in (ROOT / GGML_DIR / d).rglob("*.h"))},
    }
    # ggml-base, built once without ISA flags like upstream's; arch/x86/quants.c calls its quantize_row_*_ref
    libraries = [("freetoken_ggml_base", {"sources": [f"{GGML_DIR}/src/ggml-quants.c"], "cflags": [*cflags, c_std], **build})]
    modules, module_libs = [], {}
    for variant, feats in variants.items():
        flags, macros = _ggml_variant_flags(feats)
        defines = [define + m for m in macros]
        libraries += [
            (f"freetoken_ggml_cpu_{variant}", {
                "sources": [f"{GGML_DIR}/ggml-cpu/quants.c", f"{GGML_DIR}/ggml-cpu/arch/x86/quants.c", f"{GGML_DIR}/freetoken/cpu_kernels.c"],
                "cflags": [*cflags, *flags, *defines, c_std],
                **build,
            }),
            # the float row kernels; C++, so a library of their own
            (f"freetoken_ggml_cpu_vec_{variant}", {"sources": [f"{GGML_DIR}/ggml-cpu/vec.cpp"], "cflags": [*cflags, *flags, *defines, cxx_std], **build}),
        ]
        name = f"freetoken.kernel.ggml_cpu._{variant}"
        module_libs[name] = [f"freetoken_ggml_cpu_{variant}", f"freetoken_ggml_cpu_vec_{variant}", "freetoken_ggml_base"]
        modules.append(Extension(
            name=name,
            sources=[f"{GGML_DIR}/freetoken/variant_module.cpp"],
            # absolute: ninja compiles from the build directory
            include_dirs=[str(ROOT / d) for d in (*build["include_dirs"], f"{GGML_DIR}/freetoken")],
            extra_compile_args=[*cflags, *defines, f"{define}FT_GGML_VARIANT={variant}", cxx_std],
        ))
    return libraries, modules, module_libs


ggml_cpu_libraries, ggml_cpu_modules, ggml_cpu_module_libs = _ggml_cpu_targets()


class BuildClib(build_clib):
    def build_libraries(self, libraries):
        # the variants compile the same sources with different flags, so each library gets its own object directory
        build_temp = self.build_temp
        try:
            for lib in libraries:
                self.build_temp = os.path.join(build_temp, lib[0])
                super().build_libraries([lib])
        finally:
            self.build_temp = build_temp


class BuildExt(BuildExtension):
    def run(self):
        # build_ext alone (`setup.py build_ext --inplace`) does not run build_clib
        if self.distribution.has_c_libraries():
            self.run_command("build_clib")
        # build_ext would link every build_clib library into every extension; build_extension adds each module's own
        libraries, self.distribution.libraries = self.distribution.libraries, None
        try:
            super().run()
        finally:
            self.distribution.libraries = libraries

    def build_extension(self, ext):
        if ext.name in ggml_cpu_module_libs:
            build_clib_dir = self.get_finalized_command("build_clib").build_clib
            if build_clib_dir not in ext.library_dirs:
                ext.library_dirs.append(build_clib_dir)
            for name in ggml_cpu_module_libs[ext.name]:
                if name not in ext.libraries:
                    ext.libraries.append(name)
                lib = os.path.join(build_clib_dir, self.compiler.library_filename(name))
                # relink when only a static library changed
                if lib not in ext.depends:
                    ext.depends.append(lib)
            # every variant module compiles the same source, so each gets its own object directory
            build_temp, self.build_temp = self.build_temp, os.path.join(self.build_temp, ext.name)
            try:
                super().build_extension(ext)
            finally:
                self.build_temp = build_temp
            return
        super().build_extension(ext)


setup(
    libraries=ggml_cpu_libraries,
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs],
            library_dirs=runtime_library_dirs,
            libraries=[runtime_lib],
            extra_compile_args=extra_compile,
            extra_link_args=runtime_link_args,
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
                "python/freetoken/kernel/csrc/cpu_moe/locked_fetch.cpp",
                "python/freetoken/kernel/csrc/cpu_moe/locked_fetch_io.cu",
                f"python/freetoken/kernel/csrc/cpu_moe/gpu_sync_{'hip' if IS_ROCM else 'cuda'}.cpp",
                "python/freetoken/kernel/csrc/cpu_moe/gpu_sync_io.cu",
                *sorted(
                    str(p.relative_to(ROOT))
                    for p in (ROOT / "python/freetoken/kernel/csrc/cpu_moe/kernels").glob("*.cpp")
                ),
            ],
            include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs, str(ROOT / GGML_DIR / "freetoken")],
            library_dirs=runtime_library_dirs,
            libraries=[runtime_lib],
            extra_compile_args={"cxx": extra_compile + ["-pthread"], "nvcc": device_compile},
            extra_link_args=runtime_link_args,
        ),
        # disk-backed row store (PLE / Engram tables); Linux-only until the TableFile/BatchReader seams grow Windows bodies
        *([
            CppExtension(
                name="freetoken.kernel._row_store",
                sources=[
                    "python/freetoken/kernel/csrc/row_store/row_store_ext.cpp",
                ],
                extra_compile_args=["-O3", "-std=c++20"],
            )
        ] if sys.platform == "linux" else []),
        *ggml_cpu_modules,
    ],
    cmdclass={"build_clib": BuildClib, "build_ext": BuildExt.with_options(use_ninja=True)},
)
