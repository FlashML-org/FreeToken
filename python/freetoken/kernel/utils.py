from __future__ import annotations

import contextlib
import importlib
import os
import pathlib
import re
from functools import cache
from typing import TYPE_CHECKING, Iterator, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
KERNEL_CACHE_PACKAGE = "freetoken_kernel_cache"
KERNEL_CACHE_DIR_ENV = "FREETOKEN_KERNEL_CACHE_DIR"
DISABLE_KERNEL_CACHE_ENV = "FREETOKEN_DISABLE_KERNEL_CACHE"
DISABLE_KERNEL_CACHE_VERSION_CHECK_ENV = "FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK"
DISABLE_JIT_ENV = "FREETOKEN_DISABLE_JIT"
_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_HIP_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS = []


CUDA_ARCH_LIST_ENV = "TVM_FFI_CUDA_ARCH_LIST"
ROCM_ARCH_LIST_ENV = "TVM_FFI_ROCM_ARCH_LIST"


def _is_rocm() -> bool:
    import torch

    return getattr(torch.version, "hip", None) is not None


def _cuda_arch_list() -> List[str]:
    """Archs a CUDA build targets: the AOT build's TVM_FFI_CUDA_ARCH_LIST, else the GPU this process is bound to."""
    arch_list = os.getenv(CUDA_ARCH_LIST_ENV, "").split()
    if arch_list:
        return arch_list
    import torch

    if torch.version.hip is not None or not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    return [f"{major}.{minor}"]


def _rocm_arch_list() -> List[str]:
    """The single ROCm GPU this process is bound to, or an explicit cross-compile target."""
    from freetoken.utils.arch import get_rocm_gfx_arch

    arch = get_rocm_gfx_arch()
    if arch is None:
        raise RuntimeError(
            "Could not determine the bound ROCm GPU architecture; select a visible GPU "
            "or set FREETOKEN_ROCM_ARCH for cross-compilation"
        )
    return [arch]


@contextlib.contextmanager
def _pin_tvm_ffi_arch_ctx(arch_list: List[str], env_var: str) -> Iterator[None]:
    """Pin tvm-ffi to the resolved targets for one build, then restore the caller's environment."""
    if not arch_list:
        yield
        return
    old_arch_list = os.environ.get(env_var)
    os.environ[env_var] = " ".join(arch_list)
    try:
        yield
    finally:
        if old_arch_list is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old_arch_list


def _cuda_cflags(extra: List[str], arch_list: List[str]) -> List[str]:
    """CUDA nvcc flags for a kernel build. tvm-ffi emits one SASS cubin per arch in ``arch_list`` and no PTX, so add the PTX of the highest arch: a GPU newer than every listed arch still runs through the driver's PTX JIT. This flag also carries the arch into tvm-ffi's build hash, which skips tvm-ffi's own -gencode, so GPUs of different archs never share a cached .so."""
    import torch

    flags = list(DEFAULT_CUDA_CFLAGS)
    if torch.version.hip is not None:
        # nvcc-only: hipcc/clang rejects it outright.
        flags = [f for f in flags if f != "--expt-relaxed-constexpr"]
    flags = flags + extra
    if arch_list:
        def _rank(a: str) -> int:
            major, minor = a.rstrip("a").split(".")
            return int(major) * 100 + int(minor)

        cc = max(arch_list, key=_rank).rstrip("a").replace(".", "")
        flags = flags + [f"-gencode=arch=compute_{cc},code=compute_{cc}"]
    return flags


def _hip_cflags(extra: List[str], arch_list: List[str]) -> List[str]:
    """HIP flags for a kernel build on ROCm."""
    # TODO(ROCm): Triton autotune configs need RDNA-specific tuning (wave count, LDS size).
    return DEFAULT_HIP_CFLAGS + extra + [f"--offload-arch={arch}" for arch in arch_list]


@cache
def _rocm_link_flags() -> List[str]:
    """Make ROCm's runtime library discoverable to JIT link commands.

    Traditional ROCm installs provide ``libamdhip64.so`` under ``$ROCM_HOME/lib``.
    Some Python SDK layouts provide only the versioned soname, while TVM-FFI
    still links with ``-lamdhip64``. Supply a cache-local unversioned symlink via
    an explicit linker search path without modifying the selected ROCm environment.
    """
    candidates: list[pathlib.Path] = []
    if os.getenv("ROCM_HOME"):
        candidates.append(pathlib.Path(os.environ["ROCM_HOME"]))
    try:
        from torch.utils.cpp_extension import ROCM_HOME

        if ROCM_HOME:
            candidates.append(pathlib.Path(ROCM_HOME))
    except ImportError:
        pass
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec and spec.submodule_search_locations:
        candidates.append(pathlib.Path(next(iter(spec.submodule_search_locations))))
    candidates.append(pathlib.Path("/opt/rocm"))

    def soname_version(path: pathlib.Path) -> tuple[int, ...]:  # Compare numeric SONAME components rather than lexical filenames.
        suffix = path.name.partition(".so.")[2]  # Isolate the version suffix after the shared-library marker.
        return tuple(int(part) if part.isdigit() else -1 for part in suffix.split("."))  # Rank each numeric component independently.

    for rocm_home in dict.fromkeys(candidates):  # Preserve the explicit-root priority while removing duplicate paths.
        for library_dir in (rocm_home / "lib64", rocm_home / "lib"):  # Support both standard ROCm library layouts.
            unversioned = library_dir / "libamdhip64.so"  # Prefer the SDK-provided linker name when present.
            link_dir = library_dir  # Link directly against a complete SDK layout by default.
            if not unversioned.exists():  # Create a private compatibility name only for versioned-only SDK layouts.
                versioned = list(library_dir.glob("libamdhip64.so.*"))  # Collect every runtime SONAME supplied by this root.
                if not versioned:  # Try the next library layout when this directory has no HIP runtime.
                    continue  # Keep discovery bounded to the declared ROCm roots.
                cache_root = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"))  # Honor an explicit writable cache filesystem.
                link_dir = cache_root / "freetoken" / "rocm-lib"  # Keep compatibility state outside the SDK and full home filesystems.
                link_dir.mkdir(parents=True, exist_ok=True)  # Prepare the private linker directory idempotently.
                compat_link = link_dir / "libamdhip64.so"  # Provide the unversioned name expected by TVM-FFI.
                selected_runtime = max(versioned, key=soname_version).resolve()  # Select the greatest numeric SONAME from this SDK.
                if compat_link.exists() and not compat_link.is_symlink():  # Never replace an unexpected user-created regular file.
                    raise RuntimeError(f"Refusing to replace non-symlink ROCm compatibility file: {compat_link}")  # Fail with an actionable cache path.
                current_runtime = None  # Treat a missing or broken compatibility link as stale.
                if compat_link.is_symlink():  # Inspect both valid and broken links left by an earlier ROCm installation.
                    try:  # Resolve the complete target so relative and absolute links compare consistently.
                        current_runtime = compat_link.resolve(strict=True)  # Record the runtime currently selected by the cache.
                    except FileNotFoundError:  # A removed ROCm stack leaves a broken link that must be refreshed.
                        pass  # Keep the stale marker as None for the replacement branch.
                if current_runtime != selected_runtime:  # Refresh links that point at a different or removed ROCm runtime.
                    replacement = link_dir / f".libamdhip64.so.{os.getpid()}.tmp"  # Give each concurrent builder a process-local staging link.
                    replacement.unlink(missing_ok=True)  # Remove only this process's abandoned staging path from an interrupted attempt.
                    try:  # Publish the new link atomically so parallel ranks never observe a missing final path.
                        replacement.symlink_to(selected_runtime)  # Prepare a link to the selected runtime without changing the shared name yet.
                        replacement.replace(compat_link)  # Atomically replace the old or broken compatibility symlink.
                    finally:  # Reclaim a staging link if publication failed before the atomic replacement.
                        replacement.unlink(missing_ok=True)  # Leave no process-specific cache artifact behind.

            return [f"-L{link_dir}", f"-Wl,-rpath,{library_dir}"]  # Bind this build to the selected SDK library directory.

    raise RuntimeError("Unable to locate libamdhip64 for ROCm JIT linking")


CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


class CppArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: str) -> str:
    return "freetoken__" + "_".join(str(arg) for arg in args)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def _freetoken_version() -> str:
    from freetoken.version import __version__

    return __version__


def _version_parts(version: str) -> Tuple[str, List[str]]:
    """Split a PEP 440 version string into (release, local segments):
    "0.1.1+cu130.g3f01615" -> ("0.1.1", ["cu130", "g3f01615"])."""
    base, _, local = version.partition("+")
    return base, local.split(".") if local else []


def _build_stamps(segments: List[str]) -> set[str]:
    """The `g<sha>` commit-stamp tokens of a local version segment list
    (stamped by scripts/build-release-wheels.sh)."""
    return {s for s in segments if re.fullmatch(r"g[0-9a-f]{7,40}", s)}


def _kernel_cache_version_ok(cache_version: str, runtime_version: str) -> bool:
    """Same release -- and, when both sides carry a `g<sha>` stamp, the same build.

    The cache wheel extends the runtime's version with local segments (`+cu130`,
    `.g<sha>`), so the old string-prefix test cannot pair a stamped runtime with its
    cache; and comparing the stamps rejects a runtime/cache pair from two different
    builds, which bare release numbers (both `0.1.1`) could never detect. Either side
    may lack a stamp (dev builds) -- then only the release part is compared."""
    cache_base, cache_local = _version_parts(cache_version)
    runtime_base, runtime_local = _version_parts(runtime_version)
    if cache_base != runtime_base:
        return False
    cache_stamps = _build_stamps(cache_local)
    runtime_stamps = _build_stamps(runtime_local)
    return not (cache_stamps and runtime_stamps and cache_stamps != runtime_stamps)


def _kernel_cache_dir() -> pathlib.Path | None:
    if _env_enabled(DISABLE_KERNEL_CACHE_ENV):
        return None

    override = os.getenv(KERNEL_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()

    try:
        package = importlib.import_module(KERNEL_CACHE_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name == KERNEL_CACHE_PACKAGE:
            return None
        raise

    package_version = str(getattr(package, "__version__", "0.0.0+unknown"))
    runtime_version = _freetoken_version()
    if not _env_enabled(DISABLE_KERNEL_CACHE_VERSION_CHECK_ENV):
        if not _kernel_cache_version_ok(package_version, runtime_version):
            raise RuntimeError(
                "freetoken-kernel-cache version "
                f"{package_version!r} does not match freetoken version {runtime_version!r}"
            )
        cache_cuda = re.search(r"\+cu(\d{2,})", package_version)
        if cache_cuda is not None:
            from freetoken.kernel._toolchain import torch_cuda_major

            cache_major = int(cache_cuda.group(1)[:-1])
            torch_major = torch_cuda_major()
            if torch_major is not None and cache_major != torch_major:
                raise RuntimeError(
                    f"freetoken-kernel-cache {package_version!r} was built for CUDA "
                    f"{cache_major}.x but torch runs CUDA {torch_major}.x -- install "
                    "the kernel-cache wheel matching this torch build"
                )

    get_jit_cache_dir = getattr(package, "get_jit_cache_dir", None)
    if get_jit_cache_dir is None:
        raise RuntimeError(f"{KERNEL_CACHE_PACKAGE} does not expose get_jit_cache_dir()")
    return pathlib.Path(get_jit_cache_dir()).expanduser()


def _load_prebuilt(name: str) -> Module | None:
    cache_dir = _kernel_cache_dir()
    if cache_dir is None:
        if _env_enabled(DISABLE_JIT_ENV):
            raise RuntimeError(
                "JIT compilation is disabled by FREETOKEN_DISABLE_JIT, "
                f"but no prebuilt kernel cache is configured for {name!r}"
            )
        return None

    so_path = cache_dir / name / f"{name}.so"
    if so_path.exists():
        import tvm_ffi

        return tvm_ffi.load_module(str(so_path))

    if _env_enabled(DISABLE_JIT_ENV):
        raise RuntimeError(
            "JIT compilation is disabled by FREETOKEN_DISABLE_JIT, "
            f"but prebuilt kernel {name!r} was not found at {so_path}"
        )
    return None


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    name = _make_name(*args)
    prebuilt = _load_prebuilt(name)
    if prebuilt is not None:
        return prebuilt

    is_rocm = _is_rocm()
    arch_list: List[str] = []
    if cuda_files:
        if is_rocm:
            arch_list = _rocm_arch_list()
        else:
            from freetoken.kernel._toolchain import check_nvcc_matches_torch

            check_nvcc_matches_torch()
            arch_list = _cuda_arch_list()

    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    if is_rocm:
        cuda_cflags = _hip_cflags(extra_cuda_cflags, arch_list)
        runtime_ldflags = _rocm_link_flags() if cuda_files else []  # Keep C++-only modules independent of HIP runtime discovery.
        arch_list_env = ROCM_ARCH_LIST_ENV
    else:
        cuda_cflags = _cuda_cflags(extra_cuda_cflags, arch_list)
        runtime_ldflags = []
        arch_list_env = CUDA_ARCH_LIST_ENV

    with _pin_tvm_ffi_arch_ctx(arch_list, arch_list_env):
        return load(
            name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=DEFAULT_LDFLAGS + runtime_ldflags + extra_ldflags,
            extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
            build_directory=build_directory,
        )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    name = _make_name(*args)
    prebuilt = _load_prebuilt(name)
    if prebuilt is not None:
        return prebuilt

    is_rocm = _is_rocm()
    arch_list: List[str] = []
    if cuda_files or cuda_wrappers:
        if is_rocm:
            arch_list = _rocm_arch_list()
        else:
            from freetoken.kernel._toolchain import check_nvcc_matches_torch

            check_nvcc_matches_torch()
            arch_list = _cuda_arch_list()

    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    if is_rocm:
        cuda_cflags = _hip_cflags(extra_cuda_cflags, arch_list)
        runtime_ldflags = _rocm_link_flags() if (cuda_files or cuda_wrappers) else []  # Link HIP only when a GPU translation unit exists.
        arch_list_env = ROCM_ARCH_LIST_ENV
    else:
        cuda_cflags = _cuda_cflags(extra_cuda_cflags, arch_list)
        runtime_ldflags = []
        arch_list_env = CUDA_ARCH_LIST_ENV

    with _pin_tvm_ffi_arch_ctx(arch_list, arch_list_env):
        return load_inline(
            name,
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=DEFAULT_LDFLAGS + runtime_ldflags + extra_ldflags,
            extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
            build_directory=build_directory,
        )
