from __future__ import annotations

import concurrent.futures
import contextlib
import os
import pathlib
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

from .aot_models import (
    aggregate_fast_index_copy_feature_sizes,
    aggregate_index_variants,
    aggregate_store_element_sizes,
)
from .utils import (
    DISABLE_JIT_ENV,
    DISABLE_KERNEL_CACHE_ENV,
    _make_name,
    load_aot,
    load_jit,
    make_cpp_args,
)

BuildFn = Callable[[pathlib.Path], object]

# Shape coverage is generated from the explicit per-model support table in
# kernel/aot_models.py; add new models/formats there, not here.
DEFAULT_STORE_ELEMENT_SIZES = aggregate_store_element_sizes()
DEFAULT_INDEX_VARIANTS = aggregate_index_variants()
# Closed, model-independent set: max_k = next_pow2(top_k) capped at
# ENSURE_MAX_K_CAP=16, so these four buckets cover any top_k in [1, 16].
DEFAULT_FAST_INDEX_COPY_FEATURE_SIZES = aggregate_fast_index_copy_feature_sizes()


@dataclass(frozen=True)
class KernelSpec:
    name: str
    build: BuildFn


def _store_spec(element_size: int) -> KernelSpec:
    from .store import DEFAULT_INDEX_KERNEL_CONFIG

    args = make_cpp_args(element_size, *DEFAULT_INDEX_KERNEL_CONFIG)

    def build(build_directory: pathlib.Path) -> object:
        return load_jit(
            "store",
            *args,
            cuda_files=["store.cu"],
            cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
            build_directory=str(build_directory),
        )

    return KernelSpec(name=_make_name("store", *args), build=build)


def _index_spec(element_size: int, num_splits: int) -> KernelSpec:
    from .index import DEFAULT_INDEX_KERNEL_CONFIG

    args = make_cpp_args(element_size, num_splits, *DEFAULT_INDEX_KERNEL_CONFIG)

    def build(build_directory: pathlib.Path) -> object:
        return load_jit(
            "index",
            *args,
            cuda_files=["index.cu"],
            cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
            build_directory=str(build_directory),
        )

    return KernelSpec(name=_make_name("index", *args), build=build)



def _fast_index_copy_spec(feature_size: int) -> KernelSpec:
    from .fast_index_copy import default_worker_args, legacy_fast_index_copy_is_supported

    worker_threads, worker_feature_size, num_block = default_worker_args(feature_size)
    if not legacy_fast_index_copy_is_supported(feature_size):
        raise ValueError(
            "legacy fast_index_copy requires a whole 128-byte worker row; "
            f"feature_size={feature_size} resolves to {worker_feature_size} bytes"
        )
    args = make_cpp_args(feature_size, worker_threads, worker_feature_size, 1024, num_block, 1)

    def build(build_directory: pathlib.Path) -> object:
        return load_jit(
            "fast_index_copy",
            *args,
            cuda_files=["fast_index_copy.cuh"],
            cuda_wrappers=[
                ("launch", f"&FastIndexCopyKernel<{args}>::run"),
                ("launch_high", f"&FastIndexCopyKernel<{args}>::run_high"),
                ("launch_normal", f"&FastIndexCopyKernel<{args}>::run_normal"),
            ],
            build_directory=str(build_directory),
        )

    return KernelSpec(name=_make_name("fast_index_copy", *args), build=build)


def _fast_index_copy_update_flag_spec() -> KernelSpec:
    def build(build_directory: pathlib.Path) -> object:
        return load_jit(
            "fast_index_copy_update_flag",
            cuda_files=["fast_index_copy.cuh"],
            cuda_wrappers=[("update_copy_flag", "update_copy_flag")],
            build_directory=str(build_directory),
        )

    return KernelSpec(name=_make_name("fast_index_copy_update_flag"), build=build)


def _fast_index_copy_multi_spec(num_threads: int, blocks_per_bank: int) -> KernelSpec:
    args = make_cpp_args(num_threads, blocks_per_bank)

    def build(build_directory: pathlib.Path) -> object:
        return load_jit(
            "fast_index_copy_multi",
            *args,
            cuda_files=["fast_index_copy.cuh"],
            cuda_wrappers=[("launch", f"&MultiIndexCopyKernel<{args}>::run")],
            build_directory=str(build_directory),
        )

    return KernelSpec(name=_make_name("fast_index_copy_multi", *args), build=build)


def _batch_memcpy_spec() -> KernelSpec:
    def build(build_directory: pathlib.Path) -> object:
        return load_jit(
            "batch_memcpy",
            cuda_files=["batch_memcpy.cuh"],
            cuda_wrappers=[("batch_memcpy", "&BatchMemcpy::run")],
            build_directory=str(build_directory),
        )

    return KernelSpec(name=_make_name("batch_memcpy"), build=build)


def _radix_spec() -> KernelSpec:
    def build(build_directory: pathlib.Path) -> object:
        return load_aot("radix", cpp_files=["radix.cpp"], build_directory=str(build_directory))

    return KernelSpec(name=_make_name("radix"), build=build)


def default_kernel_specs() -> tuple[KernelSpec, ...]:
    # Import lazily with the other kernel builders. This keeps importing the
    # catalog inexpensive while sharing the validity rule with the runtime
    # argument derivation rather than duplicating a 128-byte magic number.
    from .fast_index_copy import legacy_fast_index_copy_is_supported

    specs: list[KernelSpec] = []
    specs.extend(_store_spec(element_size) for element_size in DEFAULT_STORE_ELEMENT_SIZES)
    specs.extend(_index_spec(*variant) for variant in DEFAULT_INDEX_VARIANTS)
    specs.append(_fast_index_copy_update_flag_spec())
    specs.extend(
        _fast_index_copy_spec(feature_size)
        for feature_size in DEFAULT_FAST_INDEX_COPY_FEATURE_SIZES
        # The fused multi-bank cache path supports these small rows directly.
        # Do not emit legacy per-bank templates that cannot satisfy their own
        # 128-byte vector-loop static assertion.
        if legacy_fast_index_copy_is_supported(feature_size)
    )
    specs.append(_fast_index_copy_multi_spec(num_threads=1024, blocks_per_bank=8))
    # prefill hit-D2D gather (HBM-bound: wide grid) + its miss-side batch H2D binding.
    specs.append(_fast_index_copy_multi_spec(num_threads=1024, blocks_per_bank=64))
    specs.append(_batch_memcpy_spec())
    specs.append(_radix_spec())

    names = [spec.name for spec in specs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise RuntimeError(f"Duplicate kernel cache specs: {sorted(duplicates)}")
    return tuple(specs)


@contextlib.contextmanager
def _force_source_build() -> Iterator[None]:
    old_disable_cache = os.environ.get(DISABLE_KERNEL_CACHE_ENV)
    old_disable_jit = os.environ.get(DISABLE_JIT_ENV)
    os.environ[DISABLE_KERNEL_CACHE_ENV] = "1"
    os.environ.pop(DISABLE_JIT_ENV, None)
    try:
        yield
    finally:
        if old_disable_cache is None:
            os.environ.pop(DISABLE_KERNEL_CACHE_ENV, None)
        else:
            os.environ[DISABLE_KERNEL_CACHE_ENV] = old_disable_cache
        if old_disable_jit is None:
            os.environ.pop(DISABLE_JIT_ENV, None)
        else:
            os.environ[DISABLE_JIT_ENV] = old_disable_jit


def _resolve_specs(specs: Iterable[KernelSpec | str] | None) -> tuple[KernelSpec, ...]:
    default_specs = default_kernel_specs()
    if specs is None:
        return default_specs

    by_name = {spec.name: spec for spec in default_specs}
    resolved: list[KernelSpec] = []
    for spec in specs:
        if isinstance(spec, KernelSpec):
            resolved.append(spec)
            continue
        try:
            resolved.append(by_name[spec])
        except KeyError as exc:
            raise KeyError(f"Unknown default kernel spec {spec!r}") from exc
    return tuple(resolved)


def _built_shared_object(build_dir: pathlib.Path, name: str) -> pathlib.Path:
    candidates = [
        path
        for path in build_dir.rglob(f"{name}.so")
        if path.parent.name == name or path.parent.name.startswith(f"{name}_")
    ]
    if not candidates:
        raise RuntimeError(f"Compiled kernel {name!r} was not found under {build_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _copy_kernel(build_dir: pathlib.Path, out_dir: pathlib.Path, name: str) -> pathlib.Path:
    src = _built_shared_object(build_dir, name)
    dst_dir = out_dir / name
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{name}.so"
    shutil.copy2(src, dst)
    return dst


KERNEL_CACHE_JOBS_ENV = "FREETOKEN_KERNEL_CACHE_JOBS"


def _default_jobs(num_specs: int) -> int:
    raw = os.getenv(KERNEL_CACHE_JOBS_ENV, "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, min(num_specs, os.cpu_count() or 1))


def compile_and_package_kernels(
    *,
    out_dir: str | os.PathLike[str],
    build_dir: str | os.PathLike[str],
    specs: Iterable[KernelSpec | str] | None = None,
    clean: bool = True,
    verbose: bool = False,
    jobs: int | None = None,
) -> list[pathlib.Path]:
    out_path = pathlib.Path(out_dir)
    build_path = pathlib.Path(build_dir)
    selected_specs = _resolve_specs(specs)
    if jobs is None:
        jobs = _default_jobs(len(selected_specs))

    if clean and out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    build_path.mkdir(parents=True, exist_ok=True)

    def _build_one(spec: KernelSpec) -> pathlib.Path:
        spec_build_dir = build_path / spec.name
        if clean and spec_build_dir.exists():
            shutil.rmtree(spec_build_dir)
        spec_build_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Building {spec.name}", flush=True)
        spec.build(spec_build_dir)
        return _copy_kernel(spec_build_dir, out_path, spec.name)

    # Specs are independent (each compiles in its own build directory), so they can
    # build concurrently; the compilers run as subprocesses, so threads suffice.
    with _force_source_build():
        if jobs <= 1:
            copied = [_build_one(spec) for spec in selected_specs]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                copied = list(pool.map(_build_one, selected_specs))

    if not copied:
        raise RuntimeError("No kernels were packaged")
    return copied


__all__ = [
    "KernelSpec",
    "compile_and_package_kernels",
    "default_kernel_specs",
]
