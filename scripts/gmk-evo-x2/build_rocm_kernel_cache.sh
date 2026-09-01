#!/usr/bin/env bash
# Build a reusable native ROCm kernel cache for FreeToken on GMKtec EVO-X2.
#
# FreeToken's C++/HIP helper kernels normally compile on their first matching
# call when no prebuilt cache is configured. This builder compiles the complete
# explicit model-shape catalog once for the exact source revision and writes the
# resulting shared objects into an immutable, gfx1151-specific directory. A
# subsequent server can set FREETOKEN_KERNEL_CACHE_DIR to that directory and
# FREETOKEN_DISABLE_JIT=1 to make missing coverage fail loudly instead of
# compiling during a request.
#
# The script changes only the dedicated cache root beneath freetoken-amd. It
# never starts or stops a model service, changes llama-swap, or modifies any
# production llama.cpp process.

set -euo pipefail

# Keep the host-specific locations explicit so cache provenance is easy to
# inspect after an upgrade. Callers may override ROOT_DIR for an isolated test
# checkout but must not point it at an unrelated installation.
readonly ROOT_DIR="${FREETOKEN_ROOT_DIR:-/home/david/freetoken-amd}"
readonly SOURCE_DIR="${FREETOKEN_SOURCE_DIR:-${ROOT_DIR}/source-qwen-harness-d6ee8ce}"
readonly VENV_PYTHON="${FREETOKEN_VENV_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
readonly ROCM_ROOT="${ROCM_PATH:-/opt/rocm-10.0}"

# A source revision is part of the artifact name. Reusing a cache built from a
# different commit risks loading a stale ABI after a kernel source edit.
readonly SOURCE_REVISION="$(git -C "${SOURCE_DIR}" rev-parse --short=12 HEAD)"
readonly CACHE_DIR="${FREETOKEN_ROCM_KERNEL_CACHE_DIR:-${ROOT_DIR}/cache/kernel-cache-rocm-gfx1151-${SOURCE_REVISION}}"
readonly BUILD_DIR="${FREETOKEN_ROCM_KERNEL_BUILD_DIR:-${ROOT_DIR}/cache/kernel-build-rocm-gfx1151-${SOURCE_REVISION}}"

test -x "${VENV_PYTHON}"
test -d "${SOURCE_DIR}"
test -d "${ROCM_ROOT}"

# Four parallel compilers are deliberate. The 82-module cache benefits from
# concurrency, while a much larger default fanout can contend with the shared
# memory available to the live model service.
export FREETOKEN_KERNEL_CACHE_JOBS="${FREETOKEN_KERNEL_CACHE_JOBS:-4}"
export PYTHONPATH="${SOURCE_DIR}/python"
export ROCM_PATH="${ROCM_ROOT}"
export ROCM_HOME="${ROCM_ROOT}"
export HIP_PATH="${ROCM_ROOT}"

cd "${SOURCE_DIR}"

# Compile from source even if a caller's environment names a previous cache.
# compile_and_package_kernels internally restores these settings after it has
# copied each shared object into CACHE_DIR.
"${VENV_PYTHON}" - "${CACHE_DIR}" "${BUILD_DIR}" <<'PY'
"""Compile the exact FreeToken C++/HIP cache and print auditable metadata."""

from __future__ import annotations

import json
import pathlib
import sys

import torch

from freetoken.kernel.aot import compile_and_package_kernels, default_kernel_specs

cache_dir = pathlib.Path(sys.argv[1])
build_dir = pathlib.Path(sys.argv[2])

if torch.version.hip is None:
    raise SystemExit("refusing to build a ROCm cache with a non-HIP PyTorch runtime")
if "gfx1151" not in torch.cuda.get_device_name().lower() and "8060" not in torch.cuda.get_device_name().lower():
    raise SystemExit(f"refusing non-GMKtec EVO-X2 GPU: {torch.cuda.get_device_name()}")

specs = default_kernel_specs()
paths = compile_and_package_kernels(
    out_dir=cache_dir,
    build_dir=build_dir,
    specs=specs,
    clean=False,
    verbose=True,
)

print(
    json.dumps(
        {
            "cache_dir": str(cache_dir),
            "compiled_modules": len(paths),
            "device": torch.cuda.get_device_name(),
            "hip": torch.version.hip,
            "spec_count": len(specs),
        },
        sort_keys=True,
    )
)
PY
