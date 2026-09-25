#!/usr/bin/env bash
# Verify that the explicit accelerator extras resolve independently for the
# Linux x86_64 / Python 3.12 wheel target without building or installing.
#
# The script copies only pyproject.toml into a fresh disposable directory and
# compiles requirements files there. It never installs packages, builds
# FreeToken, modifies a service, or writes resolver output in the source tree.

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR=""

cleanup() {
    if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
        rm -rf -- "${WORK_DIR}"
    fi
}

trap cleanup EXIT

if ! command -v uv >/dev/null 2>&1; then
    printf 'error: uv is required; install uv before running this resolver check\n' >&2
    exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/freetoken-accel-resolver.XXXXXX")"
cp "${SOURCE_DIR}/pyproject.toml" "${WORK_DIR}/pyproject.toml"
PYTHON_VERSION="${RESOLVER_PYTHON_VERSION:-3.12}"
PYTHON_PLATFORM="${RESOLVER_PYTHON_PLATFORM:-x86_64-manylinux_2_28}"
read -r -a ROCM_PYTHON_VERSIONS <<< \
    "${RESOLVER_ROCM_PYTHON_VERSIONS:-3.10 3.11 3.12 3.13 3.14}"

resolve() {
    local extra="$1"
    local python_version="${2:-${PYTHON_VERSION}}"
    printf '== resolving %s for Python %s ==\n' "${extra}" "${python_version}"
    local output="${WORK_DIR}/${extra}-${python_version}.txt"
    uv pip compile "${WORK_DIR}/pyproject.toml" \
        --extra "${extra}" \
        --python-version "${python_version}" \
        --python-platform "${PYTHON_PLATFORM}" \
        --output-file "${output}" \
        --no-cache > /dev/null
    grep -E '^(flashinfer-python|sglang-kernel|torch==|torchvision==|triton==|rocm-sdk-(core|libraries|devel|device-gfx1151)==)' \
        "${output}" || true
}

for version in "${ROCM_PYTHON_VERSIONS[@]}"; do
    resolve rocm "${version}"
    if ! grep -q '^triton==3\.7\.1+git0263a6a6\.rocm7\.14\.0' \
        "${WORK_DIR}/rocm-${version}.txt"; then
        printf 'error: ROCm did not resolve AMD PyTorch 7.14.0 Triton for Python %s\n' \
            "${version}" >&2
        exit 1
    fi
    if ! grep -q '^rocm-sdk-devel==7\.14\.0' \
        "${WORK_DIR}/rocm-${version}.txt"; then
        printf 'error: ROCm did not resolve the 7.14.0 development SDK for Python %s\n' \
            "${version}" >&2
        exit 1
    fi
done

resolve cuda
resolve accel

printf '== rejecting rocm plus legacy CUDA extra ==\n'
if uv pip compile "${WORK_DIR}/pyproject.toml" \
    --extra rocm \
    --extra accel \
    --python-version "${PYTHON_VERSION}" \
    --python-platform "${PYTHON_PLATFORM}" \
    --output-file "${WORK_DIR}/rocm-accel.txt" \
    --no-cache > /dev/null; then
    printf 'error: ROCm and accel resolved together; the declared conflict is missing\n' >&2
    exit 1
fi

printf 'accelerator resolver contract passed\n'
