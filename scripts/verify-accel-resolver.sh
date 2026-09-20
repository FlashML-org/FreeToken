#!/usr/bin/env bash
# Verify that the explicit accelerator extras resolve independently.
#
# The script copies only pyproject.toml into a fresh disposable directory. It
# never installs packages, builds FreeToken, modifies a service, or writes a
# lockfile in the source checkout.

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

resolve() {
    local extra="$1"
    printf '== resolving %s ==\n' "${extra}"
    uv sync --directory "${WORK_DIR}" \
        --extra "${extra}" \
        --no-install-project \
        --dry-run \
        --no-cache
}

resolve rocm
resolve cuda
resolve accel

printf '== rejecting rocm plus legacy CUDA extra ==\n'
if uv sync --directory "${WORK_DIR}" \
    --extra rocm \
    --extra accel \
    --no-install-project \
    --dry-run \
    --no-cache; then
    printf 'error: ROCm and accel resolved together; the declared conflict is missing\n' >&2
    exit 1
fi

printf 'accelerator resolver contract passed\n'
