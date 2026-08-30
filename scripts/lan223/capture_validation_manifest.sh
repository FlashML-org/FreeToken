#!/usr/bin/env bash
# Capture a read-only, secret-safe LAN-223 runtime manifest for one test run.
#
# The collector never starts or stops a model server.  It creates a new artifact
# directory, records only operational metadata needed to reproduce a benchmark,
# and deliberately avoids shell environment dumps that could contain secrets.

set -euo pipefail

# Require a caller-owned, not-yet-existing artifact location so an old result is
# never silently replaced by a later run.
readonly ARTIFACT_DIR="${1:?usage: capture_validation_manifest.sh ARTIFACT_DIR [EXPECTED_HOST]}"
readonly EXPECTED_HOST="${2:-david-Gmktec-x2-2}"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly SOURCE_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"

# The program runs only where this validation program is authorized.  A caller
# may pass the exact hostname deliberately, but a mismatched host fails closed.
readonly ACTUAL_HOST="$(hostname -s)"
if [[ "${ACTUAL_HOST,,}" != "${EXPECTED_HOST,,}" ]]; then
    echo "refusing manifest on ${ACTUAL_HOST}; expected ${EXPECTED_HOST}" >&2
    exit 2
fi
if [[ -e "${ARTIFACT_DIR}" ]]; then
    echo "refusing to overwrite existing artifact: ${ARTIFACT_DIR}" >&2
    exit 3
fi
test -d "${SOURCE_DIR}"
mkdir -p "${ARTIFACT_DIR}"

# Record stable operating-system and source provenance without modifying either.
{
    printf 'captured_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "${ACTUAL_HOST}"
    uname -a
    test -r /etc/os-release && cat /etc/os-release
} >"${ARTIFACT_DIR}/system.txt"
{
    git -C "${SOURCE_DIR}" rev-parse HEAD
    git -C "${SOURCE_DIR}" branch --show-current || true
    git -C "${SOURCE_DIR}" status --short
    git -C "${SOURCE_DIR}" diff --stat
} >"${ARTIFACT_DIR}/source-state.txt"

# Record the installed ROCm/HIP tools and live GPU policy separately so users
# can see if a later policy change altered a performance result.
{
    command -v rocminfo || true
    rocminfo 2>/dev/null || true
} >"${ARTIFACT_DIR}/rocminfo.txt"
{
    command -v rocm-smi || true
    rocm-smi --showproductname --showtemp --showperflevel --showmeminfo vram 2>&1 || true
} >"${ARTIFACT_DIR}/rocm-smi.txt"

# Memory, swap, mounted capacity, and process state explain timing outliers but
# are only observed.  The script does not clear caches, disable swap, or adjust
# clocks because those are separate reviewed actions.
{
    free -b
    swapon --show --bytes || true
    vmstat 1 3
    df -B1 / "${ROOT_DIR}"
} >"${ARTIFACT_DIR}/memory-and-storage.txt"
ps -eo pid,ppid,rss,vsz,stat,etimes,cmd --sort=-rss >"${ARTIFACT_DIR}/processes.txt"

# The manifest itself describes the collector contract and points to every raw
# component.  It intentionally stores paths, not a second lossy copy of data.
cat >"${ARTIFACT_DIR}/manifest.json" <<EOF
{
  "schema_version": 1,
  "host": "${ACTUAL_HOST}",
  "source_dir": "${SOURCE_DIR}",
  "files": [
    "system.txt",
    "source-state.txt",
    "rocminfo.txt",
    "rocm-smi.txt",
    "memory-and-storage.txt",
    "processes.txt"
  ],
  "collection": "read-only; no service, cache, clock, memory, or swap mutation"
}
EOF

printf '%s\n' "${ARTIFACT_DIR}"

