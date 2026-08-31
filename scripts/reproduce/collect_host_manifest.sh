#!/usr/bin/env bash
# Capture a portable, non-sensitive manifest for a native ROCm/HIP benchmark.
#
# This script does not start or stop a model server, adjust clocks, clear caches,
# modify swap, or collect a shell environment. It creates one new artifact
# directory only after it has confirmed that the selected Python runtime exposes
# a native HIP PyTorch device.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: collect_host_manifest.sh --source-dir PATH --artifact-dir PATH [options]

Required:
  --source-dir PATH       Git checkout whose revision is being benchmarked.
  --artifact-dir PATH     New, absent directory for this manifest.

Options:
  --python PATH           Python executable with the target ROCm PyTorch runtime.
                          Default: python3
  --expected-gfx NAME     Require an exact AMD GCN target, for example gfx1151.
  --include-hostname      Record the host name. The default writes "redacted".
  -h, --help              Show this help text.
EOF
}

SOURCE_DIR=""
ARTIFACT_DIR=""
PYTHON_BIN="python3"
EXPECTED_GFX=""
INCLUDE_HOSTNAME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)
            SOURCE_DIR="${2:?--source-dir requires a path}"
            shift 2
            ;;
        --artifact-dir)
            ARTIFACT_DIR="${2:?--artifact-dir requires a path}"
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:?--python requires a path}"
            shift 2
            ;;
        --expected-gfx)
            EXPECTED_GFX="${2:?--expected-gfx requires a target name}"
            shift 2
            ;;
        --include-hostname)
            INCLUDE_HOSTNAME=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'error: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${SOURCE_DIR}" || -z "${ARTIFACT_DIR}" ]]; then
    printf 'error: --source-dir and --artifact-dir are required\n' >&2
    usage >&2
    exit 2
fi
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    printf 'error: source directory is not a Git checkout: %s\n' "${SOURCE_DIR}" >&2
    exit 2
fi
if [[ -e "${ARTIFACT_DIR}" ]]; then
    printf 'error: artifact directory already exists: %s\n' "${ARTIFACT_DIR}" >&2
    exit 3
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf 'error: Python executable was not found: %s\n' "${PYTHON_BIN}" >&2
    exit 4
fi

GPU_PROBE="$(${PYTHON_BIN} - "${EXPECTED_GFX}" <<'PY'
import json
import sys

expected = sys.argv[1].lower()
try:
    import torch
except Exception as error:
    raise SystemExit(f"PyTorch import failed: {error!r}")

if not torch.version.hip:
    raise SystemExit("PyTorch does not report a HIP runtime")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch HIP device is unavailable")

properties = torch.cuda.get_device_properties(0)
architecture = getattr(properties, "gcnArchName", "")
if expected and architecture.lower() != expected:
    raise SystemExit(f"GPU architecture {architecture!r} does not match required {expected!r}")

print(json.dumps({
    "torch_version": torch.__version__,
    "hip_version": torch.version.hip,
    "triton_version": __import__("triton").__version__,
    "device_name": torch.cuda.get_device_name(0),
    "gcn_architecture": architecture,
}, sort_keys=True))
PY
)"

mkdir -p "${ARTIFACT_DIR}"
trap 'printf "error: manifest capture failed; incomplete artifact retained at %s\\n" "${ARTIFACT_DIR}" >&2' ERR

if [[ "${INCLUDE_HOSTNAME}" -eq 1 ]]; then
    PUBLIC_HOSTNAME="$(hostname -s)"
else
    PUBLIC_HOSTNAME="redacted"
fi

{
    printf 'captured_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "${PUBLIC_HOSTNAME}"
    # Do not use `uname -a`: its second field is the local host name and would
    # defeat the redacted default. These explicit fields preserve the useful
    # operating-system, kernel, and architecture facts without identifying the
    # machine that produced a public reproducibility bundle.
    printf 'kernel_system=%s\n' "$(uname -s)"
    printf 'kernel_release=%s\n' "$(uname -r)"
    printf 'kernel_version=%s\n' "$(uname -v)"
    printf 'machine_architecture=%s\n' "$(uname -m)"
    test -r /etc/os-release && cat /etc/os-release
    command -v lscpu >/dev/null 2>&1 && lscpu || true
} >"${ARTIFACT_DIR}/system.txt"

{
    git -C "${SOURCE_DIR}" rev-parse HEAD
    git -C "${SOURCE_DIR}" branch --show-current || true
    git -C "${SOURCE_DIR}" status --short
    git -C "${SOURCE_DIR}" diff --stat
    git -C "${SOURCE_DIR}" remote get-url origin 2>/dev/null || true
} >"${ARTIFACT_DIR}/source-state.txt"

printf '%s\n' "${GPU_PROBE}" >"${ARTIFACT_DIR}/python-hip.json"

{
    command -v rocminfo || true
    rocminfo 2>&1 || true
} >"${ARTIFACT_DIR}/rocminfo.txt"

{
    command -v rocm-smi || true
    rocm-smi --showproductname --showtemp --showperflevel --showmeminfo vram 2>&1 || true
} >"${ARTIFACT_DIR}/rocm-smi.txt"

{
    free -b 2>&1 || true
    swapon --show --bytes 2>&1 || true
    vmstat 1 3 2>&1 || true
    df -B1 / "${SOURCE_DIR}" 2>&1 || true
    lsblk -d -o NAME,MODEL,SIZE,TRAN,ROTA 2>&1 || true
} >"${ARTIFACT_DIR}/memory-and-storage.txt"

"${PYTHON_BIN}" - "${ARTIFACT_DIR}/manifest.json" "${PUBLIC_HOSTNAME}" "${EXPECTED_GFX}" "${GPU_PROBE}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
hostname = sys.argv[2]
expected_gfx = sys.argv[3] or None
gpu = json.loads(sys.argv[4])
manifest = {
    "schema_version": 1,
    "host": hostname,
    "expected_gfx": expected_gfx,
    "native_hip": gpu,
    "files": [
        "system.txt",
        "source-state.txt",
        "python-hip.json",
        "rocminfo.txt",
        "rocm-smi.txt",
        "memory-and-storage.txt",
        "manifest.json",
        "SHA256SUMS",
    ],
    "collection": "read-only; no service, cache, clock, memory, or swap mutation",
    "privacy": "hostname is redacted by default; no shell environment, process list, serial number, or network address is collected",
}
output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(
    cd "${ARTIFACT_DIR}"
    sha256sum system.txt source-state.txt python-hip.json rocminfo.txt rocm-smi.txt \
        memory-and-storage.txt manifest.json >SHA256SUMS
)

trap - ERR
printf '%s\n' "${ARTIFACT_DIR}"
