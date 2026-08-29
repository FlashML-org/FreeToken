#!/usr/bin/env bash
# Run the isolated LAN-223 Qwen scheduler workload with a temporary GPU DPM policy.
#
# This wrapper exists because the normal scheduler harness deliberately refuses an
# already-existing artifact directory, whereas policy telemetry must be written
# before the harness begins.  It therefore creates one parent evidence directory
# and reserves a new, non-existent `benchmark` child for the harness itself.
#
# The script changes only GPU DPM policy for the duration of its own process.
# Its EXIT trap restores the requested prior policy even if the benchmark fails.
# It neither starts nor stops FreeToken, touches llama-swap, nor contacts a host
# other than LAN-223's local API endpoint through the delegated harness.

set -euo pipefail

# Require the desired temporary policy explicitly so accidental invocation cannot
# silently change the GPU to an unintended policy level.
readonly TEMPORARY_POLICY="${1:?usage: run_qwen_dpm_policy_benchmark.sh POLICY [ARTIFACT_ROOT]}"

# Store preflight and restoration telemetry in a unique parent directory.  The
# second argument permits a caller to choose an immutable evidence location.
readonly ARTIFACT_ROOT="${2:-/home/david/freetoken-amd/artifacts/qwen-dpm-${TEMPORARY_POLICY}-$(date -u +%Y%m%dT%H%M%SZ)}"

# Keep the benchmark child absent.  run_qwen_scheduler_baseline.sh delegates to
# a Python harness that creates this directory atomically to prevent artifact
# collisions and preserve evidence integrity.
readonly BENCHMARK_DIR="${ARTIFACT_ROOT}/benchmark"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly HARNESS="${ROOT_DIR}/source-qwen-harness-d6ee8ce/scripts/lan223/run_qwen_scheduler_baseline.sh"
readonly POLICY_LOG="${ARTIFACT_ROOT}/dpm-policy.txt"

# Fail before a policy change if an operator supplied a reused artifact root.
if [[ -e "${ARTIFACT_ROOT}" ]]; then
    echo "error: artifact root already exists: ${ARTIFACT_ROOT}" >&2
    exit 2
fi

mkdir -p "${ARTIFACT_ROOT}"

# Restore the safe default policy and append post-run telemetry.  Each command
# is best-effort so a benchmark failure cannot conceal the restoration attempt.
restore_policy() {
    sudo rocm-smi --setperflevel auto || true
    rocm-smi --showperflevel | tee -a "${POLICY_LOG}" || true
}
trap restore_policy EXIT

# Apply and record the requested policy before the warm benchmark begins.
sudo rocm-smi --setperflevel "${TEMPORARY_POLICY}"
rocm-smi --showperflevel | tee "${POLICY_LOG}"

# Pass the guaranteed-absent child path to the existing fixed scheduler workload.
bash "${HARNESS}" "${BENCHMARK_DIR}"
