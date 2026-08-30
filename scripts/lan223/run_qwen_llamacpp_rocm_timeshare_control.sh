#!/usr/bin/env bash
# Run the LAN-223 ROCm llama.cpp Qwen control after temporarily releasing the
# isolated FreeToken benchmark server, then recover and validate FreeToken.
#
# A 64 GB Strix Halo host cannot keep the current FreeToken NVFP4 Qwen service
# and the fully offloaded 35B Q4_K_M llama.cpp control resident at the same
# time. This wrapper measures the two servers in time-share mode. It never
# touches llama-swap, systemd, or a service outside loopback port 1919.

set -euo pipefail

# Keep the fixed LAN-223 paths explicit to prevent comparison with another
# llama.cpp build or benchmark harness revision.
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly SOURCE_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly FREETOKEN_HEALTH_URL="http://127.0.0.1:1919/health"
readonly CONTROL_SCRIPT="${SOURCE_DIR}/scripts/lan223/run_qwen_llamacpp_rocm_control.sh"
readonly RECOVERY_SCRIPT="${SOURCE_DIR}/scripts/lan223/start_qwen_recovery_server.sh"
readonly ARTIFACT_ROOT="${1:-${ROOT_DIR}/artifacts/qwen35b-llamacpp-rocm10-timeshare-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly CONTROL_ARTIFACT="${ARTIFACT_ROOT}/llamacpp-control"
readonly BEFORE_HEALTH_FILE="${ARTIFACT_ROOT}/freetoken-health-before.json"
readonly AFTER_HEALTH_FILE="${ARTIFACT_ROOT}/freetoken-health-after.json"
readonly SWAP_BEFORE_FILE="${ARTIFACT_ROOT}/swap-before.txt"
readonly SWAP_AFTER_RELEASE_FILE="${ARTIFACT_ROOT}/swap-after-release.txt"
readonly SWAP_AFTER_FILE="${ARTIFACT_ROOT}/swap-after.txt"

# Avoid overwriting evidence from an earlier run and reject partial setup before
# stopping the live benchmark server.
if [[ -e "${ARTIFACT_ROOT}" ]]; then
    echo "error: artifact root already exists: ${ARTIFACT_ROOT}" >&2
    exit 2
fi
test -x "${CONTROL_SCRIPT}"
test -x "${RECOVERY_SCRIPT}"
mkdir -p "${ARTIFACT_ROOT}"

# Locate only a process that both owns the dedicated test port and identifies
# itself as the FreeToken server. This prevents targeting an unrelated process.
find_freetoken_pid() {
    local pid command
    pid="$(ss -ltnp 'sport = :1919' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)"
    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    command="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "${command}" == *"freetoken.cli serve"* ]] || return 1
    printf '%s\n' "${pid}"
}

# Wait for HTTP health instead of accepting a listener while FreeToken loads
# native modules and model state after recovery.
wait_for_freetoken_health() {
    local attempt health_payload
    for attempt in $(seq 1 720); do
        # A loading server returns HTTP 200 before its MoE expert banks and KV
        # cache are usable. Save every latest reply for diagnostics, but accept
        # recovery only when the API explicitly reports the serving state.
        health_payload="$(curl -fsS "${FREETOKEN_HEALTH_URL}" 2>/dev/null || true)"
        printf '%s\n' "${health_payload}" >"${AFTER_HEALTH_FILE}"
        if [[ "${health_payload}" == *'"status":"ok"'* ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Preserve the configured swap file while clearing pages faulted by the prior
# failed coexistence allocation. This does not alter vm.swappiness.
reset_swap_pages() {
    sudo swapoff -a
    sudo swapon -a
}

# Capture a healthy start state, then stop only the identified FreeToken child.
curl -fsS "${FREETOKEN_HEALTH_URL}" >"${BEFORE_HEALTH_FILE}"
swapon --show --bytes >"${SWAP_BEFORE_FILE}"
freetoken_pid="$(find_freetoken_pid)" || {
    echo "error: no verified FreeToken server owns loopback port 1919" >&2
    exit 1
}
printf '%s\n' "${freetoken_pid}" >"${ARTIFACT_ROOT}/freetoken-server-pid.txt"
kill "${freetoken_pid}"
for _ in $(seq 1 180); do
    if ! kill -0 "${freetoken_pid}" 2>/dev/null; then
        break
    fi
    sleep 1
done
if kill -0 "${freetoken_pid}" 2>/dev/null; then
    echo "error: FreeToken server did not stop after SIGTERM" >&2
    exit 1
fi

# Standalone llama.cpp must not inherit swapped pages from the coexistence test.
reset_swap_pages
swapon --show --bytes >"${SWAP_AFTER_RELEASE_FILE}"

# Run the unchanged ROCm llama.cpp control, preserving its status while always
# restoring FreeToken before the wrapper returns.
set +e
bash "${CONTROL_SCRIPT}" "${CONTROL_ARTIFACT}"
control_status=$?
set -e
printf '%s\n' "${control_status}" >"${ARTIFACT_ROOT}/llamacpp-control-exit-code.txt"

# The recovery script prints its own dated artifact directory. Preserve it so
# the time-shared control links to native-cache and startup evidence.
bash "${RECOVERY_SCRIPT}" | tee "${ARTIFACT_ROOT}/freetoken-recovery-artifact.txt"
if ! wait_for_freetoken_health; then
    echo "error: FreeToken did not become healthy after time-share control" >&2
    exit 1
fi
swapon --show --bytes >"${SWAP_AFTER_FILE}"

# A benchmark failure is returned only after recovered FreeToken health passes.
exit "${control_status}"
