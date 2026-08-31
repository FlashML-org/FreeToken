#!/usr/bin/env bash
# Run a long isolated Q4 endurance battery and restore LAN-223's NVFP4 service.
#
# This controller owns one deliberate GPU time-share window. It does not touch
# llama-swap or any LAN endpoint. It stops the verified dedicated loopback
# recovery session, starts the isolated Q4 test session, runs the existing
# process-scoped battery, and restores the normal service even when the battery
# fails or the controller receives a termination signal.

set -euo pipefail

# Require a caller-owned immutable root, then permit the full 24-hour default
# while also allowing a shorter explicitly labelled diagnostic duration.
readonly ARTIFACT_ROOT="${1:?usage: run_qwen_gguf_timeshare_endurance.sh ARTIFACT_ROOT [SESSION_COUNT] [INTERVAL_SECONDS]}"
readonly SESSION_COUNT="${2:-1440}"
readonly INTERVAL_SECONDS="${3:-60}"

# Keep every host-specific path explicit so an invocation cannot silently
# operate on another machine's service or an arbitrary source checkout.
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly Q4_SOURCE_DIR="${FREETOKEN_Q4_SOURCE_DIR:?set FREETOKEN_Q4_SOURCE_DIR to an isolated Q4 worktree}"
readonly RECOVERY_SOURCE_DIR="${FREETOKEN_RECOVERY_SOURCE_DIR:?set FREETOKEN_RECOVERY_SOURCE_DIR to the recovery-launcher worktree}"
readonly Q4_LAUNCHER="${Q4_SOURCE_DIR}/scripts/lan223/launch_qwen_gguf_qualified.sh"
readonly Q4_BATTERY="${Q4_SOURCE_DIR}/scripts/lan223/run_qwen_gguf_endurance_battery.sh"
readonly RECOVERY_STOPPER="${RECOVERY_SOURCE_DIR}/scripts/lan223/stop_qwen_recovery_server.sh"
readonly RECOVERY_STARTER="${RECOVERY_SOURCE_DIR}/scripts/lan223/start_qwen_recovery_server.sh"
readonly Q4_ARTIFACT_DIR="${ARTIFACT_ROOT}/q4-server"
readonly BATTERY_ARTIFACT_DIR="${ARTIFACT_ROOT}/battery"
readonly RECOVERY_ARTIFACT="${ARTIFACT_ROOT}/recovery-health.json"

# Reject unsafe input before stopping the protected model service or creating
# any artifact. The source guards mirror the candidate launcher safeguards.
case "${SESSION_COUNT}" in ''|*[!0-9]*) echo "session count must be positive" >&2; exit 2;; esac
case "${INTERVAL_SECONDS}" in ''|*[!0-9]*) echo "interval must be non-negative" >&2; exit 2;; esac
(( SESSION_COUNT > 0 )) || { echo "session count must be positive" >&2; exit 2; }
[[ ! -e "${ARTIFACT_ROOT}" ]] || { echo "artifact root already exists: ${ARTIFACT_ROOT}" >&2; exit 2; }
[[ "${Q4_SOURCE_DIR}" == "${ROOT_DIR}/source-qwen-"* ]] || { echo "Q4 source must be under ${ROOT_DIR}" >&2; exit 2; }
[[ "${RECOVERY_SOURCE_DIR}" == "${ROOT_DIR}/source-qwen-"* ]] || { echo "recovery source must be under ${ROOT_DIR}" >&2; exit 2; }
[[ -f "${Q4_LAUNCHER}" && -f "${Q4_BATTERY}" && -f "${RECOVERY_STOPPER}" && -f "${RECOVERY_STARTER}" ]] || {
    echo "missing time-share dependency" >&2
    exit 2
}

# Poll the documented health state instead of treating a bound port or model
# listing as proof that a cold server has loaded all expert banks.
wait_for_serving() {
    local port="$1" destination="$2" status maintenance
    for _ in $(seq 1 900); do
        curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >"${destination}" 2>/dev/null || true
        status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${destination}" 2>/dev/null || true)"
        maintenance="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("maintenance", ""))' "${destination}" 2>/dev/null || true)"
        [[ "${status}" == "ok" && "${maintenance}" == "serving" ]] && return 0
        sleep 1
    done
    echo "server on port ${port} did not reach serving state" >&2
    return 1
}

# Restore in every exit path. The Q4 launcher verifies its own exact process
# group before signalling it, and the recovery launcher creates the dedicated
# group needed by future time-share windows.
restore_normal_service() {
    local status=0
    if [[ -f "${Q4_ARTIFACT_DIR}/server.pid" ]]; then
        FREETOKEN_Q4_SOURCE_DIR="${Q4_SOURCE_DIR}" bash "${Q4_LAUNCHER}" stop "${Q4_ARTIFACT_DIR}" || status=1
    fi
    if ! curl -fsS --max-time 5 http://127.0.0.1:1919/health >"${RECOVERY_ARTIFACT}" 2>/dev/null; then
        bash "${RECOVERY_STARTER}" >"${ARTIFACT_ROOT}/recovery-start.log" 2>&1 || status=1
    fi
    wait_for_serving 1919 "${RECOVERY_ARTIFACT}" || status=1
    return "${status}"
}

mkdir -p "${ARTIFACT_ROOT}"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${ARTIFACT_ROOT}/controller.txt"
printf 'session_count=%s\ninterval_seconds=%s\n' "${SESSION_COUNT}" "${INTERVAL_SECONDS}" >>"${ARTIFACT_ROOT}/controller.txt"
trap 'restore_normal_service' EXIT INT TERM

# The stopper refuses an unmanaged legacy tree. That fail-closed behavior
# prevents this controller from guessing at child ownership on a shared host.
bash "${RECOVERY_STOPPER}"
FREETOKEN_Q4_SOURCE_DIR="${Q4_SOURCE_DIR}" bash "${Q4_LAUNCHER}" start "${Q4_ARTIFACT_DIR}" 0.25
wait_for_serving 1922 "${ARTIFACT_ROOT}/q4-health.json"
FREETOKEN_Q4_SOURCE_DIR="${Q4_SOURCE_DIR}" bash "${Q4_BATTERY}" "${BATTERY_ARTIFACT_DIR}" "${SESSION_COUNT}" "${INTERVAL_SECONDS}"
"${ROOT_DIR}/.venv/bin/python" "${Q4_SOURCE_DIR}/benchmarks/lan223_qwen/summarize_qwen_gguf_endurance.py" \
    "${BATTERY_ARTIFACT_DIR}" --expected-sessions "${SESSION_COUNT}" >"${ARTIFACT_ROOT}/summary.json"
printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${ARTIFACT_ROOT}/controller.txt"
