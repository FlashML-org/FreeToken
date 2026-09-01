#!/usr/bin/env bash
# Run an isolated, process-scoped Qwen GGUF endurance battery on LAN-223.
#
# Linux reports swap for every desktop and monitoring process.  A system-wide
# zero-swap requirement can therefore reject a healthy model server because an
# unrelated service such as netdata or Xwayland has one swapped page.  This
# battery keeps that whole-host number as telemetry, but enforces zero swapped
# pages only for the verified FreeToken Q4 server process group and its
# multiprocessing children.

set -euo pipefail

# Require a fresh caller-owned artifact directory for every endurance run.
readonly ARTIFACT_ROOT="${1:?usage: run_qwen_gguf_endurance_battery.sh ARTIFACT_ROOT [SESSION_COUNT] [INTERVAL_SECONDS]}"
# Default to a one-hour cadence while permitting short, explicitly labelled
# diagnostic runs that use the same request and validation contract.
readonly SESSION_COUNT="${2:-60}"
# Sleep after a completed session so normal time-based drift is visible instead
# of compressing every request into a short throughput-only batch.
readonly INTERVAL_SECONDS="${3:-60}"

# Keep all fixed LAN-223 paths explicit for reproducibility and host isolation.
readonly ROOT_DIR="/home/david/freetoken-amd"
# Allow an isolated candidate worktree to reuse the exact endurance contract.
# The caller must choose a path under the dedicated Qwen source root, so this
# override cannot accidentally execute arbitrary code or touch port 1919.
readonly SOURCE_DIR="${FREETOKEN_Q4_SOURCE_DIR:-${ROOT_DIR}/source-qwen-gguf-5c7f0fd}"
readonly VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
readonly RUNNER="${SOURCE_DIR}/benchmarks/gmk_evo_x2/run_multiturn_state_suite.py"
readonly SUITE="${SOURCE_DIR}/benchmarks/gmk_evo_x2/multiturn_state_suite.json"
readonly MODEL="qwen36-35b-a3b-q4km-gguf-amd"
readonly PORT="1922"
readonly EXPECTED_HOST="david-Gmktec-x2-2"

# Reject malformed numeric input before opening a socket or creating artifacts.
case "${SESSION_COUNT}" in ''|*[!0-9]*) echo "session count must be a positive integer" >&2; exit 2;; esac
case "${INTERVAL_SECONDS}" in ''|*[!0-9]*) echo "interval must be a non-negative integer" >&2; exit 2;; esac
(( SESSION_COUNT > 0 )) || { echo "session count must be positive" >&2; exit 2; }
[[ ! -e "${ARTIFACT_ROOT}" ]] || { echo "artifact root already exists: ${ARTIFACT_ROOT}" >&2; exit 2; }
[[ "${SOURCE_DIR}" == "${ROOT_DIR}/source-qwen-"* ]] || {
    echo "source directory must be an isolated Qwen checkout under ${ROOT_DIR}" >&2
    exit 2
}
[[ -x "${VENV_PYTHON}" && -f "${RUNNER}" && -f "${SUITE}" ]] || {
    echo "missing Qwen endurance dependency" >&2
    exit 2
}

# Resolve the current HTTP listener instead of trusting a stale PID file.
listener_pid() {
    ss -ltnp "( sport = :${PORT} )" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

# Verify that the port owner is the isolated Q4 test runner before reading its
# process tree.  This prevents a port collision from turning into an unrelated
# process inspection or false passing endurance result.
qualified_server_pid() {
    local pid command
    pid="$(listener_pid)"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    command="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${command}" == *"freetoken.cli serve"* ]] || return 1
    [[ "${command}" == *"qwen36-35b-a3b-q4km-gguf-amd"* ]] || return 1
    printf '%s\n' "${pid}"
}

# Sum VmSwap across the server's dedicated process group.  The qualified
# launcher creates a group whose ID equals the HTTP server PID.  Requiring that
# invariant detects manually started or partially recovered process trees.
runner_swap_kib() {
    local server_pid pgid pid seen=0 total=0 swapped
    server_pid="$(qualified_server_pid)" || return 1
    pgid="$(ps -o pgid= -p "${server_pid}" | tr -d ' ')"
    [[ "${pgid}" == "${server_pid}" ]] || return 1
    while read -r pid; do
        [[ -r "/proc/${pid}/status" ]] || continue
        swapped="$(awk '/^VmSwap:/{print $2}' "/proc/${pid}/status")"
        total=$((total + ${swapped:-0}))
        seen=$((seen + 1))
    done < <(ps -eo pid=,pgid= | awk -v group="${pgid}" '$2 == group {print $1}')
    (( seen > 0 )) || return 1
    printf '%s\n' "${total}"
}

# Record whole-host and process-scoped memory facts separately.  Whole-host
# swap remains useful for diagnosing host contention, but only the runner value
# is a pass or fail condition for this model-service qualification.
record_memory() {
    local destination="$1"
    local runner_swap
    runner_swap="$(runner_swap_kib)" || {
        echo "cannot resolve qualified Q4 process group" >&2
        return 1
    }
    {
        printf 'captured_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'runner_swap_kib=%s\n' "${runner_swap}"
        printf 'whole_host_swap_kib=%s\n' "$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2} END{print t-f}' /proc/meminfo)"
        awk '/^(MemAvailable|SwapCached|SwapTotal|SwapFree):/{print}' /proc/meminfo
        free -k
        rocm-smi --showtemp --showperflevel --showmeminfo vram 2>&1 || true
    } >"${destination}"
    [[ "${runner_swap}" == "0" ]]
}

mkdir -p "${ARTIFACT_ROOT}/sessions"
record_memory "${ARTIFACT_ROOT}/preflight.txt" || {
    echo "refusing endurance run: qualified Q4 process has swapped pages" >&2
    exit 2
}

# Use the exact deterministic three-turn suite for every timed session.  Each
# per-session JSON contains the visible output and tail timing, while this
# wrapper adds process-scoped swap and GPU telemetry at the session boundary.
export PYTHONPATH="${SOURCE_DIR}/python"
for session in $(seq -w 1 "${SESSION_COUNT}"); do
    started_epoch="$(date +%s)"
    "${VENV_PYTHON}" "${RUNNER}" \
        --base-url "http://127.0.0.1:${PORT}/v1" \
        --model "${MODEL}" \
        --artifact "${ARTIFACT_ROOT}/sessions/session-${session}.json" \
        --suite "${SUITE}" \
        --expected-host "${EXPECTED_HOST}" \
        --max-tokens 64 >"${ARTIFACT_ROOT}/sessions/session-${session}.log" 2>&1
    record_memory "${ARTIFACT_ROOT}/sessions/session-${session}-telemetry.txt" || {
        echo "runner swap gate failed after session ${session}" >&2
        exit 2
    }
    elapsed=$(( $(date +%s) - started_epoch ))
    # `seq -w` produces labels such as 08.  Force decimal interpretation so
    # Bash does not treat that label as an invalid octal literal in arithmetic.
    if (( 10#${session} < SESSION_COUNT && elapsed < INTERVAL_SECONDS )); then
        sleep $((INTERVAL_SECONDS - elapsed))
    fi
done

# Retain a final sample after the last conversation for recovery verification.
record_memory "${ARTIFACT_ROOT}/postflight.txt"
