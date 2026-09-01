#!/usr/bin/env bash
# Capture a safe, warm-cache ROCprof trace for the isolated Qwen3.6 Q4 control.
#
# This controller deliberately treats profiling as a diagnostic time-share job,
# not as a TPS benchmark.  It first checks the normal NVFP4 API, prewarms the
# exact Q4 HIP cache without the profiler, traces one bounded request, requests
# graceful profiler finalization, verifies a SQLite trace database, and only
# then restarts the normal service.  It never exposes the candidate beyond its
# loopback-only test port and never contacts llama-swap.

# Exit for programming errors, unset values, and failed pipeline stages.
set -euo pipefail

# Require a unique caller-owned artifact directory for immutable provenance.
readonly ARTIFACT_DIR="${1:?usage: run_qwen_q4_rocprof_trace.sh ARTIFACT_DIR [SOURCE_DIR]}"
# Permit an explicit reviewed Qwen checkout while keeping the qualified source
# as the default for ordinary diagnostic traces.
readonly SOURCE_DIR="${2:-/home/david/freetoken-amd/source-qwen-bench-metrics-f1baf13}"
# Keep all fixed host paths together so they are easy to audit before use.
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly NORMAL_SOURCE_DIR="${ROOT_DIR}/source-qwen-c06-fc3346f"
readonly MODEL_PATH="${ROOT_DIR}/models/controls/qwen36-35b-a3b-unsloth-a483e9e6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
readonly VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
readonly Q4_MODEL_NAME="qwen36-35b-a3b-q4km-gguf-amd"
readonly Q4_PORT="1922"
readonly NORMAL_PORT="1919"
# Keep reusable generated HIP code outside every source checkout and under the
# managed cache root accepted by the qualified Q4 launcher.
readonly EXTENSION_CACHE="${ROOT_DIR}/cache/rocprof-q4-warm-cache"
readonly Q4_LAUNCHER="${SOURCE_DIR}/scripts/gmk-evo-x2/launch_qwen_gguf_qualified.sh"
readonly NORMAL_STARTER="${NORMAL_SOURCE_DIR}/scripts/gmk-evo-x2/start_qwen_recovery_server.sh"
readonly NORMAL_STOPPER="${NORMAL_SOURCE_DIR}/scripts/gmk-evo-x2/stop_qwen_recovery_server.sh"
readonly PROFILER_WRAPPER="${SOURCE_DIR}/scripts/gmk-evo-x2-rocprof-wheel-sdk.sh"
readonly INSPECTOR="${SOURCE_DIR}/scripts/gmk-evo-x2/inspect_rocprof_db.py"
readonly PROFILE_DIR="${ARTIFACT_DIR}/profile"
readonly PREWARM_DIR="${ARTIFACT_DIR}/prewarm"
readonly PROFILE_PID_FILE="${PROFILE_DIR}/profile.pid"
readonly PROFILE_LOG="${PROFILE_DIR}/server.log"

# Track lifecycle ownership so the EXIT trap restores only services this script
# actually stopped or started.
normal_was_stopped=0
profile_pid=""

# Return success only when the expected loopback API has completed a models call.
wait_for_models() {
    local port="$1"
    local attempts="$2"
    local destination="$3"
    for _ in $(seq 1 "${attempts}"); do
        if curl -fsS --max-time 8 "http://127.0.0.1:${port}/v1/models" >"${destination}"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# Wait for a test listener to disappear before reusing the GPU or port.
wait_for_port_clear() {
    for _ in $(seq 1 45); do
        if ! ss -ltn "( sport = :${Q4_PORT} )" | grep -q LISTEN; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# Confirm the recorded profiler process is the exact isolated Q4 server before
# sending it an interrupt.  This guard makes the shared host safe to operate.
is_profile_process() {
    local pid="$1"
    local command
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    command="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${command}" == *"freetoken.cli serve"* ]] &&
        [[ "${command}" == *"${MODEL_PATH}"* ]] &&
        [[ "${command}" == *"--port ${Q4_PORT}"* ]]
}

# Request graceful rocprof finalization first.  The bounded wait protects the
# normal service from an indefinitely stuck trace while preserving enough time
# for rocprofv3 to write its SQLite database.
finalize_profile() {
    [[ -n "${profile_pid}" ]] || return 0
    kill -0 "${profile_pid}" 2>/dev/null || return 0
    is_profile_process "${profile_pid}" || {
        echo "refusing to stop an unrecognized profiler process: ${profile_pid}" >&2
        return 1
    }
    local pgid
    pgid="$(ps -o pgid= -p "${profile_pid}" | tr -d ' ')"
    [[ "${pgid}" == "${profile_pid}" ]] || {
        echo "profiler process lacks its dedicated group: ${profile_pid}/${pgid}" >&2
        return 1
    }
    kill -INT -- "-${pgid}" || true
    for _ in $(seq 1 90); do
        kill -0 "${profile_pid}" 2>/dev/null || return 0
        sleep 1
    done
    echo "profiler did not finalize within 90 seconds" >&2
    return 1
}

# Always restore the normal API when this controller owns its time-share slot.
restore_normal_service() {
    local status="$?"
    set +e
    finalize_profile
    if [[ "${normal_was_stopped}" == "1" ]]; then
        bash "${NORMAL_STARTER}" "${ARTIFACT_DIR}/normal-recovery" || true
        wait_for_models "${NORMAL_PORT}" 240 "${ARTIFACT_DIR}/normal-health-after.json" || true
    fi
    exit "${status}"
}

# Install recovery before stopping the normal service so interrupts do not leave
# the GMKtec EVO-X2 without its normal local OpenAI-compatible endpoint.
trap restore_normal_service EXIT INT TERM

# Fail closed when a caller supplies an unexpected source tree or missing tools.
[[ "${SOURCE_DIR}" == "${ROOT_DIR}/source-qwen-"* ]] || { echo "invalid source directory" >&2; exit 2; }
[[ -f "${MODEL_PATH}" && -x "${VENV_PYTHON}" && -x "${Q4_LAUNCHER}" ]] || { echo "missing Q4 prerequisites" >&2; exit 2; }
[[ -x "${NORMAL_STARTER}" && -x "${NORMAL_STOPPER}" && -x "${PROFILER_WRAPPER}" ]] || { echo "missing lifecycle helper" >&2; exit 2; }
[[ -f "${INSPECTOR}" ]] || { echo "missing ROCprof inspector" >&2; exit 2; }
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo "artifact directory already exists: ${ARTIFACT_DIR}" >&2; exit 2; }
mkdir -p "${PREWARM_DIR}" "${PROFILE_DIR}" "${EXTENSION_CACHE}"

# Preserve proof that the protected API was healthy before reclaiming the GPU.
wait_for_models "${NORMAL_PORT}" 1 "${ARTIFACT_DIR}/normal-health-before.json" || {
    echo "normal API is not healthy; refusing profiler time-share" >&2
    exit 2
}

# Reserve the GPU through the existing tested stopper, then prewarm the exact
# Q4 server and extension cache without ROCprof overhead.
bash "${NORMAL_STOPPER}"
normal_was_stopped=1
FREETOKEN_Q4_SOURCE_DIR="${SOURCE_DIR}" FREETOKEN_Q4_EXTENSION_CACHE_DIR="${EXTENSION_CACHE}" \
    bash "${Q4_LAUNCHER}" start "${PREWARM_DIR}" 0.25 0
wait_for_models "${Q4_PORT}" 120 "${PREWARM_DIR}/models.json"

# Force one short deterministic request so startup and kernel compilation occur
# before tracing.  The response is retained as evidence, not scored for TPS.
curl -fsS --max-time 90 -H 'Content-Type: application/json' \
    -d "{\"model\":\"${Q4_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: warm cache confirmed\"}],\"temperature\":0,\"max_tokens\":16,\"stream\":false}" \
    "http://127.0.0.1:${Q4_PORT}/v1/chat/completions" >"${PREWARM_DIR}/response.json"
FREETOKEN_Q4_SOURCE_DIR="${SOURCE_DIR}" FREETOKEN_Q4_EXTENSION_CACHE_DIR="${EXTENSION_CACHE}" \
    bash "${Q4_LAUNCHER}" stop "${PREWARM_DIR}" 0.25 0
wait_for_port_clear

# Start the identical Q4 command inside a dedicated session.  The profile has a
# short delayed collection window that excludes most initialization, while the
# post-ready request below remains inside the 75-second capture interval.
cd "${SOURCE_DIR}"
ROCM_HOME=/opt/rocm-10.0 ROCM_PATH=/opt/rocm-10.0 HIP_PATH=/opt/rocm-10.0 \
PYTHONPATH=python TORCH_EXTENSIONS_DIR="${EXTENSION_CACHE}" \
setsid nohup bash "${PROFILER_WRAPPER}" -d "${PROFILE_DIR}/rocprof" -f rocpd \
    --runtime-trace --kernel-trace --memory-copy-trace --collection-period 45:75:1 \
    --process-sync true -- "${VENV_PYTHON}" -m freetoken.cli serve \
    --model-path "${MODEL_PATH}" --served-model-name "${Q4_MODEL_NAME}" \
    --host 127.0.0.1 --port "${Q4_PORT}" --max-running-requests 4 \
    --attention-backend triton --moe-backend offload --nvfp4-backend triton \
    --expert-load serial --moe-cache-auto --memory-ratio 0.25 \
    --max-seq-len-override 8192 --kv-reserve-tokens 8192 --cuda-graph-max-bs 0 \
    --disable-pynccl --disable-moe-prefill-overlap >"${PROFILE_LOG}" 2>&1 &
profile_pid="$!"
printf '%s\n' "${profile_pid}" >"${PROFILE_PID_FILE}"
wait_for_models "${Q4_PORT}" 120 "${PROFILE_DIR}/models.json"

# Generate one bounded greedy request.  Its fixed shape gives the trace a clear
# prefill and decode region while avoiding a variable reasoning-stream workload.
curl -fsS --max-time 120 -H 'Content-Type: application/json' \
    -d "{\"model\":\"${Q4_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write exactly 300 numbered lines. Every line must contain the words cache and expert.\"}],\"temperature\":0,\"top_p\":1,\"max_tokens\":900,\"stream\":false}" \
    "http://127.0.0.1:${Q4_PORT}/v1/chat/completions" >"${PROFILE_DIR}/workload-response.json"

# Give asynchronous ROCprof writers a short post-request interval before the
# graceful interrupt, then require one finalized SQLite database as the gate.
sleep 5
finalize_profile
profile_pid=""
database="$(find "${PROFILE_DIR}/rocprof" -type f -name '*_results.db' -print -quit)"
[[ -n "${database}" ]] || { echo "ROCprof database was not finalized" >&2; exit 3; }
"${VENV_PYTHON}" "${INSPECTOR}" --tail-seconds 30 "${database}" >"${PROFILE_DIR}/trace-summary.txt"
printf 'database=%s\n' "${database}" >"${PROFILE_DIR}/trace-database.txt"
