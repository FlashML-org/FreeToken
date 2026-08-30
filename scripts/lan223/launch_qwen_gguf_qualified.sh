#!/usr/bin/env bash
# Start or stop the qualified LAN-223 Qwen3.6 Q4_K_M FreeToken test server.
#
# This helper is deliberately limited to the isolated loopback test port.  It
# does not start the normal NVFP4 service, contact llama-swap, change system
# swap policy, or make a model available on the LAN.  The start action puts the
# entire FreeToken multiprocessing tree in its own session and process group.
# The stop action verifies that group before stopping it, which prevents the
# orphaned distributed worker and internal-port collision observed during the
# Q4 SVM-resident-memory investigation.

set -euo pipefail

# Require a deliberate lifecycle action instead of guessing whether a caller
# intended to start a service or release the GPU for a llama.cpp control.
readonly ACTION="${1:?usage: launch_qwen_gguf_qualified.sh start|stop ARTIFACT_DIR [MEMORY_RATIO]}"
# Require a caller-owned evidence directory.  The script writes only its PID
# file and server log there, so every test run preserves its own provenance.
readonly ARTIFACT_DIR="${2:?usage: launch_qwen_gguf_qualified.sh start|stop ARTIFACT_DIR [MEMORY_RATIO]}"
# Keep the memory-safe recovery profile as the explicit default.  Callers may
# supply a different ratio for a recorded experiment, never for a silent
# production configuration change.
readonly MEMORY_RATIO="${3:-0.25}"

# Keep durable models, kernel caches, and artifacts separate from the checked
# out source so source switching cannot delete benchmark evidence or weights.
readonly ROOT_DIR="/home/david/freetoken-amd"
# This is the isolated Q4-capable checkout used for the native GGUF controls.
readonly SOURCE_DIR="${ROOT_DIR}/source-qwen-gguf-5c7f0fd"
# The exact file is also used by the matching ROCm llama.cpp control.
readonly MODEL_PATH="${ROOT_DIR}/models/controls/qwen36-35b-a3b-unsloth-a483e9e6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
# Preserve the shared checkpoint tokenizer for API and benchmark token counts.
readonly SERVED_MODEL="qwen36-35b-a3b-q4km-gguf-amd"
# Restrict this helper to the disposable loopback endpoint, never port 1919.
readonly PORT="1922"
# The FreeToken engine creates a local distributed TCP store on this next port.
# An existing listener means an earlier multiprocessing group was not cleaned.
readonly INTERNAL_PORT="1923"
# Keep HIP extension artifacts in the revisioned shared cache established by
# the native strict no-JIT qualification rather than compiling per run.
readonly EXTENSION_CACHE="${ROOT_DIR}/cache/torch_extensions"
# Store lifecycle data next to the supplied immutable test artifact.
readonly PID_FILE="${ARTIFACT_DIR}/server.pid"
readonly LOG_FILE="${ARTIFACT_DIR}/server.log"

# Resolve a listener PID without assuming that a stale PID file identifies the
# live owner of a TCP port.  An empty answer is a valid no-listener condition.
listener_pid() {
    local port="$1"
    ss -ltnp "( sport = :${port} )" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

# Return success only for the known test-server command.  This is the guard
# that makes a PID or process-group signal safe in a shared LAN-223 shell.
is_qualified_q4_process() {
    local pid="$1"
    local command
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    command="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${command}" == *"freetoken.cli serve"* ]] &&
        [[ "${command}" == *"${MODEL_PATH}"* ]] &&
        [[ "${command}" == *"--port ${PORT}"* ]]
}

# Stop a dedicated session only after proving the main process owns that group.
# `setsid` makes the server PID, session ID, and process-group ID equal, so one
# signal reaches the HTTP parent, scheduler, tokenizer worker, and tracker.
stop_qualified_group() {
    local pid="$1"
    local pgid
    is_qualified_q4_process "${pid}" || {
        echo "refusing to stop an unrecognized process: ${pid}" >&2
        return 1
    }
    pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
    [[ "${pgid}" == "${pid}" ]] || {
        echo "refusing to stop process ${pid}: expected dedicated process group, got ${pgid}" >&2
        return 1
    }
    kill -TERM -- "-${pgid}" || true
    for _ in $(seq 1 30); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
    done
    # A stuck HIP kernel can prevent graceful Python exit.  Escalate only the
    # already-verified dedicated group after the bounded graceful wait.
    kill -0 "${pid}" 2>/dev/null && kill -KILL -- "-${pgid}" || true
}

# Validate fixed paths before any lifecycle action so a changed layout fails
# closed rather than starting a different model or a CPU fallback.
validate_paths() {
    [[ -d "${SOURCE_DIR}" ]] || { echo "missing source directory: ${SOURCE_DIR}" >&2; return 1; }
    [[ -f "${MODEL_PATH}" ]] || { echo "missing Q4 model: ${MODEL_PATH}" >&2; return 1; }
    [[ -x "${ROOT_DIR}/.venv/bin/python" ]] || { echo "missing benchmark Python" >&2; return 1; }
    [[ "${MEMORY_RATIO}" =~ ^0\.[0-9]+$|^1\.0+$ ]] || { echo "invalid memory ratio: ${MEMORY_RATIO}" >&2; return 1; }
}

case "${ACTION}" in
    start)
        validate_paths
        # Artifacts must be unique so a retry cannot overwrite the first log.
        [[ -e "${PID_FILE}" ]] && { echo "PID file already exists: ${PID_FILE}" >&2; exit 2; }
        # The HTTP and internal distributed ports must both be clear.  The
        # internal-port check detects orphaned workers before touching the GPU.
        [[ -z "$(listener_pid "${PORT}")" ]] || { echo "test port ${PORT} is already listening" >&2; exit 2; }
        [[ -z "$(listener_pid "${INTERNAL_PORT}")" ]] || { echo "internal port ${INTERNAL_PORT} is already listening" >&2; exit 2; }
        mkdir -p "${ARTIFACT_DIR}"
        cd "${SOURCE_DIR}"
        # Set only this child environment.  ROCm paths select the native HIP
        # stack, while PYTHONPATH and TORCH_EXTENSIONS_DIR select the reviewed
        # Q4 source and prebuilt extension cache without changing the login
        # shell or normal service environment.
        ROCM_HOME=/opt/rocm-10.0 ROCM_PATH=/opt/rocm-10.0 HIP_PATH=/opt/rocm-10.0 \
        PYTHONPATH=python TORCH_EXTENSIONS_DIR="${EXTENSION_CACHE}" \
        setsid nohup "${ROOT_DIR}/.venv/bin/python" -m freetoken.cli serve \
            --model-path "${MODEL_PATH}" \
            --served-model-name "${SERVED_MODEL}" \
            --host 127.0.0.1 --port "${PORT}" \
            --max-running-requests 4 \
            --attention-backend triton --moe-backend offload --nvfp4-backend triton \
            --expert-load serial --moe-cache-auto --memory-ratio "${MEMORY_RATIO}" \
            --max-seq-len-override 8192 --kv-reserve-tokens 8192 \
            --cuda-graph-max-bs 0 --disable-pynccl --disable-moe-prefill-overlap \
            >"${LOG_FILE}" 2>&1 &
        echo "$!" >"${PID_FILE}"
        ;;
    stop)
        # Prefer the recorded server PID, but verify it before a signal.  This
        # keeps a malformed artifact from targeting an unrelated user process.
        [[ -f "${PID_FILE}" ]] || { echo "missing server PID file: ${PID_FILE}" >&2; exit 2; }
        stop_qualified_group "$(<"${PID_FILE}")"
        ;;
    *)
        echo "unknown action: ${ACTION}; expected start or stop" >&2
        exit 2
        ;;
esac
