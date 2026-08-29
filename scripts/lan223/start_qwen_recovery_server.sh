#!/usr/bin/env bash
# Start the isolated FreeToken Qwen NVFP4 recovery server on LAN-223.
#
# This script never touches systemd, llama-swap, or the masked production
# llama.cpp service on port 18302. It launches one loopback-only FreeToken
# process on port 1919 and writes all output into a uniquely timestamped
# artifact directory so post-reboot results remain reproducible.

set -euo pipefail

# Keep every recovery run separate from previous logs and benchmark artifacts.
readonly RUN_ID="qwen-reboot-recovery-$(date -u +%Y%m%dT%H%M%SZ)"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly SOURCE_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
readonly MODEL_DIR="${ROOT_DIR}/models/Qwen3.6-35B-A3B-NVFP4"
readonly ARTIFACT_DIR="${ROOT_DIR}/artifacts/${RUN_ID}"
readonly LOG_FILE="${ARTIFACT_DIR}/server.log"
readonly PID_FILE="${ARTIFACT_DIR}/server.pid"
readonly NATIVE_BUILD_LOG="${ARTIFACT_DIR}/native-extension-build.log"
readonly NATIVE_IMPORT_LOG="${ARTIFACT_DIR}/native-extension-import.txt"

# Refuse to launch if another process already owns the dedicated test port.
if ss -ltn "sport = :1919" | grep -q LISTEN; then
    echo "refusing to start: loopback benchmark port 1919 is already listening" >&2
    exit 1
fi

# Validate all immutable runtime inputs before starting a background process.
test -d "${SOURCE_DIR}"
test -x "${VENV_PYTHON}"
test -d "${MODEL_DIR}"
mkdir -p "${ARTIFACT_DIR}"

# These variables select the native ROCm toolchain and retain the existing HIP
# extension cache. Reusing the cache prevents a JIT build from contaminating the
# warm API benchmark that follows server readiness.
export PYTHONPATH="${SOURCE_DIR}/python"
export TORCH_EXTENSIONS_DIR="${ROOT_DIR}/cache/torch_extensions"
export ROCM_PATH="/opt/rocm-10.0"
export HIP_PATH="/opt/rocm-10.0"
export ROCM_HOME="/opt/rocm-10.0"

# FreeToken's MoE offload path requires the in-tree pinned-tensor extension.
# A clean git worktree does not contain generated shared objects, so verify the
# import first and build the two native modules in that worktree only when it
# is absent. The build log is an artifact because the extension's compiler,
# ROCm headers, and link result are part of a reproducible HIP validation.
if ! "${VENV_PYTHON}" -c 'import freetoken.kernel._pinned_tensor' >/dev/null 2>&1; then
    (
        cd "${SOURCE_DIR}"
        "${VENV_PYTHON}" setup.py build_ext --inplace
    ) >"${NATIVE_BUILD_LOG}" 2>&1
fi
"${VENV_PYTHON}" -c \
    'import freetoken.kernel._pinned_tensor as pinned; print(pinned.__file__)' \
    >"${NATIVE_IMPORT_LOG}"

# The fixed policy is the previously successful LAN-223 Qwen configuration.
# The 0.35 memory budget and 2,048-token KV reserve avoid the OOM observed with
# the larger automatic allocation. Serial expert loading is the ROCm-correct
# route and prefill overlap stays disabled for the validated safe baseline.
nohup "${VENV_PYTHON}" -m freetoken.cli serve \
    --model-path "${MODEL_DIR}" \
    --served-model-name qwen3.6-35b-a3b-nvfp4-amd \
    --host 127.0.0.1 \
    --port 1919 \
    --attention-backend triton \
    --moe-backend offload \
    --nvfp4-backend triton \
    --expert-load serial \
    --moe-cache-auto \
    --memory-ratio 0.35 \
    --max-seq-len-override 8192 \
    --kv-reserve-tokens 2048 \
    --cuda-graph-max-bs 0 \
    --disable-pynccl \
    --disable-moe-prefill-overlap \
    >"${LOG_FILE}" 2>&1 < /dev/null &

# Persist the child PID for diagnostics and explicit shutdown after the run.
echo "$!" >"${PID_FILE}"
printf '%s\n' "${ARTIFACT_DIR}"
