#!/usr/bin/env bash
# Run one isolated Qwen GGUF raw-prompt quality control on LAN-223.
#
# This script deliberately takes the production API offline only while an
# isolated checkout owns the Strix Halo GPU.  Its EXIT trap always stops that
# temporary server and invokes the production recovery script before returning.
# The control sends a caller-rendered prompt through /v1/completions, allowing
# direct comparison with llama.cpp without a server-specific chat template.

set -euo pipefail

# The isolated checkout is required so this procedure can never modify the
# production source tree while proving a candidate change.
readonly CHECKOUT="${1:?usage: run_qwen_gguf_raw_control.sh ISOLATED_CHECKOUT [DECODE_TOKENS]}"
# A 512-token budget is normally sufficient to finish the fixed AIME answer;
# callers may supply another positive limit when investigating longer outputs.
readonly DECODE_TOKENS="${2:-512}"
# LAN-223's persistent project root keeps models, artifacts, and production
# recovery tooling outside the disposable candidate checkout.
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly PRODUCTION_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly MODEL_PATH="${ROOT_DIR}/models/controls/qwen36-35b-a3b-unsloth-a483e9e6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
readonly TOKENIZER_PATH="${ROOT_DIR}/models/Qwen3.6-35B-A3B-NVFP4"
readonly TEST_PORT="1922"
readonly PRODUCTION_PORT="1919"
readonly SERVED_MODEL="qwen36-35b-a3b-q4km-gguf-amd"
readonly ARTIFACT_DIR="${ROOT_DIR}/artifacts/qwen-gguf-raw-$(date -u +%Y%m%dT%H%M%SZ)"

# Artifact immutability makes a repeated timestamp collision an explicit error.
mkdir -p "${ARTIFACT_DIR}"

port_pid() {
    # Resolve the actual listener PID rather than relying on a stale pidfile.
    ss -ltnp "( sport = :$1 )" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

restore_production() {
    # Stop only the temporary listener, if it reached startup.
    local test_pid
    test_pid="$(port_pid "${TEST_PORT}")"
    [[ -z "${test_pid}" ]] || kill "${test_pid}" || true
    # Avoid a duplicate recovery when the production endpoint survived a setup
    # failure.  The recovery helper owns the production command and its logs.
    if ! timeout 5 curl -fsS "http://127.0.0.1:${PRODUCTION_PORT}/health" >/dev/null; then
        bash "${PRODUCTION_DIR}/scripts/gmk-evo-x2/start_qwen_recovery_server.sh" \
            | tee "${ARTIFACT_DIR}/recovery.log"
    fi
}
trap restore_production EXIT

# Release the one GPU from the persistent service before launching the candidate.
production_pid="$(port_pid "${PRODUCTION_PORT}")"
if [[ -n "${production_pid}" ]]; then
    kill "${production_pid}"
fi
for _ in {1..60}; do
    ss -ltn "( sport = :${PRODUCTION_PORT} )" | grep -q "${PRODUCTION_PORT}" || break
    sleep 1
done

# Start a test-only server with the same AMD execution configuration used by the
# prior Q4 controls.  Output is retained verbatim with the evidence artifact.
cd "${CHECKOUT}"
ROCM_HOME=/opt/rocm-10.0 ROCM_PATH=/opt/rocm-10.0 HIP_PATH=/opt/rocm-10.0 \
PYTHONPATH=python TORCH_EXTENSIONS_DIR="${ROOT_DIR}/cache/torch_extensions" \
nohup "${ROOT_DIR}/.venv/bin/python" -m freetoken.cli serve \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL}" \
    --host 127.0.0.1 --port "${TEST_PORT}" \
    --attention-backend triton --moe-backend offload --nvfp4-backend triton \
    --expert-load serial --moe-cache-auto --memory-ratio 0.35 \
    --max-seq-len-override 8192 --kv-reserve-tokens 2048 \
    --cuda-graph-max-bs 0 --disable-pynccl --disable-moe-prefill-overlap \
    >"${ARTIFACT_DIR}/server.log" 2>&1 &
candidate_pid=$!
for _ in {1..480}; do
    grep -q 'API server is ready to serve' "${ARTIFACT_DIR}/server.log" && break
    kill -0 "${candidate_pid}" 2>/dev/null || exit 1
    sleep 1
done
grep -q 'API server is ready to serve' "${ARTIFACT_DIR}/server.log"

# Persist the request body, final text, exact prompt hash, server usage, and
# first-token/decode timings in one self-contained JSON control artifact.
PYTHONPATH=python "${ROOT_DIR}/.venv/bin/python" \
    scripts/gmk-evo-x2/verify_qwen_raw_prompt_quality.py \
    --base-url "http://127.0.0.1:${TEST_PORT}" --model "${SERVED_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" --decode "${DECODE_TOKENS}" \
    --artifact "${ARTIFACT_DIR}/raw-quality.json" \
    >"${ARTIFACT_DIR}/raw-quality.log" 2>&1
