#!/usr/bin/env bash
# Run a caller-rendered Qwen GGUF raw-prompt quality control against ROCm llama.cpp.
# This uses the same model file, prompt renderer, decoding parameters, and evidence
# schema as run_qwen_gguf_raw_control.sh, then restores FreeToken on exit.

set -euo pipefail

readonly DECODE_TOKENS="${1:-1024}"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly PRODUCTION_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly LLAMA_SERVER="${ROOT_DIR}/llama.cpp-rocm10-b10141/build-rocm10-clang/bin/llama-server"
readonly MODEL_PATH="${ROOT_DIR}/models/controls/qwen36-35b-a3b-unsloth-a483e9e6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
readonly TOKENIZER_PATH="${ROOT_DIR}/models/Qwen3.6-35B-A3B-NVFP4"
readonly TEST_PORT="1921"
readonly PRODUCTION_PORT="1919"
readonly SERVED_MODEL="qwen36-35b-a3b-q4km-llama-raw"
readonly HARNESS_DIR="${ROOT_DIR}/validation-qwen-gguf-d1dd473"
readonly ARTIFACT_DIR="${ROOT_DIR}/artifacts/qwen-llama-raw-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${ARTIFACT_DIR}"

port_pid() { ss -ltnp "( sport = :$1 )" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1; }
restore_production() {
    local test_pid
    test_pid="$(port_pid "${TEST_PORT}")"
    [[ -z "${test_pid}" ]] || kill "${test_pid}" || true
    if ! timeout 5 curl -fsS "http://127.0.0.1:${PRODUCTION_PORT}/health" >/dev/null; then
        bash "${PRODUCTION_DIR}/scripts/gmk-evo-x2/start_qwen_recovery_server.sh" | tee "${ARTIFACT_DIR}/recovery.log"
    fi
}
trap restore_production EXIT

production_pid="$(port_pid "${PRODUCTION_PORT}")"
[[ -z "${production_pid}" ]] || kill "${production_pid}"
for _ in {1..60}; do ss -ltn "( sport = :${PRODUCTION_PORT} )" | grep -q "${PRODUCTION_PORT}" || break; sleep 1; done

export LD_LIBRARY_PATH="/opt/rocm-10.0/llvm/lib:/opt/rocm-10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
nohup "${LLAMA_SERVER}" -m "${MODEL_PATH}" --alias "${SERVED_MODEL}" -ngl all -c 8192 -np 1 \
    -b 2048 -ub 512 -ctk q8_0 -ctv q8_0 -fa on --jinja --reasoning-format deepseek \
    --no-context-shift --no-warmup --host 127.0.0.1 --port "${TEST_PORT}" \
    >"${ARTIFACT_DIR}/server.log" 2>&1 &
candidate_pid=$!
for _ in {1..180}; do
    timeout 5 curl -fsS "http://127.0.0.1:${TEST_PORT}/health" >"${ARTIFACT_DIR}/health.json" && break
    kill -0 "${candidate_pid}" 2>/dev/null || exit 1
    sleep 1
done
test -s "${ARTIFACT_DIR}/health.json"

cd "${HARNESS_DIR}"
PYTHONPATH=python "${ROOT_DIR}/.venv/bin/python" scripts/gmk-evo-x2/verify_qwen_raw_prompt_quality.py \
    --base-url "http://127.0.0.1:${TEST_PORT}" --model "${SERVED_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" --decode "${DECODE_TOKENS}" \
    --artifact "${ARTIFACT_DIR}/raw-quality.json" >"${ARTIFACT_DIR}/raw-quality.log" 2>&1
