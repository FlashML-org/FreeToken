#!/usr/bin/env bash
# Run Gemma4 Q4 plus its vision projector through the ROCm 10 llama.cpp control.
#
# This is the matched comparison companion to run_gemma4_gguf_text_control.sh:
# it uses the identical text GGUF, sibling mmproj file, loopback isolation, text
# question, and OpenAI data-URL image fixtures. It never modifies llama-swap or
# the protected Qwen source checkout.

set -euo pipefail

readonly CHECKOUT="${1:?usage: run_gemma4_llamacpp_vision_control.sh ISOLATED_CHECKOUT}"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly PRODUCTION_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly LLAMA_SERVER="${ROOT_DIR}/llama.cpp-rocm10-b10141/build-rocm10-clang/bin/llama-server"
readonly MODEL_PATH="${ROOT_DIR}/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf"
readonly MMPROJ_PATH="${ROOT_DIR}/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B-it-mmproj.gguf"
readonly TEST_PORT="1924"
readonly PRODUCTION_PORT="1919"
readonly MODEL_NAME="gemma4-26b-q4-llamacpp-rocm10"
readonly ARTIFACT_DIR="${ROOT_DIR}/artifacts/gemma4-llamacpp-vision-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${ARTIFACT_DIR}"

port_pid() { ss -ltnp "( sport = :$1 )" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1; }
production_ready() {
    timeout 5 curl -fsS "http://127.0.0.1:${PRODUCTION_PORT}/health" | grep -q '"status":"ok"'
}
restore_production() {
    local test_pid recovered
    test_pid="$(port_pid "${TEST_PORT}")"
    if [[ -n "${test_pid}" ]]; then
        kill "${test_pid}" || true
        for _ in {1..30}; do kill -0 "${test_pid}" 2>/dev/null || break; sleep 1; done
        # HIP teardown outlives the listener. Do not race the next ROCm process.
        sleep 10
    fi
    if ! production_ready; then
        recovered=0
        for _ in {1..3}; do
            bash "${PRODUCTION_DIR}/scripts/lan223/start_qwen_recovery_server.sh" \
                | tee -a "${ARTIFACT_DIR}/recovery.log" || true
            for _ in {1..20}; do
                timeout 5 curl -fsS "http://127.0.0.1:${PRODUCTION_PORT}/health" >/dev/null && {
                    recovered=1
                    break 2
                }
                sleep 1
            done
            sleep 10
        done
        [[ "${recovered}" == "1" ]] || echo "WARNING: Qwen recovery did not become reachable" >&2
    fi
}
trap restore_production EXIT

[[ -x "${LLAMA_SERVER}" ]] || { echo "missing llama-server: ${LLAMA_SERVER}" >&2; exit 2; }
[[ -f "${MODEL_PATH}" && -f "${MMPROJ_PATH}" ]] || { echo "missing Gemma GGUF or mmproj" >&2; exit 2; }
production_ready

production_pid="$(port_pid "${PRODUCTION_PORT}")"
[[ -z "${production_pid}" ]] || kill "${production_pid}"
for _ in {1..60}; do ss -ltn "( sport = :${PRODUCTION_PORT} )" | grep -q "${PRODUCTION_PORT}" || break; sleep 1; done

# Use the same ROCm 10 libraries, full text-model and projector offload, Q8 KV,
# Flash Attention, one slot, and 8,192-token context as the existing Qwen
# llama.cpp controls. The projector is explicit so no download or auto-selection
# alters the comparison.
export LD_LIBRARY_PATH="/opt/rocm-10.0/llvm/lib:/opt/rocm-10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${LLAMA_SERVER}" -m "${MODEL_PATH}" -mm "${MMPROJ_PATH}" --mmproj-offload \
    --alias "${MODEL_NAME}" -ngl all -c 8192 -np 1 -b 2048 -ub 512 \
    -ctk q8_0 -ctv q8_0 -fa on --jinja --no-context-shift --no-warmup \
    --host 127.0.0.1 --port "${TEST_PORT}" >"${ARTIFACT_DIR}/server.log" 2>&1 &
candidate_pid=$!
for _ in {1..240}; do
    timeout 5 curl -fsS "http://127.0.0.1:${TEST_PORT}/health" >"${ARTIFACT_DIR}/health.json" && break
    kill -0 "${candidate_pid}" 2>/dev/null || { tail -120 "${ARTIFACT_DIR}/server.log" >&2; exit 1; }
    sleep 1
done
test -s "${ARTIFACT_DIR}/health.json"

cd "${CHECKOUT}"
PYTHONPATH=python "${ROOT_DIR}/.venv/bin/python" scripts/lan223/verify_gemma4_gguf_text.py \
    --base-url "http://127.0.0.1:${TEST_PORT}" --model "${MODEL_NAME}" \
    --gguf "${MODEL_PATH}" --artifact "${ARTIFACT_DIR}/quality.json" \
    >"${ARTIFACT_DIR}/quality.log" 2>&1
PYTHONPATH=python "${ROOT_DIR}/.venv/bin/python" scripts/lan223/verify_gemma4_gguf_image.py \
    --base-url "http://127.0.0.1:${TEST_PORT}" --model "${MODEL_NAME}" \
    --max-tokens 128 --artifact "${ARTIFACT_DIR}/image-quality.json" \
    >"${ARTIFACT_DIR}/image-quality.log" 2>&1
