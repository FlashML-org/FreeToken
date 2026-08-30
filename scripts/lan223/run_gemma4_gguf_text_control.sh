#!/usr/bin/env bash
# Launch Gemma4 Q4 GGUF in an isolated LAN-223 control slot and restore Qwen.

set -euo pipefail

readonly CHECKOUT="${1:?usage: run_gemma4_gguf_text_control.sh ISOLATED_CHECKOUT}"
readonly MODE="${2:-text}"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly PRODUCTION_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly MODEL_PATH="${ROOT_DIR}/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf"
readonly TEST_PORT="1923"
readonly PRODUCTION_PORT="1919"
readonly ARTIFACT_DIR="${ROOT_DIR}/artifacts/gemma4-gguf-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${ARTIFACT_DIR}"

port_pid() { ss -ltnp "( sport = :$1 )" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1; }
production_ready() {
    # A TCP listener and a 200 response can both exist while FreeToken is still
    # loading its expert groups.  Inspect the authoritative health status so a
    # temporary candidate never begins while Qwen is only partially recovered.
    timeout 5 curl -fsS "http://127.0.0.1:${PRODUCTION_PORT}/health" | grep -q '"status":"ok"'
}
restore_production() {
    local test_pid
    test_pid="$(port_pid "${TEST_PORT}")"
    if [[ -n "${test_pid}" ]]; then
        # Do not race the Qwen recovery process against the temporary Gemma
        # process still releasing its ROCm context. A bare kill followed by an
        # immediate recovery launch intermittently produced an empty Qwen log
        # and a dead child on LAN-223.
        kill "${test_pid}" || true
        for _ in {1..30}; do
            kill -0 "${test_pid}" 2>/dev/null || break
            sleep 1
        done
        # The port owner can exit before HIP finishes tearing down its GPU
        # context. Give ROCm a bounded grace period before Qwen tries to claim
        # the device, avoiding a child that dies before it can write server.log.
        sleep 10
    fi
    if ! production_ready; then
        # The recovery script is intentionally external and protects the source
        # checkout. Verify its observable health result rather than treating a
        # background PID or an artifact-directory print as successful recovery.
        local recovered=0
        # Launch exactly once. Qwen takes several minutes to load its three
        # serial NVFP4 expert groups on LAN-223. Retrying the launcher while
        # its listener already exists only produces a misleading refusal and
        # wastes the short recovery window.
        bash "${PRODUCTION_DIR}/scripts/lan223/start_qwen_recovery_server.sh" \
            | tee -a "${ARTIFACT_DIR}/recovery.log" || true
        # The protected model normally needs roughly six to eight minutes from
        # a cold recovery. Wait a bounded eight minutes for the authoritative
        # ready status, rather than mistaking a temporary loading response for
        # success or declaring a healthy in-progress recovery a failure.
        for _ in {1..480}; do
            production_ready && {
                recovered=1
                break
            }
            sleep 1
        done
        [[ "${recovered}" == "1" ]] || echo "WARNING: Qwen recovery did not become reachable" >&2
    fi
}
trap restore_production EXIT

case "${MODE}" in
    text|vision) ;;
    *) echo "mode must be text or vision, got ${MODE}" >&2; exit 2 ;;
esac

# Refuse to evict the protected service during its multi-minute NVFP4 recovery.
production_ready

production_pid="$(port_pid "${PRODUCTION_PORT}")"
[[ -z "${production_pid}" ]] || kill "${production_pid}"
for _ in {1..60}; do ss -ltn "( sport = :${PRODUCTION_PORT} )" | grep -q "${PRODUCTION_PORT}" || break; sleep 1; done

cd "${CHECKOUT}"
vision_env=()
if [[ "${MODE}" == "vision" ]]; then
    # This explicit opt-in causes the isolated Gemma candidate to allocate and
    # load its sibling 1.2 GiB mmproj vision tower. Text mode preserves the
    # normal production memory budget.
    vision_env=(FREETOKEN_LOAD_VISION=1)
fi
# ``env`` is required here: an expanded Bash array is not parsed as assignment
# words, so placing ``${vision_env[@]}`` before ``nohup`` directly would try to
# execute the literal ``FREETOKEN_LOAD_VISION=1`` string as a program.
env ROCM_HOME=/opt/rocm-10.0 ROCM_PATH=/opt/rocm-10.0 HIP_PATH=/opt/rocm-10.0 \
PYTHONPATH=python TORCH_EXTENSIONS_DIR="${ROOT_DIR}/cache/torch_extensions" \
"${vision_env[@]}" nohup "${ROOT_DIR}/.venv/bin/python" -m freetoken.cli serve \
    --model-path "${MODEL_PATH}" --served-model-name gemma4-26b-q4-amd \
    --host 127.0.0.1 --port "${TEST_PORT}" --attention-backend triton \
    --moe-backend offload --expert-load serial --moe-cache-auto --memory-ratio 0.35 \
    --max-seq-len-override 8192 --kv-reserve-tokens 2048 --cuda-graph-max-bs 0 \
    --disable-pynccl --disable-moe-prefill-overlap >"${ARTIFACT_DIR}/server.log" 2>&1 &
candidate_pid=$!
for _ in {1..480}; do
    grep -q 'API server is ready to serve' "${ARTIFACT_DIR}/server.log" && break
    kill -0 "${candidate_pid}" 2>/dev/null || exit 1
    sleep 1
done
grep -q 'API server is ready to serve' "${ARTIFACT_DIR}/server.log"
curl -fsS "http://127.0.0.1:${TEST_PORT}/health" >"${ARTIFACT_DIR}/health.json"
PYTHONPATH=python "${ROOT_DIR}/.venv/bin/python" scripts/lan223/verify_gemma4_gguf_text.py \
    --base-url "http://127.0.0.1:${TEST_PORT}" --model gemma4-26b-q4-amd \
    --gguf "${MODEL_PATH}" --artifact "${ARTIFACT_DIR}/quality.json" \
    >"${ARTIFACT_DIR}/quality.log" 2>&1

if [[ "${MODE}" == "vision" ]]; then
    # Keep the candidate alive through the actual OpenAI image_url contract
    # control. The verifier writes a self-contained response/usage artifact;
    # only after it succeeds does the EXIT trap reclaim port 1923 and restore
    # the protected Qwen server.
    PYTHONPATH=python "${ROOT_DIR}/.venv/bin/python" scripts/lan223/verify_gemma4_gguf_image.py \
        --base-url "http://127.0.0.1:${TEST_PORT}" --model gemma4-26b-q4-amd \
        --stream --artifact "${ARTIFACT_DIR}/image-quality.json" \
        >"${ARTIFACT_DIR}/image-quality.log" 2>&1
    # The long-response fixture supplies an output-length quality gate, which
    # makes its stream timing suitable for a visual decode-TPS measurement.
    PYTHONPATH=python "${ROOT_DIR}/.venv/bin/python" scripts/lan223/verify_gemma4_gguf_visual_tps.py \
        --base-url "http://127.0.0.1:${TEST_PORT}" --model gemma4-26b-q4-amd \
        --artifact "${ARTIFACT_DIR}/visual-tps.json" \
        >"${ARTIFACT_DIR}/visual-tps.log" 2>&1
fi
