#!/usr/bin/env bash
# Measure warm Qwen decode throughput against the isolated LAN-223 FreeToken API.
#
# The workload is deliberately a fixed 48-times scheduler paragraph. It preserves
# the former 733-token-class LAN-223 baseline shape while remaining separate from
# the unrecovered upstream paper workload. This script neither starts nor stops a
# server and never contacts llama-swap or any non-LAN-223 endpoint.

set -euo pipefail

# Accept a caller-supplied artifact root so each run has immutable evidence.
readonly ARTIFACT_DIR="${1:?usage: run_qwen_scheduler_baseline.sh ARTIFACT_DIR}"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly SOURCE_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
readonly MODEL_DIR="${ROOT_DIR}/models/Qwen3.6-35B-A3B-NVFP4"
readonly MODEL_NAME="qwen3.6-35b-a3b-nvfp4-amd"
readonly BASE_URL="http://127.0.0.1:1919/v1"
readonly EXPECTED_HOST="david-Gmktec-x2-2"
readonly BASE_PROMPT="The scheduler manages incoming inference requests by prioritizing, batching, and assigning them to available compute resources to optimize throughput and latency. "

# Form the fixed input without shell interpolation at call time. The harness
# records its SHA-256 and checkpoint token count, so any future wording change
# becomes visible in the result artifact rather than silently changing TPS.
PROMPT=""
for _ in $(seq 1 48); do
    PROMPT+="${BASE_PROMPT}"
done

export PYTHONPATH="${SOURCE_DIR}/python"
cd "${SOURCE_DIR}"

# Forced-length greedy decoding yields a comparable stream interval. Qwen's
# reasoning stream is explicitly disabled because this measures final-token
# decoding, not variable-length internal reasoning. A warmup is retained but
# saved separately by the harness before the three scored samples.
"${VENV_PYTHON}" benchmarks/lan223_qwen/run_api_benchmark.py \
    --model "${MODEL_NAME}" \
    --tokenizer "${MODEL_DIR}" \
    --base-url "${BASE_URL}" \
    --expected-host "${EXPECTED_HOST}" \
    --artifact-dir "${ARTIFACT_DIR}" \
    --samples 3 \
    --warmup \
    --mode throughput \
    --expected-text "" \
    --max-tokens 256 \
    --prompt "${PROMPT}" \
    --reasoning-effort none
