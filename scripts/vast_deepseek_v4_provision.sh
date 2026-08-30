#!/usr/bin/env bash
set -Eeuo pipefail

# DeepSeek-V4-Flash uses the same cacheable Vast/PyWorker bootstrap as GLM.
# Keep the model-specific contract in this public wrapper so Vast workers can
# cold-start without credentials for a separate deployment repository.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TEKIZAI_FREETOKEN_MODEL_PATH="${TEKIZAI_FREETOKEN_MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-0731}"
export TEKIZAI_MODEL_REPO="${TEKIZAI_MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}"
export TEKIZAI_MODEL_BENCH_DTYPE="${TEKIZAI_MODEL_BENCH_DTYPE:-ds_fp4}"
export TEKIZAI_SERVED_MODEL="${TEKIZAI_SERVED_MODEL:-deepseek-v4-flash}"
export TEKIZAI_FREETOKEN_LOG="${TEKIZAI_FREETOKEN_LOG:-/workspace/logs/freetoken-deepseek-v4-flash.log}"

exec "$script_dir/vast_glm53_provision.sh"
