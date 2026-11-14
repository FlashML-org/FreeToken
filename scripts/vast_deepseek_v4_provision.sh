#!/usr/bin/env bash
set -Eeuo pipefail

# DeepSeek-V4-Flash uses the same cacheable Vast/PyWorker bootstrap as GLM.
# Keep the model-specific contract in this public wrapper so Vast workers can
# cold-start without credentials for a separate deployment repository.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
shared_provisioner="$script_dir/vast_glm53_provision.sh"
freetoken_ref="${TEKIZAI_FREETOKEN_REF:-feat/glm53-flash}"

export TEKIZAI_MODEL_SOURCE_PATH="${TEKIZAI_MODEL_SOURCE_PATH:-/workspace/models/DeepSeek-V4-Flash-0731-hf}"
export TEKIZAI_FREETOKEN_MODEL_PATH="${TEKIZAI_FREETOKEN_MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-0731-ftw}"
export TEKIZAI_CONVERT_FTW="${TEKIZAI_CONVERT_FTW:-1}"
export TEKIZAI_MODEL_REPO="${TEKIZAI_MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}"
export TEKIZAI_MODEL_BENCH_DTYPE="${TEKIZAI_MODEL_BENCH_DTYPE:-ds_fp4}"
export TEKIZAI_SERVED_MODEL="${TEKIZAI_SERVED_MODEL:-deepseek-v4-flash}"
export TEKIZAI_FREETOKEN_LOG="${TEKIZAI_FREETOKEN_LOG:-/workspace/logs/freetoken-deepseek-v4-flash.log}"

if [[ ! -x "$shared_provisioner" ]]; then
  curl -fsSL \
    "https://raw.githubusercontent.com/earlvanze/FreeToken/${freetoken_ref}/scripts/vast_glm53_provision.sh" \
    -o "$shared_provisioner"
  chmod 0755 "$shared_provisioner"
fi

exec "$shared_provisioner"
