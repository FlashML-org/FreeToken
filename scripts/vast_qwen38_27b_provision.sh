#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen3.8-27B NVFP4 uses the cacheable Vast/PyWorker bootstrap shared with
# GLM and DeepSeek. Keep source and FTW paths distinct so conversion is
# resumable and a stopped worker can restart directly from the FTW checkpoint.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
shared_provisioner="$script_dir/vast_glm53_provision.sh"
freetoken_ref="${TEKIZAI_FREETOKEN_REF:-feat/glm53-flash}"

export TEKIZAI_MODEL_SOURCE_PATH="${TEKIZAI_MODEL_SOURCE_PATH:-/workspace/models/Qwen3.8-27B-NVFP4-hf}"
export TEKIZAI_FREETOKEN_MODEL_PATH="${TEKIZAI_FREETOKEN_MODEL_PATH:-/workspace/models/Qwen3.8-27B-NVFP4-ftw}"
export TEKIZAI_CONVERT_FTW="${TEKIZAI_CONVERT_FTW:-1}"
export TEKIZAI_MODEL_REPO="${TEKIZAI_MODEL_REPO:-RadixArk/Qwen3.8-27B-NVFP4}"
export TEKIZAI_MODEL_BENCH_DTYPE="${TEKIZAI_MODEL_BENCH_DTYPE:-nvfp4}"
export TEKIZAI_SERVED_MODEL="${TEKIZAI_SERVED_MODEL:-qwen3.8:27b}"
export TEKIZAI_FREETOKEN_LOG="${TEKIZAI_FREETOKEN_LOG:-/workspace/logs/freetoken-qwen38-27b.log}"
export TEKIZAI_PROVISION_MARKER="${TEKIZAI_PROVISION_MARKER:-/workspace/.tekizai-qwen38-27b-provisioned}"
export TEKIZAI_MEMORY_RATIO="${TEKIZAI_MEMORY_RATIO:-0.90}"
export TEKIZAI_MAX_SEQ_LEN="${TEKIZAI_MAX_SEQ_LEN:-8192}"
export TEKIZAI_MAX_RUNNING_REQUESTS="${TEKIZAI_MAX_RUNNING_REQUESTS:-1}"

if [[ ! -x "$shared_provisioner" ]]; then
  curl -fsSL \
    "https://raw.githubusercontent.com/earlvanze/FreeToken/${freetoken_ref}/scripts/vast_glm53_provision.sh" \
    -o "$shared_provisioner"
  chmod 0755 "$shared_provisioner"
fi

exec "$shared_provisioner"
