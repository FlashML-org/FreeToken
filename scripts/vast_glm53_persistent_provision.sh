#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${WORKSPACE_DIR:-/workspace}"
freetoken_dir="${TEKIZAI_FREETOKEN_DIR:-${workspace}/freetoken}"
model_dir="${TEKIZAI_FREETOKEN_MODEL_PATH:-${workspace}/models/GLM-5.3-Flash-NVFP4}"
repo="${TEKIZAI_FREETOKEN_REPO:-https://github.com/earlvanze/FreeToken.git}"
ref="${TEKIZAI_FREETOKEN_REF:-feat/glm53-flash}"
expected_commit="${TEKIZAI_FREETOKEN_EXPECTED_COMMIT:-}"
model_repo="${TEKIZAI_MODEL_REPO:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"

export DEBIAN_FRONTEND=noninteractive
export PATH="${HOME}/.local/bin:${PATH}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "$workspace" "${workspace}/logs"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

download_model() {
  local attempt status
  for attempt in 1 2 3 4 5; do
    echo "FREETOKEN_PROVISION_STAGE=model_download attempt=${attempt}"
    if uvx --from huggingface-hub==1.29.0 hf download "$model_repo" --local-dir "$model_dir"; then
      echo "FREETOKEN_PROVISION_STAGE=model_download_complete"
      return 0
    else
      status=$?
    fi
    echo "FREETOKEN_PROVISION_STAGE=model_download_retry status=${status}"
    sleep "$((attempt * 10))"
  done
  return "$status"
}

model_download_pid=""
cleanup() {
  if [[ -n "$model_download_pid" ]] && kill -0 "$model_download_pid" 2>/dev/null; then
    kill "$model_download_pid"
    wait "$model_download_pid" || true
  fi
}
trap cleanup EXIT INT TERM

download_model &
model_download_pid=$!

apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates cuda-compiler-13-0 cuda-cudart-dev-13-0 \
  cuda-curand-dev-13-0 \
  curl git ninja-build numactl python3.12-dev util-linux
apt-get clean

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -d "$freetoken_dir/.git" ]]; then
  mkdir -p "$freetoken_dir"
  git -C "$freetoken_dir" init
  git -C "$freetoken_dir" remote add origin "$repo"
fi
git -C "$freetoken_dir" fetch --depth 1 origin "$ref"
git -C "$freetoken_dir" checkout --detach --force FETCH_HEAD
if [[ -n "$expected_commit" ]]; then
  resolved_commit="$(git -C "$freetoken_dir" rev-parse HEAD)"
  if [[ "$resolved_commit" != "$expected_commit" && "$resolved_commit" != "$expected_commit"* ]]; then
    echo "FreeToken ref resolved to unexpected commit: ${resolved_commit}" >&2
    exit 1
  fi
fi

echo "FREETOKEN_PROVISION_STAGE=dependencies"
if [[ ! -x "$freetoken_dir/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$freetoken_dir/.venv"
fi
uv pip install --python "$freetoken_dir/.venv/bin/python" -e "$freetoken_dir[accel]"

wait "$model_download_pid"
model_download_pid=""

echo "FREETOKEN_PROVISION_STAGE=bandwidth_check"
"$freetoken_dir/.venv/bin/ft" bench bw --dtype nvfp4

echo "FREETOKEN_PROVISION_STAGE=model_start"
exec "$freetoken_dir/scripts/vast_glm53_start.sh"
