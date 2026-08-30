#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${WORKSPACE_DIR:-/workspace}"
freetoken_dir="${TEKIZAI_FREETOKEN_DIR:-${workspace}/freetoken}"
model_dir="${TEKIZAI_FREETOKEN_MODEL_PATH:-${workspace}/models/GLM-5.3-Flash-NVFP4}"
worker_source_dir="${TEKIZAI_WORKER_SOURCE_DIR:-${workspace}/freetoken-vast-worker}"
repo="${TEKIZAI_FREETOKEN_REPO:-https://github.com/earlvanze/FreeToken.git}"
ref="${TEKIZAI_FREETOKEN_REF:-feat/glm53-flash}"
model_repo="${TEKIZAI_MODEL_REPO:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
bootstrap_ref="${TEKIZAI_PYWORKER_BOOTSTRAP_REF:-2207a3f94b55a0921c1641520eeb83de5a0c1611}"

export DEBIAN_FRONTEND=noninteractive
export PATH="${HOME}/.local/bin:${PATH}"
export MODEL_LOG="${TEKIZAI_FREETOKEN_LOG:-${workspace}/logs/freetoken-glm53.log}"
export PYWORKER_REPO="${PYWORKER_REPO:-$repo}"
export PYWORKER_REF="${PYWORKER_REF:-$ref}"

mkdir -p "$workspace" "$(dirname "$MODEL_LOG")"
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates cuda-compiler-13-0 cuda-cudart-dev-13-0 \
  curl git ninja-build numactl util-linux
apt-get clean

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

checkout_ref() {
  local destination="$1"
  if [[ ! -d "$destination/.git" ]]; then
    mkdir -p "$destination"
    git -C "$destination" init
    git -C "$destination" remote add origin "$repo"
  fi
  git -C "$destination" fetch --depth 1 origin "$ref"
  git -C "$destination" checkout --detach --force FETCH_HEAD
}

checkout_ref "$freetoken_dir"
checkout_ref "$worker_source_dir"

if [[ ! -x "$freetoken_dir/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$freetoken_dir/.venv"
fi
uv pip install --python "$freetoken_dir/.venv/bin/python" -e "$freetoken_dir[accel]"
"$freetoken_dir/.venv/bin/hf" download "$model_repo" --local-dir "$model_dir"
"$freetoken_dir/.venv/bin/ft" bench bw --dtype nvfp4

"$freetoken_dir/scripts/vast_glm53_start.sh" &
model_launcher_pid=$!

cleanup() {
  if kill -0 "$model_launcher_pid" 2>/dev/null; then
    kill "$model_launcher_pid"
    wait "$model_launcher_pid" || true
  fi
}
trap cleanup EXIT INT TERM

bootstrap="${workspace}/vast-pyworker-bootstrap.sh"
curl -fsSL \
  "https://raw.githubusercontent.com/vast-ai/pyworker/${bootstrap_ref}/start_server.sh" \
  -o "$bootstrap"
chmod 0755 "$bootstrap"
"$bootstrap"
