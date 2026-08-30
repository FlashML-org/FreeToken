#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${WORKSPACE_DIR:-/workspace}"
freetoken_dir="${TEKIZAI_FREETOKEN_DIR:-${workspace}/freetoken}"
model_dir="${TEKIZAI_FREETOKEN_MODEL_PATH:-${workspace}/models/GLM-5.3-Flash-NVFP4}"
worker_source_dir="${TEKIZAI_WORKER_SOURCE_DIR:-${workspace}/vast-pyworker}"
pyworker_uv_cache="${TEKIZAI_PYWORKER_UV_CACHE:-${workspace}/pyworker-uv-cache}"
repo="${TEKIZAI_FREETOKEN_REPO:-https://github.com/earlvanze/FreeToken.git}"
ref="${TEKIZAI_FREETOKEN_REF:-feat/glm53-flash}"
expected_commit="${TEKIZAI_FREETOKEN_EXPECTED_COMMIT:-}"
model_repo="${TEKIZAI_MODEL_REPO:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
bootstrap_ref="${TEKIZAI_PYWORKER_BOOTSTRAP_REF:-2207a3f94b55a0921c1641520eeb83de5a0c1611}"
bootstrap="${workspace}/vast-pyworker-bootstrap.sh"
provision_marker="${workspace}/.tekizai-glm53-provisioned"
provision_marker_value="${expected_commit:-$ref}"

export DEBIAN_FRONTEND=noninteractive
export PATH="${HOME}/.local/bin:${PATH}"
export MODEL_LOG="${TEKIZAI_FREETOKEN_LOG:-${workspace}/logs/freetoken-glm53.log}"
export PYWORKER_REPO="${PYWORKER_REPO:-$repo}"
export PYWORKER_REF="${PYWORKER_REF:-$ref}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "$workspace" "$(dirname "$MODEL_LOG")"

model_download_pid=""
pyworker_pid=""
model_launcher_pid=""
cleanup() {
  local pid
  for pid in "$model_launcher_pid" "$pyworker_pid" "$model_download_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      wait "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_runtime() {
  echo "FREETOKEN_PROVISION_STAGE=pyworker_start"
  UV_CACHE_DIR="$pyworker_uv_cache" \
    USE_SYSTEM_PYTHON=true \
    ROTATE_MODEL_LOG=true \
    "$bootstrap" &
  pyworker_pid=$!

  echo "FREETOKEN_PROVISION_STAGE=model_start"
  "$freetoken_dir/scripts/vast_glm53_start.sh" &
  model_launcher_pid=$!

  wait "$pyworker_pid"
}

checkout_matches_expected_commit() {
  local checkout
  [[ -n "$expected_commit" ]] || return 0
  for checkout in "$freetoken_dir" "$worker_source_dir"; do
    [[ -d "$checkout/.git" ]] || return 1
    [[ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null)" == "$expected_commit" ]] || return 1
  done
}

if [[ -f "$provision_marker" ]] \
  && [[ "$(<"$provision_marker")" == "$provision_marker_value" ]] \
  && [[ -x "$bootstrap" ]] \
  && [[ -x "$freetoken_dir/.venv/bin/ft" ]] \
  && [[ -x "$freetoken_dir/scripts/vast_glm53_start.sh" ]] \
  && [[ -s "$model_dir/config.json" ]] \
  && [[ -x /usr/local/cuda-13.0/bin/nvcc ]] \
  && checkout_matches_expected_commit; then
  echo "FREETOKEN_PROVISION_STAGE=fast_resume"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  start_runtime
  exit $?
fi

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

download_model &
model_download_pid=$!

apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates cuda-compiler-13-0 cuda-cudart-dev-13-0 \
  curl git libcurand-dev-13-0 ninja-build numactl python3.12-dev util-linux
apt-get clean

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

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

if [[ -n "$expected_commit" ]]; then
  for checkout in "$freetoken_dir" "$worker_source_dir"; do
    actual_commit="$(git -C "$checkout" rev-parse HEAD)"
    if [[ "$actual_commit" != "$expected_commit" ]]; then
      printf 'Expected FreeToken commit %s at %s, got %s\n' \
        "$expected_commit" "$checkout" "$actual_commit" >&2
      exit 1
    fi
  done
fi

echo "FREETOKEN_PROVISION_STAGE=pyworker_bootstrap"
curl -fsSL \
  "https://raw.githubusercontent.com/vast-ai/pyworker/${bootstrap_ref}/start_server.sh" \
  -o "$bootstrap"
chmod 0755 "$bootstrap"
UV_CACHE_DIR="$pyworker_uv_cache" \
  USE_SYSTEM_PYTHON=true \
  ROTATE_MODEL_LOG=true \
  "$bootstrap" &
pyworker_pid=$!

echo "FREETOKEN_PROVISION_STAGE=dependencies"
if [[ ! -x "$freetoken_dir/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$freetoken_dir/.venv"
fi
uv pip install --python "$freetoken_dir/.venv/bin/python" -e "$freetoken_dir[accel]"

wait "$model_download_pid"
echo "FREETOKEN_PROVISION_STAGE=bandwidth_check"
"$freetoken_dir/.venv/bin/ft" bench bw --dtype nvfp4

printf '%s\n' "$provision_marker_value" >"$provision_marker"

echo "FREETOKEN_PROVISION_STAGE=model_start"
"$freetoken_dir/scripts/vast_glm53_start.sh" &
model_launcher_pid=$!

wait "$pyworker_pid"
