#!/usr/bin/env bash
set -Eeuo pipefail

model_path="${TEKIZAI_FREETOKEN_MODEL_PATH:-/workspace/models/GLM-5.3-Flash-NVFP4}"
served_model="${TEKIZAI_SERVED_MODEL:-glm-5.3-flash-nvfp4}"
ft_executable="${TEKIZAI_FREETOKEN_EXECUTABLE:-/workspace/freetoken/.venv/bin/ft}"
port="${TEKIZAI_FREETOKEN_PORT:-1919}"
log_file="${TEKIZAI_FREETOKEN_LOG:-/workspace/logs/freetoken-glm53.log}"

mkdir -p "$(dirname "$log_file")"
: >"$log_file"

serve_cmd=("$ft_executable" serve \
  --model "$model_path" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --moe-backend auto \
  --moe-cpu-threads "${TEKIZAI_CPU_THREADS:-48}" \
  --memory-ratio "${TEKIZAI_MEMORY_RATIO:-0.95}" \
  --max-seq-len-override "${TEKIZAI_MAX_SEQ_LEN:-32768}" \
  --max-running-requests 1 \
  --disable-moe-prefill-overlap)

gpu_cpu_affinity="${TEKIZAI_GPU_CPU_AFFINITY:-}"
if [[ -z "$gpu_cpu_affinity" ]] && command -v nvidia-smi >/dev/null; then
  gpu_cpu_affinity="$(nvidia-smi topo -m 2>/dev/null | awk '$1 == "GPU0" { print $3; exit }')"
fi
if [[ "$gpu_cpu_affinity" =~ ^[0-9,-]+$ ]] && command -v taskset >/dev/null; then
  serve_cmd=(taskset -c "$gpu_cpu_affinity" "${serve_cmd[@]}")
fi

"${serve_cmd[@]}" >>"$log_file" 2>&1 &
backend_pid=$!

on_exit() {
  if kill -0 "$backend_pid" 2>/dev/null; then
    kill "$backend_pid"
    wait "$backend_pid" || true
  fi
}
trap on_exit EXIT INT TERM

for _ in $(seq 1 "${TEKIZAI_READY_POLLS:-900}"); do
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    printf '%s\n' 'FREETOKEN_SERVER_EXITED' >>"$log_file"
    wait "$backend_pid"
  fi
  if curl --fail --silent --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    printf '%s\n' 'FREETOKEN_SERVER_READY' >>"$log_file"
    wait "$backend_pid"
    exit $?
  fi
  sleep 2
done

printf '%s\n' 'FREETOKEN_SERVER_EXITED readiness_timeout' >>"$log_file"
exit 1
