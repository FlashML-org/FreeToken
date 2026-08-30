#!/usr/bin/env bash
# Start the isolated FreeToken Qwen NVFP4 recovery server on LAN-223.
#
# This script never touches systemd, llama-swap, or the masked production
# llama.cpp service on port 18302. It launches one loopback-only FreeToken
# process on port 1919 and writes all output into a uniquely timestamped
# artifact directory so post-reboot results remain reproducible.

set -euo pipefail

# Keep every recovery run separate from previous logs and benchmark artifacts.
readonly RUN_ID="qwen-reboot-recovery-$(date -u +%Y%m%dT%H%M%SZ)"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly SOURCE_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
readonly MODEL_DIR="${ROOT_DIR}/models/Qwen3.6-35B-A3B-NVFP4"
# Pair a strict native cache with the exact source revision that built it.
# Unlike a generic Torch extension directory, the FreeToken cache identifies
# individual helper-kernel ABI names, so loading objects built from another
# source revision could silently defeat the no-JIT guarantee.
readonly SOURCE_REVISION="$(git -C "${SOURCE_DIR}" rev-parse --short=12 HEAD)"
readonly ROCM_KERNEL_CACHE_DIR="${FREETOKEN_ROCM_KERNEL_CACHE_DIR:-${ROOT_DIR}/cache/kernel-cache-rocm-gfx1151-${SOURCE_REVISION}}"
readonly MEMORY_RATIO="${FREETOKEN_MEMORY_RATIO:-0.35}"
# The previous 2,048-token reserve made the advertised 8,192-token sequence
# limit unreachable because --moe-cache-auto allocated the remaining budget to
# experts.  LAN-223 validation proved an 8,192-token reserve keeps zero swap,
# preserves short-decode TPS, and enables a real 6,856-token cold-prefill test.
# Permit a small, explicit set of recovery overrides for isolated experiments.
readonly KV_RESERVE_TOKENS="${FREETOKEN_KV_RESERVE_TOKENS:-8192}"
readonly CUDA_GRAPH_MAX_BS="${FREETOKEN_CUDA_GRAPH_MAX_BS:-0}"
readonly FP8_GEMV_BLOCK_N="${FREETOKEN_FP8_GEMV_BLOCK_N:-16}"
readonly FP8_GEMV_NUM_WARPS="${FREETOKEN_FP8_GEMV_NUM_WARPS:-1}"
readonly FP8_GEMV_SCALE_ACTIVATION="${FREETOKEN_FP8_GEMV_SCALE_ACTIVATION:-0}"
readonly MOE_COLLECT_STATS="${FREETOKEN_MOE_COLLECT_STATS:-0}"
readonly FUSED_COPY_BLOCKS_PER_BANK="${FREETOKEN_FUSED_COPY_BLOCKS_PER_BANK:-8}"
readonly ARTIFACT_DIR="${ROOT_DIR}/artifacts/${RUN_ID}"
readonly LOG_FILE="${ARTIFACT_DIR}/server.log"
readonly PID_FILE="${ARTIFACT_DIR}/server.pid"
readonly NATIVE_BUILD_LOG="${ARTIFACT_DIR}/native-extension-build.log"
readonly NATIVE_IMPORT_LOG="${ARTIFACT_DIR}/native-extension-import.txt"

# Refuse to launch if another process already owns the dedicated test port.
if ss -ltn "sport = :1919" | grep -q LISTEN; then
    echo "refusing to start: loopback benchmark port 1919 is already listening" >&2
    exit 1
fi

# Validate all immutable runtime inputs before starting a background process.
test -d "${SOURCE_DIR}"
test -x "${VENV_PYTHON}"
test -d "${MODEL_DIR}"
test -d "${ROCM_KERNEL_CACHE_DIR}"
case "${MEMORY_RATIO}" in
    0.[0-9][0-9]) ;;
    *) echo "invalid FREETOKEN_MEMORY_RATIO: ${MEMORY_RATIO}" >&2; exit 2 ;;
esac
case "${KV_RESERVE_TOKENS}" in
    2048|4096|8192) ;;
    *) echo "invalid FREETOKEN_KV_RESERVE_TOKENS: ${KV_RESERVE_TOKENS}" >&2; exit 2 ;;
esac
case "${CUDA_GRAPH_MAX_BS}" in
    0|1|2|4|8) ;;
    *) echo "invalid FREETOKEN_CUDA_GRAPH_MAX_BS: ${CUDA_GRAPH_MAX_BS}" >&2; exit 2 ;;
esac
case "${FP8_GEMV_BLOCK_N}" in
    16|32) ;;
    *) echo "invalid FREETOKEN_FP8_GEMV_BLOCK_N: ${FP8_GEMV_BLOCK_N}" >&2; exit 2 ;;
esac
case "${FP8_GEMV_NUM_WARPS}" in
    1|2|4) ;;
    *) echo "invalid FREETOKEN_FP8_GEMV_NUM_WARPS: ${FP8_GEMV_NUM_WARPS}" >&2; exit 2 ;;
esac
case "${FP8_GEMV_SCALE_ACTIVATION}" in
    0|1) ;;
    *) echo "invalid FREETOKEN_FP8_GEMV_SCALE_ACTIVATION: ${FP8_GEMV_SCALE_ACTIVATION}" >&2; exit 2 ;;
esac
case "${MOE_COLLECT_STATS}" in
    0|1) ;;
    *) echo "invalid FREETOKEN_MOE_COLLECT_STATS: ${MOE_COLLECT_STATS}" >&2; exit 2 ;;
esac
case "${FUSED_COPY_BLOCKS_PER_BANK}" in
    8|64) ;;
    *) echo "invalid FREETOKEN_FUSED_COPY_BLOCKS_PER_BANK: ${FUSED_COPY_BLOCKS_PER_BANK}" >&2; exit 2 ;;
esac
mkdir -p "${ARTIFACT_DIR}"

# These variables select the native ROCm toolchain and retain the existing HIP
# extension cache. Reusing the cache prevents a JIT build from contaminating the
# warm API benchmark that follows server readiness.
export PYTHONPATH="${SOURCE_DIR}/python"
export TORCH_EXTENSIONS_DIR="${ROOT_DIR}/cache/torch_extensions"
export ROCM_PATH="/opt/rocm-10.0"
export HIP_PATH="/opt/rocm-10.0"
export ROCM_HOME="/opt/rocm-10.0"
# The completed gfx1151 cache contains every valid C++/HIP helper in the
# FreeToken catalog. Make the server resolve objects only from that cache and
# fail explicitly if a source edit introduces a missing specialization. Triton
# keeps its own persistent code cache; this flag governs FreeToken's C++/HIP
# helper JIT rather than disabling native Triton execution.
export FREETOKEN_KERNEL_CACHE_DIR="${ROCM_KERNEL_CACHE_DIR}"
export FREETOKEN_DISABLE_JIT=1
# Pass the explicitly recorded FP8 output-row tile to the isolated process.
# The code permits only 16 (validated baseline) and 32 (a deterministic,
# quality-gated gfx1151 candidate), so an accidental shell value cannot create
# an untracked Triton specialization.
export FREETOKEN_FP8_GEMV_BLOCK_N="${FP8_GEMV_BLOCK_N}"
# Keep every additional kernel specialization explicit in the artifact's
# launch environment. This makes a subsequent quality failure attributable to
# one bounded variable rather than an implicit, inherited shell setting.
export FREETOKEN_FP8_GEMV_NUM_WARPS="${FP8_GEMV_NUM_WARPS}"
export FREETOKEN_FP8_GEMV_SCALE_ACTIVATION="${FP8_GEMV_SCALE_ACTIVATION}"
# Both fused-copy grid widths are precompiled into the strict gfx1151 cache.
# The default eight blocks is the established service baseline; sixty-four is
# an isolated, quality-gated copy-path candidate.
export FREETOKEN_FUSED_COPY_BLOCKS_PER_BANK="${FUSED_COPY_BLOCKS_PER_BANK}"

# Cache counters are opt-in because their atomic updates are diagnostic work.
# The default leaves the verified performance service unchanged, while an
# isolated launch can enable a single post-workload read-only snapshot.
EXTRA_ARGS=()
if [[ "${MOE_COLLECT_STATS}" == "1" ]]; then
    EXTRA_ARGS+=(--moe-collect-stats)
fi

# FreeToken's MoE offload path requires the in-tree pinned-tensor extension.
# A clean git worktree does not contain generated shared objects, so verify the
# import first and build the two native modules in that worktree only when it
# is absent. The build log is an artifact because the extension's compiler,
# ROCm headers, and link result are part of a reproducible HIP validation.
if ! "${VENV_PYTHON}" -c 'import freetoken.kernel._pinned_tensor' >/dev/null 2>&1; then
    (
        cd "${SOURCE_DIR}"
        "${VENV_PYTHON}" setup.py build_ext --inplace
    ) >"${NATIVE_BUILD_LOG}" 2>&1
fi
"${VENV_PYTHON}" -c \
    'import freetoken.kernel._pinned_tensor as pinned; print(pinned.__file__)' \
    >"${NATIVE_IMPORT_LOG}"

# The fixed policy is the validated LAN-223 Qwen configuration. The default
# 0.35 memory budget and 8,192-token KV reserve make the advertised context
# limit real while --moe-cache-auto retains as many MoE experts as safely fit.
# A constrained environment override supports isolated cache-capacity controls
# without editing the server command. Serial expert loading is the ROCm-correct route and prefill
# overlap stays disabled for the validated safe baseline. Graph capture defaults
# to zero because ROCm correctness takes priority; the bounded override enables
# an isolated batch-size experiment without changing the baseline command. The
# FP8 row-tile override changes neither split-K partitioning nor reduction order
# and is only used with a separately saved deterministic quality result. Wave
# count and activation scaling are likewise disabled defaults and require their
# own raw-output plus model-level quality evidence before any promotion.
nohup "${VENV_PYTHON}" -m freetoken.cli serve \
    --model-path "${MODEL_DIR}" \
    --served-model-name qwen3.6-35b-a3b-nvfp4-amd \
    --host 127.0.0.1 \
    --port 1919 \
    --attention-backend triton \
    --moe-backend offload \
    --nvfp4-backend triton \
    --expert-load serial \
    --moe-cache-auto \
    --memory-ratio "${MEMORY_RATIO}" \
    --max-seq-len-override 8192 \
    --kv-reserve-tokens "${KV_RESERVE_TOKENS}" \
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
    --disable-pynccl \
    --disable-moe-prefill-overlap \
    "${EXTRA_ARGS[@]}" \
    >"${LOG_FILE}" 2>&1 < /dev/null &

# Persist the child PID for diagnostics and explicit shutdown after the run.
echo "$!" >"${PID_FILE}"
printf '%s\n' "${ARTIFACT_DIR}"
