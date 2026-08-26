#!/usr/bin/env bash
# serve-qwen-moe.sh
#
# Serve the Qwen3.5/3.6-35B-A3B hybrid MoE GGUF (GatedDeltaNet + full-attention)
# on AMD ROCm (gfx1100 / RX 7900 XTX). Uses the offload MoE backend (experts on the
# CPU/offload cache) and the triton attention backend by default.
#
# Native context window: 262144 (256K) tokens (max_position_embeddings in the model).
# Graph capture is settled as a failure on ROCm, so this runs eager kernel-launch decode.
#
# VS CODE NOTES:
#   - The server command is built as a bash array and launched on ONE physical
#     line. Backslash-newline continuations get mangled by some VS Code shells /
#     task runners (each continuation line then executes as its own command),
#     which is exactly how `nohup.out` ended up with bare "--model: command not
#     found" errors. Do NOT reintroduce multi-line command strings here.
#   - .vscode/settings.json pins files.eol=\n for *.sh; the CRLF guard below
#     catches any violation early instead of failing obscurely mid-launch.
#
# Usage:
#   ./serve-qwen-moe.sh                 # launch on 127.0.0.1:1920, triton attention
#   FT_ATTN=torch ./serve-qwen-moe.sh   # A/B against the pure-torch reference backend
#   FT_PORT=1930 ./serve-qwen-moe.sh    # pick another port
#   ./serve-qwen-moe.sh stop            # kill the running server
#   ./serve-qwen-moe.sh status          # running? + tail of the log
#
# Or from the VS Code Command Palette: "Tasks: Run Task" -> FreeToken: ...

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv-rocm/bin/python}"

MODEL="${FT_MODEL:-}"
HOST="${FT_HOST:-127.0.0.1}"
PORT="${FT_PORT:-1920}"
ATNN="${FT_ATTN:-triton}"        # triton | torch
MOE_BACKEND="${FT_MOE:-offload}" # offload required for K-quant experts
# radix = cross-request GDN-state prefix reuse (default); naive = no prefix reuse.
# Suspect for long-context degradation: radix GDN-state reuse corrupting later requests.
CACHE_TYPE="${FT_CACHE_TYPE:-radix}"
# MoE cache hit/miss stats in the decode log line (--moe-collect-stats).
MOE_STATS="${FT_MOE_STATS:-1}"
# Disable the two-buffer prefill MoE overlap (diagnostic: race check).
PREFILL_OVERLAP="${FT_PREFILL_OVERLAP:-1}"
# MoE layers computed on the CPU executor (diagnostic: '0' = all-GPU, no CPU path).
CPU_LAYERS="${FT_CPU_LAYERS:-}"
# --num-tokens sizes the KV cache in tokens (hybrid arch: only 10/40 layers are full
# attention, 2 kv heads x 256 dim -> ~20 KiB/token bf16; 128k ~= 2.7 GiB).
# The GPU expert-slot cache is sized by FT_MOE_CACHE:
#   auto    -> --moe-cache-auto: the engine derives slot bytes from the real expert
#              tensors and fills all free VRAM AFTER reserving kv-reserve-tokens for KV.
#              --kv-reserve-tokens MUST equal --num-tokens here: with an explicit
#              --num-tokens the engine skips auto's own KV-half plan (num_page_override
#              is set), so without a matching reservation greedy expert fill would eat
#              VRAM the pinned KV still needs -> late CUDA OOM.
#   <int>   -> fixed --moe-cache-size N slots (legacy behavior).
# Headroom: the engine may use FT_MEMORY_RATIO of free VRAM for weights+KV+experts
# combined (default here 0.80, upstream default 0.9). The remainder absorbs prefill
# transients -- the MoE overlap double-buffer alone needs ~3.8 GiB at this model's
# batch shapes; 0.9 left only ~2 GiB and OOM'd mid-prefill under VS Code payloads.
MEMORY_RATIO="${FT_MEMORY_RATIO:-0.80}"
# --max-prefill-length caps chunked-prefill chunk size (engine default 8192). The lm_head
# materializes logits for EVERY chunk token: an 8192-token chunk spikes ~3.8 GiB
# transiently -- enough to OOM on VS Code-sized prompts even with healthy headroom.
# 4096 halves the spike; long prompts just prefill in more chunks.
PREFILL_CHUNK="${FT_PREFILL_CHUNK:-4096}"
KV_TOKENS="${FT_KV_TOKENS:-131072}"
# Default max output tokens for requests that omit max_tokens. The reasoning model
# sometimes needs more room to finish its reasoning before answering; the engine
# default is 32k. Copilot sends its own max_tokens, which we cannot override, but
# other clients inherit this.
MAX_OUTPUT="${FT_MAX_OUTPUT:-65536}"
MOE_CACHE="${FT_MOE_CACHE:-auto}"
LOG="${FT_LOG:-/tmp/serve_qwen_moe.log}"

die() { echo "ERROR: $*" >&2; exit 1; }

# Map FT_MOE_CACHE to argv. Only the auto path carries --kv-reserve-tokens:
# with a fixed size the CLI already validates fit against the pinned KV.
# NOTE: validation must run in THIS shell (no process substitution), or die()
# would only exit a subshell and the launch would continue unvalidated.

# Guard against the exact failure mode VS Code caused before: if this file ever
# gets saved with CRLF endings, every argument silently grows a trailing \r.
if grep -q $'\r' "${BASH_SOURCE[0]}"; then
    die "CRLF line endings detected in $(basename "${BASH_SOURCE[0]}"). Run: sed -i 's/\r$//' ${BASH_SOURCE[0]}"
fi

[ -x "$PY" ] || die "python not found: $PY (set PY=/path/to/venv-python)"

server_pids() {
    pgrep -f "freetoke[n].cli serve" || true
}

start() {
    [ -n "$MODEL" ] || die "no model configured: set FT_MODEL=/path/to/model.gguf"
    if [ -n "$(server_pids)" ]; then
        echo "A server is already running (pid $(server_pids | tr '\n' ' ')). Use 'stop' first."
        exit 1
    fi
    [ -f "$MODEL" ] || die "model not found: $MODEL"

    # One arg per array element; expanded once, single line, no continuations.
    local -a SERVE_ARGS=(
        "--model" "$MODEL"
        "--moe-backend" "$MOE_BACKEND"
        "--cache-type" "$CACHE_TYPE"
        $( [ "$MOE_STATS" = 1 ] && echo "--moe-collect-stats" )
        $( [ "$PREFILL_OVERLAP" = 0 ] && echo "--disable-moe-prefill-overlap" )
        $( [ -n "$CPU_LAYERS" ] && echo "--moe-cpu-layers" "$CPU_LAYERS" )
        "--attention-backend" "$ATNN"
        "--num-tokens" "$KV_TOKENS"
        "--memory-ratio" "$MEMORY_RATIO"
        "--max-prefill-length" "$PREFILL_CHUNK"
        "--max-output-tokens" "$MAX_OUTPUT"
        "--host" "$HOST"
        "--port" "$PORT"
    )
    case "$MOE_CACHE" in
        auto)
            SERVE_ARGS+=("--moe-cache-auto" "--kv-reserve-tokens" "$KV_TOKENS")
            ;;
        ''|*[!0-9]*)
            die "FT_MOE_CACHE='$MOE_CACHE' is invalid: use 'auto' or a slot count"
            ;;
        *)
            SERVE_ARGS+=("--moe-cache-size" "$MOE_CACHE")
            ;;
    esac

    echo "Launching FreeToken server"
    echo "  model   : $MODEL"
    echo "  listen  : $HOST:$PORT"
    echo "  attn    : $ATNN"
    echo "  moe     : $MOE_BACKEND"
    echo "  kv      : $KV_TOKENS tokens (gpu moe cache: ${MOE_CACHE}${MOE_CACHE:+ }$( [ "$MOE_CACHE" = auto ] && echo "kv-reserve $KV_TOKENS" || echo slots))"
    echo "  python  : $PY"
    echo "  log     : $LOG"

    # setsid: give the server its OWN session/process group. nohup alone only
    # ignores SIGHUP — a caller that dies (e.g. a VS Code task cancelled, an agent
    # tool timeout) still takes down the whole process group with SIGKILL/SIGTERM,
    # which silently killed the server mid model-load once already.
    cd "$REPO"
    PYTHONPATH="$REPO/python" setsid nohup "$PY" -m freetoken.cli serve "${SERVE_ARGS[@]}" >"$LOG" 2>&1 </dev/null &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "pid=$pid — waiting for readiness (model load takes ~3-4 min)..."
}

wait_ready() {
    for _ in $(seq 1 90); do
        if grep -q "API server is ready" "$LOG" 2>/dev/null; then
            echo "READY on $HOST:$PORT"
            return 0
        fi
        if [ -z "$(server_pids)" ]; then
            echo "server exited; last log lines:" >&2
            tail -20 "$LOG" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "timed out waiting for readiness; see $LOG" >&2
    return 1
}

stop() {
    local pids
    pids="$(server_pids)"
    if [ -z "$pids" ]; then
        echo "no server running"
    else
        pkill -9 -f "freetoke[n].cli serve" 2>/dev/null || true
        echo "stopped (was pid $pids)"
    fi
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
    # free the distributed worker port (server_port+1)
    local p pid
    for p in "$PORT" "$((PORT + 1))"; do
        pid="$(ss -ltnp 2>/dev/null | grep ":$p" | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

status() {
    local pids
    pids="$(server_pids)"
    if [ -n "$pids" ]; then
        echo "RUNNING (pid $(echo "$pids" | tr '\n' ' ')) on $HOST:$PORT"
    else
        echo "NOT RUNNING"
    fi
    # Surface the auto-resolved expert-cache split so users don't have to read
    # engine log lines; only present when FT_MOE_CACHE=auto booted the server.
    if [ -f "$LOG" ]; then
        local resolved
        resolved="$(grep -o -- '--moe-cache-auto resolved moe_cache_size=[0-9]* num_pages=[0-9]*' "$LOG" | tail -1)"
        [ -n "$resolved" ] && echo "moe cache: ${resolved//--moe-cache-auto resolved /}"
    fi
    [ -f "$LOG" ] && echo "--- last 5 log lines ($LOG) ---" && tail -5 "$LOG"
}

case "${1:-start}" in
    start)  start && wait_ready ;;
    stop)   stop ;;
    status) status ;;
    log)    exec tail -f "$LOG" ;;
    *) echo "usage: $0 [start|stop|status|log]" >&2; exit 1 ;;
esac
