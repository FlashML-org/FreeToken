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

MODEL="${FT_MODEL:-/media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
HOST="${FT_HOST:-127.0.0.1}"
PORT="${FT_PORT:-1920}"
ATNN="${FT_ATTN:-triton}"        # triton | torch
MOE_BACKEND="${FT_MOE:-offload}" # offload required for K-quant experts
# VS Code / Copilot injects a large chat context, so the KV cache must be bigger than
# the tiny 8K the MoE-auto cache leaves. --num-tokens sizes the KV cache in tokens;
# --moe-cache-size limits the GPU expert cache so KV has room (fewer slots = slower decode).
# Hybrid arch: only 10/40 layers are full attention (2 kv heads x 256 dim) ->
# ~20 KiB/token bf16, i.e. 128k ~= 2.7 GiB, full native 256k ~= 5.4 GiB.
KV_TOKENS="${FT_KV_TOKENS:-131072}"
MOE_CACHE="${FT_MOE_CACHE:-2048}"
LOG="${FT_LOG:-/tmp/serve_qwen_moe.log}"

die() { echo "ERROR: $*" >&2; exit 1; }

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
    if [ -n "$(server_pids)" ]; then
        echo "A server is already running (pid $(server_pids | tr '\n' ' ')). Use 'stop' first."
        exit 1
    fi
    [ -f "$MODEL" ] || die "model not found: $MODEL"

    # One arg per array element; expanded once, single line, no continuations.
    local -a SERVE_ARGS=(
        "--model" "$MODEL"
        "--moe-backend" "$MOE_BACKEND"
        "--attention-backend" "$ATNN"
        "--moe-cache-size" "$MOE_CACHE"
        "--num-tokens" "$KV_TOKENS"
        "--host" "$HOST"
        "--port" "$PORT"
    )

    echo "Launching FreeToken server"
    echo "  model   : $MODEL"
    echo "  listen  : $HOST:$PORT"
    echo "  attn    : $ATNN"
    echo "  moe     : $MOE_BACKEND"
    echo "  kv      : $KV_TOKENS tokens (gpu moe cache: $MOE_CACHE slots)"
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
    [ -f "$LOG" ] && echo "--- last 5 log lines ($LOG) ---" && tail -5 "$LOG"
}

case "${1:-start}" in
    start)  start && wait_ready ;;
    stop)   stop ;;
    status) status ;;
    log)    exec tail -f "$LOG" ;;
    *) echo "usage: $0 [start|stop|status|log]" >&2; exit 1 ;;
esac
