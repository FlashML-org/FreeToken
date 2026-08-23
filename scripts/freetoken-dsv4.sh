#!/usr/bin/env bash
# Start / stop DeepSeek-V4-Flash-0731 on the 4x RTX A4000, served on 0.0.0.0:8081.
#
#   ./freetoken-dsv4.sh start     start it (returns when it is really ready)
#   ./freetoken-dsv4.sh stop      stop it cleanly
#   ./freetoken-dsv4.sh status    is it up, and what is it holding
#   ./freetoken-dsv4.sh logs      follow the log
#   ./freetoken-dsv4.sh test      send one request
#
# Override any setting from the environment, e.g.
#   MEMORY_RATIO=0.88 ./freetoken-dsv4.sh start

set -uo pipefail

FT_DIR="${FT_DIR:-$HOME/FreeToken}"
MODEL="${MODEL:-$HOME/models/DeepSeek-V4-Flash-0731}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"
TP_SIZE="${TP_SIZE:-4}"
MEMORY_RATIO="${MEMORY_RATIO:-0.90}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
EXPERT_LOAD="${EXPERT_LOAD:-serial}"
# SPECULATIVE_DSPARK=1 builds and loads the checkpoint's dSpark drafter (the mtp.* stack).
# It costs 9.5 GiB of host expert banks and a slice of the GPU slot cache. Leave it off
# until the draft/verify loop is wired -- today it would load the drafter and never call it.
SPECULATIVE_DSPARK="${SPECULATIVE_DSPARK:-0}"
LOG="${LOG:-/tmp/freetoken-dsv4.log}"
PIDFILE="${PIDFILE:-/tmp/freetoken-dsv4.pid}"
START_TIMEOUT="${START_TIMEOUT:-2400}"     # seconds; the expert banks take a while

# nvcc rejects gcc newer than 15, and this host runs gcc 16. Point the JIT host pass
# at gcc-15, which is installed alongside it. Without this every kernel build fails
# with "unsupported GNU version".
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--ccbin /usr/bin/g++-15}"
export CXX="${CXX:-/usr/bin/g++-15}"
export CC="${CC:-/usr/bin/gcc-15}"

FT="$FT_DIR/.venv/bin/ft"

die() { echo "error: $*" >&2; exit 1; }

running_pid() {
    [ -f "$PIDFILE" ] || return 1
    local pid; pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    echo "$pid"
}

cmd_start() {
    [ -x "$FT" ] || die "no ft binary at $FT (run: cd $FT_DIR && uv venv && uv pip install -e '.[accel]')"
    [ -d "$MODEL" ] || die "no model directory at $MODEL"
    if running_pid >/dev/null; then
        echo "already running (pid $(running_pid)) on $HOST:$PORT"; return 0
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        die "port $PORT is already in use -- every inference service on this host shares it, so stop the other one first"
    fi

    local spec=()
    [ "$SPECULATIVE_DSPARK" = "1" ] && spec=(--speculative-dspark)

    echo "starting DeepSeek-V4-Flash on $HOST:$PORT (TP=$TP_SIZE, memory-ratio $MEMORY_RATIO${spec:+, dSpark drafter})"
    : > "$LOG"
    cd "$FT_DIR" || die "cannot cd to $FT_DIR"
    setsid "$FT" serve \
        --model "$MODEL" \
        --host "$HOST" --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --memory-ratio "$MEMORY_RATIO" \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --expert-load "$EXPERT_LOAD" \
        "${spec[@]}" \
        >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"

    # The HTTP frontend binds BEFORE the model loads, so /v1/models answers 200 while
    # generation would still fail. The log line below is the only honest readiness signal.
    echo "loading the expert banks -- this is the slow part; follow it with: $0 logs"
    local waited=0
    while [ "$waited" -lt "$START_TIMEOUT" ]; do
        if grep -aq "ready to serve" "$LOG"; then
            echo
            echo "ready on $HOST:$PORT after ${waited}s"
            grep -a "moe_cache_size\|Allocating .* tokens\|Weights:" "$LOG" \
                | sed "s/.*INFO *//" | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-120 | tail -3
            return 0
        fi
        if grep -aqE "Traceback|OutOfMemoryError|cannot be restarted" "$LOG"; then
            echo; echo "startup FAILED -- last lines:" >&2
            grep -aE "Error|OutOfMemory|assert" "$LOG" | tail -3 >&2
            echo "hint: an OOM during CUDA-graph capture means MEMORY_RATIO is too high." >&2
            cmd_stop >/dev/null 2>&1
            return 1
        fi
        sleep 10; waited=$((waited + 10))
        printf '.'
    done
    echo; die "still not ready after ${START_TIMEOUT}s -- see $LOG"
}

cmd_stop() {
    # SIGTERM, never SIGKILL. The host expert banks are cudaHostRegister'd shared
    # mappings; a SIGKILL'd rank leaves them behind (145 GB of ownerless Shmem, seen
    # in practice) and only a reboot gets it back.
    local pid; pid=$(running_pid) || pid=""
    if [ -z "$pid" ]; then
        pkill -TERM -f "ft serve --model $MODEL" 2>/dev/null
    else
        kill -TERM "$pid" 2>/dev/null
    fi
    echo -n "stopping"
    for _ in $(seq 1 30); do
        pgrep -f "ft serve --model $MODEL" >/dev/null 2>&1 || break
        sleep 2; printf '.'
    done
    echo
    rm -f "$PIDFILE"
    if pgrep -f "ft serve --model $MODEL" >/dev/null 2>&1; then
        echo "warning: something is still running. Give it longer before forcing it --" >&2
        echo "a SIGKILL here leaks the pinned host banks." >&2
        return 1
    fi
    echo "stopped"
}

cmd_status() {
    if grep -aq "ready to serve" "$LOG" 2>/dev/null && curl -sf -m 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
        echo "UP on $HOST:$PORT"
    elif running_pid >/dev/null; then
        echo "LOADING (pid $(running_pid)) -- not ready yet"
        tail -c 300 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -a . | tail -1
    else
        echo "DOWN"
    fi
    echo
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'
    free -g | sed -n '2p' | awk '{printf "  host ram: %s GiB used, %s GiB available of %s GiB\n", $3, $7, $2}'
}

cmd_logs() { tail -f "$LOG"; }

cmd_test() {
    curl -sf -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"DeepSeek-V4-Flash-0731",
             "messages":[{"role":"user","content":"Say hello in one short sentence."}],
             "max_tokens":32}' \
    && echo
}

case "${1:-}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    status) cmd_status ;;
    logs)   cmd_logs ;;
    test)   cmd_test ;;
    *) echo "usage: $0 {start|stop|restart|status|logs|test}"; exit 2 ;;
esac
