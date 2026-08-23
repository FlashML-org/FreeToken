#!/usr/bin/env bash
# Start / stop DeepSeek-V4-Flash-0731 on the 4x RTX A4000, served on 0.0.0.0:8081.
#
#   ./freetoken-dsv4.sh start     start it (returns when it is really ready)
#   ./freetoken-dsv4.sh stop      stop it cleanly (add --force to escalate)
#   ./freetoken-dsv4.sh kill      SIGKILL a wedged rank, then reclaim /dev/shm
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
# SPECULATIVE_DSPARK=0 turns OFF the checkpoint's dSpark drafter (the mtp.* stack).
# On by default: the draft/verify loop is wired, and dSpark is what the checkpoint was
# trained to decode with. It costs 9.5 GiB of host expert banks and a slice of the GPU
# slot cache. Set 0 to measure a plain-decode baseline -- the drafter's 3 MoE layers then
# disappear from the log's `layers=` count, which is the quickest way to tell which
# mode a run was in.
SPECULATIVE_DSPARK="${SPECULATIVE_DSPARK:-1}"
LOG="${LOG:-/tmp/freetoken-dsv4.log}"
# The TP ranks' torch.distributed rendezvous. Held by every rank, not just the
# frontend, so it is the one that lingers after a stop.
RDZV_PORT="${RDZV_PORT:-8082}"
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


# Every process belonging to this service, as PIDs.
#
# The TP ranks are multiprocessing.spawn children: their cmdline is a generic
#   python3 -c from multiprocessing.spawn import spawn_main; spawn_main(...)
# with no "ft serve" and no model path in it, and launch.py detaches them into their own
# process group. So neither a pattern match on the serve command nor a process-group kill
# reaches them -- which is why a "stopped" service could still be holding its ports, and
# why the ranks had to be found by hand more than once.
#
# Two handles that do work: the venv interpreter they were spawned from (specific to this
# checkout), and whoever is listening on the service's ports.
service_pids() {
    {
        pgrep -u "$(id -u)" -f "ft serve --model $MODEL" 2>/dev/null
        pgrep -u "$(id -u)" -f "^$FT_DIR/.venv/bin/python3 -c from multiprocessing.spawn" 2>/dev/null
        ss -ltnp 2>/dev/null \
            | grep -E ":($PORT|$RDZV_PORT) " \
            | grep -oE 'pid=[0-9]+' | cut -d= -f2
    } | sort -un
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

    # The TP ranks rendezvous over a second port, and it outlives the API port on a
    # restart: the previous run's ranks hold it while they tear down. Starting into that
    # window fails every rank at once with a message about the wrong thing --
    #   DistNetworkError: ... port: 8082 ... EADDRINUSE
    # -- after the model has already begun loading. So wait for it, and say what for.
    local waited=0
    while ss -ltn 2>/dev/null | grep -q ":$RDZV_PORT "; do
        if [ "$waited" -eq 0 ]; then
            echo -n "waiting for the TP rendezvous port $RDZV_PORT to be released"
        fi
        [ "$waited" -ge 60 ] && { echo; die "port $RDZV_PORT still held after 60s -- a previous rank is stuck; try '$0 kill'"; }
        sleep 2; waited=$((waited + 2)); printf '.'
    done
    [ "$waited" -gt 0 ] && echo " ok"

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
    local pids; pids=$(service_pids)
    [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
    echo -n "stopping"
    for _ in $(seq 1 30); do
        [ -z "$(service_pids)" ] && break
        sleep 2; printf '.'
    done
    echo
    rm -f "$PIDFILE"
    if [ -n "$(service_pids)" ]; then
        if [ "${1:-}" = "--force" ]; then
            echo "still running after 60s -- forcing" >&2
            cmd_kill
            return $?
        fi
        echo "warning: something is still running. Give it longer before forcing it --" >&2
        echo "a SIGKILL here leaks the pinned host banks. Use '$0 kill' to force." >&2
        return 1
    fi
    # "stopped" must mean "ready to start again". The ranks release the rendezvous port
    # slightly after the processes go, and a start inside that window fails every rank.
    for _ in $(seq 1 15); do
        ss -ltn 2>/dev/null | grep -q ":$RDZV_PORT " || break
        sleep 1
    done
    echo "stopped"
}

# SIGKILL, plus the cleanup that makes it survivable.
#
# A killed rank never runs its atexit, so the host expert banks stay cudaHostRegister'd:
# ownerless Shmem that no process owns and no reboot-free path reclaims -- 145 GB of it,
# seen here. The banks are backed by files under /dev/shm, so once every rank is gone
# those files ARE the leak, and deleting them is what a clean shutdown would have done.
#
# Use this when a rank is wedged (a stuck decode, a hung collective) and SIGTERM has not
# landed. Prefer 'stop': a graceful exit needs no cleanup at all.
cmd_kill() {
    echo "SIGKILL -- the pinned host banks will be reclaimed by hand below" >&2
    local pids; pids=$(service_pids)
    [ -n "$pids" ] && kill -KILL $pids 2>/dev/null
    for _ in $(seq 1 15); do
        [ -z "$(service_pids)" ] && break
        sleep 1
    done
    rm -f "$PIDFILE"
    if [ -n "$(service_pids)" ]; then
        echo "error: processes survived SIGKILL -- they are in uninterruptible sleep (D)," >&2
        echo "usually a stuck GPU or NFS call. Check 'ps -o stat,pid,cmd -p \$(pgrep -f ft)'." >&2
        return 1
    fi

    # Reclaim only what this model's ranks left behind, and only with no rank alive to
    # be using it -- deleting a live rank's segment corrupts a running model.
    local before after
    before=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
    find /dev/shm -maxdepth 1 \( -name 'freetoken*' -o -name 'ft_*' -o -name 'torch_*' \) \
        -user "$(id -un)" -delete 2>/dev/null
    ipcs -m | awk -v me="$(id -u)" '$3==me && $6==0 {print $2}' \
        | xargs -r -n1 ipcrm -m 2>/dev/null
    # The kernel returns the pages asynchronously, so a reading taken straight after the
    # kill still shows the whole model resident and the check below cries leak on a
    # perfectly clean shutdown. Wait for it to stop moving.
    # `$_` is a bash special variable (the previous command's last argument), not the
    # loop counter, so reading it here compared an empty string as an integer.
    after=$before
    local tick=0 now
    while [ "$tick" -lt 10 ]; do
        sleep 1
        tick=$((tick + 1))
        now=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
        # Stop once the figure stops falling, after giving it a couple of ticks to start.
        if [ "$tick" -gt 2 ] && [ "$now" -ge "$after" ]; then
            after=$now
            break
        fi
        after=$now
    done
    echo "killed; Shmem $((before/1024/1024)) GiB -> $((after/1024/1024)) GiB"

    # Only a shortfall RELATIVE to what was released is evidence of a leak. An absolute
    # threshold flags every host that legitimately uses shared memory for something else.
    if [ "$after" -gt $((before / 2)) ] && [ "$before" -gt $((32*1024*1024)) ]; then
        echo "note: less than half the shared memory came back. If nothing else on this" >&2
        echo "host uses it, some pinned banks were not reclaimed." >&2
    fi
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
    stop)   cmd_stop "${2:-}" ;;
    kill)   cmd_kill ;;
    restart) cmd_stop; cmd_start ;;
    status) cmd_status ;;
    logs)   cmd_logs ;;
    test)   cmd_test ;;
    *) echo "usage: $0 {start|stop [--force]|kill|restart|status|logs|test}"; exit 2 ;;
esac
