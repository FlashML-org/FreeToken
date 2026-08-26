#!/usr/bin/env bash
# kill-freetoken.sh
#
# Kill every process related to FreeToken: the API server, backend workers, and the
# multiprocessing spawn/resource-tracker children, then free the listener ports.
#
# Usage:
#   ./scripts/kill-freetoken.sh [PORTS...]   # default: 1920 1921

set -u

PORTS=("${@:-1920 1921}")
SELF=$$
SELFCMDLINE="$(ps -p "$SELF" -o args= 2>/dev/null || true)"

# Kill a pattern, excluding this script's own shell.
kill_matching() {
    local pat="$1"
    for pid in $(pgrep -f "$pat" 2>/dev/null); do
        [ "$pid" = "$SELF" ] && continue
        cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        [ -n "$cmdline" ] && [ "$cmdline" = "$SELFCMDLINE" ] && continue
        kill -9 "$pid" 2>/dev/null || true
    done
}

# 1) FreeToken CLI / server / backend supervisor / backend workers.
kill_matching "freetoke[n].cli serve"
kill_matching "freetoke[n].cli"
kill_matching "freetoken"
kill_matching "multiprocessing.spawn"
kill_matching "multiprocessing.resource_tracker"
kill_matching "multiprocessing.semaphore"
kill_matching "torch.distributed.launch"

# 2) Free the listener ports (a worker may still hold one).
for p in "${PORTS[@]}"; do
    pid="$(ss -ltnp 2>/dev/null | grep ":$p" | grep -oP 'pid=\K[0-9]+' | head -1)"
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
done

sleep 1

left="$(pgrep -af "freetoken|multiprocessing.spawn|multiprocessing.resource" 2>/dev/null | grep -v "kill-freetoken.sh" | grep -v "$$" || true)"
if [ -n "$left" ]; then
    echo "WARNING: still running (forced KILL):"
    echo "$left"
    kill_matching "freetoken"
    kill_matching "multiprocessing.spawn"
    sleep 1
fi

echo "FreeToken processes killed; ports ${PORTS[*]} freed."
