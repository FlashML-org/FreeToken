#!/usr/bin/env bash
# Stop only the GMKtec EVO-X2 loopback NVFP4 recovery server as one process group.
#
# The FreeToken frontend creates scheduler and tokenizer child processes. A
# parent-only signal can leave one of those children holding GPU memory or the
# internal distributed port. This helper verifies the listener's exact model
# and port before signalling its dedicated session, so it cannot target an
# unrelated service on the shared machine.

set -euo pipefail

# Keep the protected service identity explicit rather than inferring it from a
# PID file that might be stale after a reboot or failed experimental run.
readonly PORT="1919"
readonly MODEL_PATH="/home/david/freetoken-amd/models/Qwen3.6-35B-A3B-NVFP4"

# Resolve the actual TCP listener because it is the authoritative owner of the
# endpoint that this helper is permitted to stop.
listener_pid() {
    ss -ltnp "( sport = :${PORT} )" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

# Require the intended FreeToken command and model path before a process group
# signal. This prevents an accidental port reuse from becoming a destructive
# signal to another local application.
is_recovery_server() {
    local pid="$1" command
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    command="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${command}" == *"freetoken.cli serve"* ]] || return 1
    [[ "${command}" == *"${MODEL_PATH}"* ]] || return 1
    [[ "${command}" == *"--port ${PORT}"* ]]
}

# A successful start uses setsid, making the server PID its own process-group
# ID. Refuse legacy non-isolated launches rather than guessing which children
# belong to the server. The caller can keep the service running and inspect it.
stop_server() {
    local pid="$1" pgid
    is_recovery_server "${pid}" || {
        echo "refusing to stop an unrecognized port ${PORT} listener: ${pid}" >&2
        return 1
    }
    pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
    [[ "${pgid}" == "${pid}" ]] || {
        echo "refusing legacy non-isolated recovery server ${pid}; restart it with the current launcher first" >&2
        return 1
    }
    kill -TERM -- "-${pgid}" || true
    for _ in $(seq 1 90); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
    done
    # Escalate only the already-verified dedicated process group if a stuck HIP
    # operation prevented graceful Python shutdown.
    kill -0 "${pid}" 2>/dev/null && kill -KILL -- "-${pgid}" || true
}

pid="$(listener_pid)"
[[ -n "${pid}" ]] || { echo "no recovery server is listening on port ${PORT}"; exit 0; }
stop_server "${pid}"

# Confirm the port was released before the caller starts an isolated candidate.
for _ in $(seq 1 15); do
    [[ -z "$(listener_pid)" ]] && exit 0
    sleep 1
done
echo "recovery server listener remained on port ${PORT}" >&2
exit 1
