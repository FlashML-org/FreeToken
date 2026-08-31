#!/usr/bin/env bash
# profile-rocm-decode.sh
#
# Inc 2 of .plans/rocm-perf-parity: stage-time breakdown of decode on gfx1100 for
# the Qwen3.6-35B-A3B GGUF (matching the P217 baseline conditions).
#
# Runs ft serve with FREETOKEN_TORCH_PROFILE=<warm>:<steps>:<out> (the env reaches
# the server because serve-qwen-moe.sh execs $PY from this shell), sends one
# AIME-style request, waits for the exported trace, then stops the server. Env:
#   PROFILE_OUT   chrome-trace base path (default /tmp/ft-rocm-profile/chrome.json)
#   PROFILE_WARM  decode steps skipped before the profiled window (default 40)
#   PROFILE_STEPS steps inside the profiled window (default 40)
#   FT_MODEL      model path (default the 7900 XTX box's Qwen3.6 GGUF)
#   FT_PORT       server port (default 1920)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv-rocm/bin/python}"

PROFILE_OUT="${PROFILE_OUT:-/tmp/ft-rocm-profile/chrome.json}"
PROFILE_WARM="${PROFILE_WARM:-40}"
PROFILE_STEPS="${PROFILE_STEPS:-40}"
FT_MODEL="${FT_MODEL:-/home/smk/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
FT_PORT="${FT_PORT:-1920}"
PORT="$FT_PORT"
export FT_MODEL FT_PORT

if grep -q $'\r' "${BASH_SOURCE[0]}"; then
    echo "ERROR: CRLF line endings in profile-rocm-decode.sh" >&2
    exit 1
fi
[ -x "$PY" ] || { echo "ERROR: python not found: $PY" >&2; exit 1; }
[ -f "$FT_MODEL" ] || { echo "ERROR: model not found: $FT_MODEL" >&2; exit 1; }

OUT_DIR="$(dirname "$PROFILE_OUT")"
mkdir -p "$OUT_DIR"
rm -f "$PROFILE_OUT" "${PROFILE_OUT%.json}-kernels.log"

echo "Profile env: FREETOKEN_TORCH_PROFILE=${PROFILE_WARM}:${PROFILE_STEPS}:${PROFILE_OUT}"

# serve-qwen-moe.sh refuses a double start; clear any previous instance first.
./scripts/serve-qwen-moe.sh stop >/dev/null 2>&1 || true

# serve-qwen-moe.sh refuses a double start and returns once ready (~3-4 min load).
FREETOKEN_TORCH_PROFILE="${PROFILE_WARM}:${PROFILE_STEPS}:${PROFILE_OUT}" \
    ./scripts/serve-qwen-moe.sh

# Send the profiled request (max_tokens > PROFILE_STEPS, so the engine stays
# decoding through the whole profiled window; the trace exports at its last step).
FREETOKEN_TORCH_PROFILE_MAX="${PROFILE_STEPS}" \
"${PY}" - <<'PYEOF'
import json, os, sys, urllib.request

port = os.environ["FT_PORT"]
max_tokens = (
    int(os.environ["PROFILE_WARM"]) + int(os.environ["PROFILE_STEPS"]) + 80
)
prompt = (
    "Every morning Aya goes for a 9 kilometer walk, stops at a coffee shop, then "
    "walks back home. She walks at 4 km/h, and the coffee shop detour adds 15 "
    "minutes. On a day when she walks at t km/h and the detour still costs her 2 "
    "hours total, what is t? Answer with just the number."
)
body = json.dumps({
    "model": os.path.basename(os.environ["FT_MODEL"]),
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "stream": False,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=1200) as r:
    data = json.loads(r.read())
print("request completed; completion_tokens:", data.get("usage", {}).get("completion_tokens"))
PYEOF

# The window's last step exports on exit; decode continues past the window, so poll
# briefly rather than waiting for request completion (server stays decoding).
for i in $(seq 1 10); do
    [ -f "$PROFILE_OUT" ] && break
    sleep 2
done

./scripts/serve-qwen-moe.sh stop || true

if [ -f "$PROFILE_OUT" ]; then
    echo "Trace:   $PROFILE_OUT"
    echo "Kernels: ${PROFILE_OUT%.json}-kernels.log"
else
    echo "ERROR: trace not exported to $PROFILE_OUT; check /tmp/serve_qwen_moe.log" >&2
    tail -20 /tmp/serve_qwen_moe.log >&2 || true
    exit 1
fi