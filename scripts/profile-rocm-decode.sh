#!/usr/bin/env bash
# profile-rocm-decode.sh
#
# Inc 2 of .plans/qwen-moe-speed: stage-time breakdown of decode on gfx1100 for
# the Qwen3.6-35B-A3B GGUF (matching the P217 baseline conditions).
#
# Runs ft serve with torch-profiler or low-overhead rocprofv3 marker mode, sends the
# same pinned AIME fixture/request shape as bench_decode_moe.py, then stops server.
# Env:
#   PROFILE_MODE  torch|rocprofv3 (default torch)
#   PROFILE_CAPTURE_START  launch|attach for rocprofv3 (default launch)
#   PROFILE_OUT   chrome-trace base path (default /tmp/ft-rocm-profile/chrome.json)
#   PROFILE_WARM  decode steps skipped before the profiled window (default 40)
#   PROFILE_STEPS steps inside the profiled window (default 40)
#   PROFILE_DECODE exact max_tokens for the request (default 512)
#   AIME_JSONL    local immutable AIME JSONL; required
#   AIME_PROBLEM  0-based fixture row (default 0)
#   FT_MODEL      model path (default the 7900 XTX box's Qwen3.6 GGUF)
#   FT_PORT       server port (default 1920)
#   ROCPROF_BIN   rocprofv3 path (default rocprofv3)
#   ROCPROF_TARGET_PID  attach target override; default frontend PID
#   PROFILE_ATTACH_MS  rocprofv3 attach duration (default 30000)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv-rocm/bin/python}"

PROFILE_MODE="${PROFILE_MODE:-torch}"
PROFILE_CAPTURE_START="${PROFILE_CAPTURE_START:-launch}"
PROFILE_OUT="${PROFILE_OUT:-/tmp/ft-rocm-profile/chrome.json}"
PROFILE_WARM="${PROFILE_WARM:-40}"
PROFILE_STEPS="${PROFILE_STEPS:-40}"
PROFILE_DECODE="${PROFILE_DECODE:-512}"
FT_MODEL="${FT_MODEL:-/home/smk/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
FT_PORT="${FT_PORT:-1920}"
AIME_JSONL="${AIME_JSONL:-${FREETOKEN_AIME25_JSONL:-}}"
AIME_PROBLEM="${AIME_PROBLEM:-0}"
FT_ATTN="${FT_ATTN:-triton}"
FT_MOE_CACHE="${FT_MOE_CACHE:-auto}"
FT_MEMORY_RATIO="${FT_MEMORY_RATIO:-0.9}"
FT_KV_TOKENS="${FT_KV_TOKENS:-9216}"
FT_MAX_OUTPUT="${FT_MAX_OUTPUT:-$PROFILE_DECODE}"
ROCPROF_BIN="${ROCPROF_BIN:-rocprofv3}"
PROFILE_ATTACH_MS="${PROFILE_ATTACH_MS:-30000}"
PORT="$FT_PORT"
ROCPROF_LOG="${ROCPROF_LOG:-${PROFILE_OUT%.json}-rocprofv3.log}"
export FT_MODEL FT_PORT AIME_JSONL AIME_PROBLEM FT_ATTN FT_MOE_CACHE FT_MEMORY_RATIO FT_KV_TOKENS FT_MAX_OUTPUT PROFILE_DECODE PROFILE_MODE

case "$PROFILE_MODE" in
    torch|rocprofv3) ;;
    *) echo "ERROR: PROFILE_MODE must be torch or rocprofv3" >&2; exit 1 ;;
esac
case "$PROFILE_CAPTURE_START" in
    launch|attach) ;;
    *) echo "ERROR: PROFILE_CAPTURE_START must be launch or attach" >&2; exit 1 ;;
esac
if [ "$PROFILE_MODE" = torch ] && [ "$PROFILE_CAPTURE_START" != launch ]; then
    echo "ERROR: PROFILE_CAPTURE_START only applies to PROFILE_MODE=rocprofv3" >&2
    exit 1
fi

if grep -q $'\r' "${BASH_SOURCE[0]}"; then
    echo "ERROR: CRLF line endings in profile-rocm-decode.sh" >&2
    exit 1
fi
[ -x "$PY" ] || { echo "ERROR: python not found: $PY" >&2; exit 1; }
[ -f "$FT_MODEL" ] || { echo "ERROR: model not found: $FT_MODEL" >&2; exit 1; }
[ -n "$AIME_JSONL" ] && [ -f "$AIME_JSONL" ] || {
    echo "ERROR: AIME_JSONL must point to an immutable local JSONL fixture" >&2
    exit 1
}

OUT_DIR="$(dirname "$PROFILE_OUT")"
mkdir -p "$OUT_DIR"
rm -f "$PROFILE_OUT" "${PROFILE_OUT%.json}-kernels.log"
TRACE_DIR="${PROFILE_OUT%.json}-rocprofv3"
ENV_MANIFEST="${PROFILE_OUT%.json}-env.txt"
MARKER_SIDECAR="${PROFILE_OUT%.json}-markers.json"
rm -rf "$TRACE_DIR"
{
    echo "profile_mode=$PROFILE_MODE"
    echo "profile_capture_start=$PROFILE_CAPTURE_START"
    echo "profile_out=$PROFILE_OUT"
    echo "profile_warm=$PROFILE_WARM"
    echo "profile_steps=$PROFILE_STEPS"
    echo "profile_decode=$PROFILE_DECODE"
    echo "model=$FT_MODEL"
    echo "model_sha256=$(sha256sum "$FT_MODEL" | awk '{print $1}')"
    echo "aime=$AIME_JSONL"
    echo "aime_sha256=$(sha256sum "$AIME_JSONL" | awk '{print $1}')"
    echo "git_revision=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "python=$PY"
    echo "rocprofv3=$ROCPROF_BIN"
    echo "rocprofv3_version=$($ROCPROF_BIN --version 2>&1 | head -1 || echo unavailable)"
    echo "hipcc_version=$(hipcc --version 2>/dev/null | head -1 || echo unavailable)"
    env | sort
} > "$ENV_MANIFEST"

echo "Profile mode: $PROFILE_MODE"
echo "Profile env: FREETOKEN_TORCH_PROFILE=${PROFILE_WARM}:${PROFILE_STEPS}:${PROFILE_OUT}"

# serve-qwen-moe.sh refuses a double start; clear any previous instance first.
./scripts/serve-qwen-moe.sh stop >/dev/null 2>&1 || true

# serve-qwen-moe.sh refuses a double start and returns once ready (~3-4 min load).
if [ "$PROFILE_MODE" = torch ]; then
    FREETOKEN_TORCH_PROFILE="${PROFILE_WARM}:${PROFILE_STEPS}:${PROFILE_OUT}" \
    FREETOKEN_ROCTX_MARKERS="$PROFILE_OUT" ./scripts/serve-qwen-moe.sh
else
    command -v "$ROCPROF_BIN" >/dev/null 2>&1 || {
        echo "ERROR: rocprofv3 not found: $ROCPROF_BIN" >&2
        exit 1
    }
    mkdir -p "$TRACE_DIR"
    if [ "$PROFILE_CAPTURE_START" = launch ]; then
        FREETOKEN_ROCTX_MARKERS="$PROFILE_OUT" \
        FT_ROCPROF_BIN="$ROCPROF_BIN" \
        FT_ROCPROF_TRACE_DIR="$TRACE_DIR" \
        FT_ROCPROF_LOG="$ROCPROF_LOG" \
        ./scripts/serve-qwen-moe.sh
        ROCPROF_PID="$(pgrep -f 'rocprofv3.*freetoken-rocprof' | head -1 || true)"
        echo "rocprofv3 launch profiler_pid=${ROCPROF_PID:-unknown}"
    else
        FREETOKEN_ROCTX_MARKERS="$PROFILE_OUT" ./scripts/serve-qwen-moe.sh
        TARGET_PID="${ROCPROF_TARGET_PID:-$(pgrep -f 'freetoke[n].cli serve' | head -1 || true)}"
        [ -n "$TARGET_PID" ] || {
            echo "ERROR: could not find FreeToken server PID for rocprofv3 attach" >&2
            exit 1
        }
        "$ROCPROF_BIN" --runtime-trace --marker-trace --kernel-trace \
            --memory-copy-trace --memory-allocation-trace -d "$TRACE_DIR" \
            --attach "$TARGET_PID" --attach-duration-msec "$PROFILE_ATTACH_MS" \
            > "$ROCPROF_LOG" 2>&1 &
        ROCPROF_PID=$!
        echo "rocprofv3 attach pid=$TARGET_PID profiler_pid=$ROCPROF_PID"
    fi
fi

# Send exact benchmark request shape (max_tokens > PROFILE_STEPS, so the engine
# stays decoding through the whole profiled window; trace exports at its last step).
FREETOKEN_TORCH_PROFILE_MAX="${PROFILE_STEPS}" \
PYTHONPATH="$REPO/python:$REPO/benchmarks" "${PY}" - <<'PYEOF'
import json, os, urllib.request

from bench_decode_moe import load_problem_details, resolve_sampling

port = os.environ["FT_PORT"]
max_tokens = int(os.environ["PROFILE_DECODE"]) + 1
prompt, _, _ = load_problem_details(
    os.environ["AIME_JSONL"],
    int(os.environ["AIME_PROBLEM"]),
)
sampling, _ = resolve_sampling(os.environ["FT_MODEL"], greedy=False)
body = json.dumps({
    "model": os.path.basename(os.environ["FT_MODEL"]),
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "ignore_eos": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": True},
    "stream": True,
    **sampling,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=1200) as r:
    data = {}
    for raw in r:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if payload == b"[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            data["usage"] = chunk["usage"]
print("request completed; completion_tokens:", data.get("usage", {}).get("completion_tokens"))
PYEOF

# The window's last step exports on exit; decode continues past the window, so poll
# briefly rather than waiting for request completion (server stays decoding).
for i in $(seq 1 10); do
    [ -f "$PROFILE_OUT" ] && break
    sleep 2
done

./scripts/serve-qwen-moe.sh stop || true

if [ "$PROFILE_MODE" = rocprofv3 ] && [ -n "${ROCPROF_PID:-}" ]; then
    wait "$ROCPROF_PID" || true
fi

if [ -f "$PROFILE_OUT" ]; then
    echo "Trace:   $PROFILE_OUT"
    echo "Kernels: ${PROFILE_OUT%.json}-kernels.log"
    if [ -f "$MARKER_SIDECAR" ]; then
        SUMMARY_OUT="${PROFILE_OUT%.json}-summary.json"
        PYTHONPATH="$REPO/python" "$PY" -m freetoken.utils.step_profiler \
            "$MARKER_SIDECAR" --out "$SUMMARY_OUT"
        echo "Stages:  $SUMMARY_OUT"
    fi
else
    if [ "$PROFILE_MODE" = rocprofv3 ] && find "$TRACE_DIR" -type f -print -quit 2>/dev/null | grep -q .; then
        echo "Trace:   $TRACE_DIR"
        echo "Env:     $ENV_MANIFEST"
        echo "Markers: rocprofv3 runtime/marker/kernel trace"
        if [ -f "$MARKER_SIDECAR" ]; then
            SUMMARY_OUT="${PROFILE_OUT%.json}-summary.json"
            PYTHONPATH="$REPO/python" "$PY" -m freetoken.utils.step_profiler \
                "$MARKER_SIDECAR" --out "$SUMMARY_OUT"
            echo "Stages:  $SUMMARY_OUT"
        else
            echo "Stages:  unavailable (ROCTX marker sidecar missing)"
        fi
    else
        echo "ERROR: trace not exported to $PROFILE_OUT; check /tmp/serve_qwen_moe.log" >&2
        tail -20 /tmp/serve_qwen_moe.log >&2 || true
        exit 1
    fi
fi
