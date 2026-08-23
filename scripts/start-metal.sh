#!/usr/bin/env bash
#
# start-metal.sh — start FreeToken on the Apple Silicon Metal backend, with chat.
#
# One command for the local-testing loop on a Mac: it checks the environment,
# (re)creates the venv if needed, starts the Metal server, and — unless
# NO_CHAT=1 — attaches `ft shell` to it for interactive testing. On exit the
# server is stopped, so nothing is left holding ports 1919/190xx.
#
# Usage:
#   scripts/start-metal.sh [model] [options]
#
#   model            an MLX/HF repo id (mlx-community/*) or a local .gguf file
#                    (default: mlx-community/Qwen3-0.6B-4bit)
#   options          passed through to `ft serve-metal` (--backend llama,
#                    --port 1919, ...)
#
# Environment:
#   NO_CHAT=1        start the server only (no interactive shell)
#   FREETOKEN_MODEL  default model, overridden by the first positional arg
#   FREETOKEN_PORT   server port (default 1919)
#
# Examples:
#   scripts/start-metal.sh                                  # tiny Qwen3 + chat
#   scripts/start-metal.sh mlx-community/Llama-3.2-1B-Instruct-4bit
#   scripts/start-metal.sh ~/models/foo.Q4_K_M.gguf --backend llama
#   NO_CHAT=1 scripts/start-metal.sh                        # API only
#   curl -s localhost:1919/v1/models                        # in another terminal
#
set -euo pipefail

MODEL="${1:-${FREETOKEN_MODEL:-mlx-community/Qwen3-0.6B-4bit}}"
# consume the model arg so the rest go to ft serve-metal
if [[ $# -gt 0 ]]; then shift; fi
PORT="${FREETOKEN_PORT:-1919}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SERVER_PID=""

die() { echo "start-metal.sh: $*" >&2; exit 1; }

cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

command -v uv >/dev/null 2>&1 || die "uv not found — install it from https://docs.astral.sh/uv/"

# --- environment checks ------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || die "Metal backend is macOS-only (this is $(uname -s))"
[[ "$(uname -m)" == "arm64" ]] || die "native arm64 shell required — an x86_64 (Rosetta) terminal cannot install mlx wheels"
[[ -x "$PY" ]] || die "no .venv at $ROOT — run: cd $ROOT && uv venv && uv pip install -e . && uv pip install mlx-lm"

# --- pick the engine ---------------------------------------------------------
BACKEND="auto"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --backend)
      [[ $((i + 1)) -lt ${#ARGS[@]} ]] && BACKEND="${ARGS[$((i + 1))]}"
      ;;
    --backend=*)
      BACKEND="${ARGS[$i]#--backend=}"
      ;;
    --port)
      [[ $((i + 1)) -lt ${#ARGS[@]} ]] && PORT="${ARGS[$((i + 1))]}"
      ;;
    --port=*)
      PORT="${ARGS[$i]#--port=}"
      ;;
  esac
done

case "$BACKEND" in
  mlx)
    "$PY" -c "import mlx_lm" 2>/dev/null || die "mlx-lm not installed in .venv — run: uv pip install mlx-lm"
    ;;
  llama)
    command -v llama-server >/dev/null 2>&1 || die "llama-server not on PATH — run: brew install llama.cpp"
    ;;
  auto)
    if ! "$PY" -c "import mlx_lm" 2>/dev/null && ! command -v llama-server >/dev/null 2>&1; then
      die "no Metal backend installed — run: uv pip install mlx-lm, or: brew install llama.cpp"
    fi
    ;;
  *)
    die "invalid backend '$BACKEND' — expected auto, mlx, or llama"
    ;;
esac

# --- port pre-check -----------------------------------------------------------
# Fail fast and clearly instead of racing another server into "address already in
# use" (or worse: silently attaching our chat to someone else's server on that port).
if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  die "port $PORT is already serving a FreeToken server ($(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | "$PY" -c 'import json,sys
try:
    print([m["id"] for m in json.load(sys.stdin)["data"]])
except Exception:
    print("?")' 2>/dev/null)). Stop it first, or set FREETOKEN_PORT."
fi

# --- start -------------------------------------------------------------------
cd "$ROOT"
echo "▶ FreeToken Metal backend: model=$MODEL backend=$BACKEND port=$PORT"
"$ROOT/.venv/bin/ft" serve-metal --model "$MODEL" --port "$PORT" "$@" &
SERVER_PID=$!

# --- wait for readiness ------------------------------------------------------
# /health answering 200 means the API is up; /v1/models returning the served id
# means the Metal upstream finished loading (mlx_lm only serves /v1/models
# meaningfully once ready). Wait for both, so the chat attaches to a live engine.
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    SERVED="$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | "$PY" -c 'import json,sys
try:
    ids = [m["id"] for m in json.load(sys.stdin)["data"]]
except Exception:
    ids = []
print(ids[0] if len(ids) == 1 else "")' 2>/dev/null || true)"
    if [[ -n "$SERVED" ]]; then
      echo "✔ server ready at http://127.0.0.1:$PORT (serving: $SERVED)"
      break
    fi
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    die "server exited during startup (see its output above)"
  fi
  sleep 1
done
[[ -n "${SERVED:-}" ]] || die "server did not become ready within 240s"

if [[ "${NO_CHAT:-0}" != "1" ]]; then
  echo "▶ starting chat (Ctrl-D or /exit to quit; the server stops with it)"
  "$ROOT/.venv/bin/ft" shell --server "http://127.0.0.1:$PORT" || true
else
  # server-only mode: run until interrupted
  wait "$SERVER_PID"
fi
