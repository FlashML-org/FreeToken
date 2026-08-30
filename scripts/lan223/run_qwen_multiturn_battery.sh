#!/usr/bin/env bash
# Run a fixed number of isolated Qwen multi-turn state-retention sessions.
#
# Each session reuses the versioned three-turn suite and writes its own immutable
# JSON artifact. The wrapper never starts, stops, or rebuilds Qwen. It requires
# a healthy, swap-free LAN-223 server before the first request and writes an
# aggregate summary only after every requested session has completed.

set -euo pipefail

readonly ARTIFACT_ROOT="${1:?usage: run_qwen_multiturn_battery.sh ARTIFACT_ROOT [SESSION_COUNT]}"
readonly SESSION_COUNT="${2:-30}"
readonly ROOT_DIR="/home/david/freetoken-amd"
readonly SOURCE_DIR="${ROOT_DIR}/source-qwen-harness-d6ee8ce"
readonly VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
readonly RUNNER="${SOURCE_DIR}/benchmarks/lan223_qwen/run_multiturn_state_suite.py"
readonly SUITE="${SOURCE_DIR}/benchmarks/lan223_qwen/multiturn_state_suite.json"
readonly MODEL="qwen3.6-35b-a3b-nvfp4-amd"
readonly EXPECTED_HOST="david-Gmktec-x2-2"

case "${SESSION_COUNT}" in
    ''|*[!0-9]*) echo "session count must be a positive integer" >&2; exit 2 ;;
esac
if (( SESSION_COUNT < 1 )); then
    echo "session count must be positive" >&2
    exit 2
fi
if [[ -e "${ARTIFACT_ROOT}" ]]; then
    echo "refusing to overwrite artifact root: ${ARTIFACT_ROOT}" >&2
    exit 2
fi
test -x "${VENV_PYTHON}"
test -f "${RUNNER}"
test -f "${SUITE}"

# Do not start an endurance-style workload from an already degraded memory
# state. Swap is a failure signal for this campaign, not a performance cache.
# Ubuntu can immediately fault one bookkeeping page after a clean swap reset,
# so permit at most 64 KiB. Any larger value is treated as real pressure.
swap_used_kb() {
    local total free
    total="$(awk '/SwapTotal/ {print $2}' /proc/meminfo)"
    free="$(awk '/SwapFree/ {print $2}' /proc/meminfo)"
    echo $((total - free))
}
assert_clean_swap() {
    local used
    used="$(swap_used_kb)"
    if (( used > 64 )); then
        echo "refusing multi-turn battery with swap in use: ${used} KiB" >&2
        exit 2
    fi
}
assert_clean_swap
curl -fsS "http://127.0.0.1:1919/health" | grep -q '"status":"ok"'
# The health request can wake a lazily swapped worker page. Check again before
# the first test request so a seemingly clean preflight cannot mask that state.
assert_clean_swap

mkdir -p "${ARTIFACT_ROOT}/sessions"
export PYTHONPATH="${SOURCE_DIR}/python"

for session in $(seq -w 1 "${SESSION_COUNT}"); do
    "${VENV_PYTHON}" "${RUNNER}" \
        --base-url "http://127.0.0.1:1919/v1" \
        --model "${MODEL}" \
        --artifact "${ARTIFACT_ROOT}/sessions/session-${session}.json" \
        --suite "${SUITE}" \
        --expected-host "${EXPECTED_HOST}" \
        --max-tokens 64 \
        >"${ARTIFACT_ROOT}/sessions/session-${session}.log" 2>&1
done

# The summary retains raw per-session files and records tail values across all
# sessions, which makes a single late response visible instead of averaged out.
"${VENV_PYTHON}" - "${ARTIFACT_ROOT}" "${SESSION_COUNT}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "sessions").glob("session-*.json"))]
ttft = [item["tail_metrics"]["max_ttft_seconds"] for item in records if item["tail_metrics"]["max_ttft_seconds"] is not None]
gaps = [item["tail_metrics"]["max_token_gap_seconds"] for item in records if item["tail_metrics"]["max_token_gap_seconds"] is not None]
def observed(values, percentile):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, int(len(values) * percentile + 0.999999999) - 1)]
summary = {
    "schema_version": 1,
    "requested_sessions": expected,
    "completed_sessions": len(records),
    "passed_sessions": sum(item["status"] == "passed" for item in records),
    "max_turn_ttft_seconds": {
        "mean": statistics.mean(ttft) if ttft else None,
        "p95": observed(ttft, 0.95),
        "p99": observed(ttft, 0.99),
        "max": max(ttft) if ttft else None,
    },
    "max_token_gap_seconds": {
        "mean": statistics.mean(gaps) if gaps else None,
        "p95": observed(gaps, 0.95),
        "p99": observed(gaps, 0.99),
        "max": max(gaps) if gaps else None,
    },
    "status": "passed" if len(records) == expected and all(item["status"] == "passed" for item in records) else "failed",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
