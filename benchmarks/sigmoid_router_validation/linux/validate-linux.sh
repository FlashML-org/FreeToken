#!/bin/bash
set -u
out=/mnt/d/AI/FreeToken-validation/wsl-setup
work=/home/freetoken/freetoken-work
source "$work/env.sh"
run_step() {
    local name=$1
    shift
    date -u +%FT%TZ > "$out/$name.started"
    "$@" > "$out/$name.log" 2>&1
    local result=$?
    printf '%s\n' "$result" > "$out/$name.exit"
    tail -15 "$out/$name.log"
    return "$result"
}
run_step dependencies env UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=3 uv pip install --python "$work/.venv/bin/python" -r "$work/requirements-linux-resolved.txt" || exit $?
run_step cuda-smoke python "$out/smoke.py" || exit $?
run_step nvcc nvcc --version || exit $?
cd "$work/FreeToken"
run_step build uv pip install --python "$work/.venv/bin/python" -e . --no-build-isolation --no-deps || exit $?
uv pip freeze --python "$work/.venv/bin/python" > "$out/environment-linux.txt"
export PYTHONPATH=python
run_step targeted-tests python -m pytest tests/kernels/test_sigmoid_router.py tests/models/test_sigmoid_routing.py -q -ra --tb=short
local_result=$?
run_step regression-tests python -m pytest tests/kernels/test_dsv4_router_select.py tests/models/test_glm5_next_config.py tests/models/test_minimax_m3.py tests/checkpoint/test_ftw_weights.py -q -ra --tb=short
regression_result=$?
printf 'targeted=%s regression=%s\n' "$local_result" "$regression_result" > "$out/validation-summary.txt"