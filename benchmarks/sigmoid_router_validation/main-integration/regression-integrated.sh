#!/bin/bash
source /home/freetoken/freetoken-work/env.sh
out=/home/freetoken/freetoken-work/results/main-integrated
for tree in FreeToken-integrated FreeToken-main-baseline; do
 cd /home/freetoken/freetoken-work/$tree
 PYTHONPATH=python python -m pytest tests/kernels/test_dsv4_router_select.py tests/models/test_glm5_next_config.py tests/models/test_minimax_m3.py tests/checkpoint/test_ftw_weights.py tests/engine/test_cache_budget.py tests/engine/test_moe_cpu_layers.py tests/scheduler/test_scheduler_status.py -q -ra --tb=short > "$out/$tree-regression.log" 2>&1
 printf '%s\n' "$?" > "$out/$tree-regression.exit"
 tail -4 "$out/$tree-regression.log"
done
uv pip freeze --python /home/freetoken/freetoken-work/.venv/bin/python > "$out/environment.txt"