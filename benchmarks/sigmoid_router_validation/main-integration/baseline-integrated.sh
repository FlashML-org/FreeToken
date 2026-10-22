#!/bin/bash
set -e
source /home/freetoken/freetoken-work/env.sh
python - <<'PY'
from pathlib import Path
import shutil
src=Path('/home/freetoken/freetoken-work/FreeToken-integrated')
dst=Path('/home/freetoken/freetoken-work/FreeToken-main-baseline')
for p in (src/'python').rglob('*.so'):
    q=dst/p.relative_to(src)
    q.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,q)
PY
cd /home/freetoken/freetoken-work/FreeToken-main-baseline
out=/home/freetoken/freetoken-work/results/main-integrated
PYTHONPATH=python python -m pytest tests/kernels/test_dsv4_router_select.py tests/models/test_glm5_next_config.py tests/models/test_minimax_m3.py tests/checkpoint/test_ftw_weights.py tests/engine/test_cache_budget.py tests/engine/test_moe_cpu_layers.py tests/scheduler/test_scheduler_status.py -q -ra --tb=short > "$out/baseline-regression-tests.log" 2>&1
printf '0\n' > "$out/baseline.exit"
tail -3 "$out/baseline-regression-tests.log"