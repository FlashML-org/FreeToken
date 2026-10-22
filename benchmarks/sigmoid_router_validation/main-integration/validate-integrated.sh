#!/bin/bash
set -e
source /home/freetoken/freetoken-work/env.sh
cd /home/freetoken/freetoken-work/FreeToken-integrated
out=/home/freetoken/freetoken-work/results/main-integrated
export PYTHONPATH=python
python benchmarks/sigmoid_router_validation/deepen_validation.py --router-cache /home/freetoken/freetoken-work/GLM-4.5-Air-router --prefix-cache /home/freetoken/freetoken-work/prefix-activations --output "$out" > "$out/ablation.log" 2>&1
python - <<'PY'
import json
from pathlib import Path
out=Path('/home/freetoken/freetoken-work/results/main-integrated')
cases=json.loads((out/'tie-audit.json').read_text())['mismatches']
assert len(cases)==4
p=Path('tests/kernels/data/sigmoid_router_ties.json')
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(cases,indent=2))
PY
python -m pytest tests/kernels/test_sigmoid_router.py tests/models/test_sigmoid_routing.py -q -ra --tb=short > "$out/targeted-tests.log" 2>&1
python -m pytest tests/kernels/test_dsv4_router_select.py tests/models/test_glm5_next_config.py tests/models/test_minimax_m3.py tests/checkpoint/test_ftw_weights.py tests/engine/test_cache_budget.py tests/engine/test_moe_cpu_layers.py tests/scheduler/test_scheduler_status.py -q -ra --tb=short > "$out/regression-tests.log" 2>&1
printf '0\n' > "$out/validation.exit"
tail -3 "$out/targeted-tests.log"
tail -3 "$out/regression-tests.log"