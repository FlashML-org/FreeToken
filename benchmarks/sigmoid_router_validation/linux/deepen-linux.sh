#!/bin/bash
set -u
source /home/freetoken/freetoken-work/env.sh
cd /home/freetoken/freetoken-work/FreeToken
out=/mnt/d/AI/FreeToken-validation/wsl-setup
PYTHONPATH=python python benchmarks/sigmoid_router_validation/deepen_validation.py --router-cache /mnt/d/AI/FreeToken-validation/GLM-4.5-Air-router --prefix-cache /mnt/d/AI/FreeToken-validation/GLM-4.5-Air-prefix --output "$out/deepening-verified" > "$out/deepening-verified.log" 2>&1
result=$?
printf '%s\n' "$result" > "$out/deepening-verified.exit"
tail -8 "$out/deepening-verified.log"
exit "$result"