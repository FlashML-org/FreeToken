#!/usr/bin/env bash
# Convenience launcher for FreeToken in this repo's venv.
# Sets CUDA_HOME to the pip-installed CUDA 13 toolkit (nvcc/cicc/ptxas) that the
# runtime CUDA-JIT kernels need, plus PATH/LD_LIBRARY_PATH, then runs `ft`.
#
# Usage:  ./run-freetoken.sh <ft-args...>
#   e.g.  ./run-freetoken.sh serve --model ~/models/gpt-oss-20b
#         ./run-freetoken.sh shell
#         ./run-freetoken.sh bench bw
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
CU="$VENV/lib/python3.13/site-packages/nvidia/cu13"
export CUDA_HOME="$CU"
export PATH="$VENV/bin:$CU/bin:$PATH"
export LD_LIBRARY_PATH="$CU/lib:${LD_LIBRARY_PATH:-}"
exec "$VENV/bin/ft" "$@"
