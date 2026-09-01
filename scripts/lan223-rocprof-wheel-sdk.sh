#!/usr/bin/env bash
# Launch rocprofv3 against the ROCm SDK bundled with the active PyTorch wheel.
#
# On LAN-223, FreeToken's PyTorch ROCm wheel loads its own LLVM and
# rocprofiler-sdk libraries.  Launching rocprofv3 against /opt/rocm injects a
# second copy of LLVM, which aborts during `import torch` because LLVM command
# line options are registered twice.  This wrapper selects the wheel's matching
# SDK so the profiler and application load one library identity.

# Stop on programming errors.  The wrapped application exit status is preserved
# so callers can distinguish profiler setup failures from application failures.
set -euo pipefail

# Require the application separator used by rocprofv3.  Keeping profiler flags
# before `--` makes arbitrary HIP applications usable without hard-coding a
# FreeToken server command in this helper.
if [[ "$#" -lt 1 ]]; then
  printf 'Usage: %s [rocprofv3 options] -- application [arguments...]\n' "$0" >&2
  exit 64
fi

# Prefer an explicit virtual environment and otherwise use the LAN-223 layout
# where `.venv` is adjacent to the source checkout that contains this script.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="${FREETOKEN_VENV_ROOT:-$(dirname "$repo_root")/.venv}"

# Locate the wheel-owned ROCm SDK rather than assuming a Python minor version.
# The glob is validated to prevent a shell literal from being passed to rocprof.
sdk_candidates=("$venv_root"/lib/python*/site-packages/_rocm_sdk_core)
if [[ ! -d "${sdk_candidates[0]}" ]]; then
  printf 'Cannot find PyTorch wheel ROCm SDK under %s. Set FREETOKEN_VENV_ROOT.\n' "$venv_root" >&2
  exit 66
fi
sdk_root="${sdk_candidates[0]}"

# Verify the two libraries needed by rocprofv3 exist in the selected SDK.  This
# catches an incomplete or non-ROCm PyTorch wheel before it starts an app.
if [[ ! -r "$sdk_root/lib/librocprofiler-sdk.so.1" || ! -r "$sdk_root/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so.1" ]]; then
  printf 'The selected SDK lacks rocprofiler-sdk 1.3 components: %s\n' "$sdk_root" >&2
  exit 66
fi

# Use the host's rocprofv3 front end but direct every profiler library lookup to
# the exact SDK already used by PyTorch.  Do not set LD_PRELOAD here: rocprofv3
# owns its preload order and forwards the selected tool to the child process.
exec /opt/rocm/bin/rocprofv3 --rocm-root "$sdk_root" "$@"
