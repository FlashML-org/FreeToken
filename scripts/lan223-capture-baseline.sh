#!/usr/bin/env bash
# Capture a secret-free, read-only LAN-223 ROCm baseline for a FreeToken run.
#
# The script intentionally does not start a server, alter GPU clocks, install
# packages, delete cache entries, or edit system configuration.  It records
# the environment that makes a later throughput claim reproducible.

# Fail on an unset variable, an unsuccessful command in a pipeline, or a
# command error.  Individual optional probes use `|| true` so that a missing
# diagnostic utility is recorded without invalidating the whole manifest.
set -euo pipefail

# Keep the output path explicit.  A caller may pass a unique campaign folder;
# the default is suitable only for a one-off local capture.
output_dir="${1:-./artifacts/lan223-baseline-$(date -u +%Y%m%dT%H%M%SZ)}"

# Accept the GGUF path as an optional second argument.  Hashing the exact
# payload prevents a same-name but different model file from contaminating a
# benchmark comparison.
model_path="${2:-}"

# Accept the llama.cpp executable as an optional third argument.  Its checksum
# establishes the comparison binary without assuming a particular install path.
llama_binary="${3:-}"

# Resolve this script's repository root.  This makes the Git metadata capture
# independent of the shell's starting directory.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer an explicit virtual environment, then support FreeToken's LAN-223
# layout where the environment is a sibling of the source checkout, and finally
# support a conventional in-repository `.venv`.  Resolving this once prevents
# later runtime probes from silently using the system Python.
venv_python="${FREETOKEN_VENV:-}"
if [[ -z "$venv_python" && -x "$(dirname "$repo_root")/.venv/bin/python" ]]; then
  venv_python="$(dirname "$repo_root")/.venv/bin/python"
elif [[ -z "$venv_python" && -x "$repo_root/.venv/bin/python" ]]; then
  venv_python="$repo_root/.venv/bin/python"
fi

# Create the requested artifact directory without overwriting prior captures.
mkdir -p "$output_dir"

# Write one command's standard output and standard error to a named text file.
# The function returns success even for unavailable optional commands so the
# artifact shows the diagnostic failure instead of silently omitting it.
capture_command() {
  local name="$1"
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n\n'
    "$@"
  } >"$output_dir/$name" 2>&1 || true
}

# Record Git identity and local changes before inspecting the host.  Later
# benchmark reports use these files to prove which source was executed.
capture_command git-status.txt git -C "$repo_root" status --short
capture_command git-head.txt git -C "$repo_root" rev-parse HEAD
capture_command git-branch.txt git -C "$repo_root" branch --show-current
capture_command git-remotes.txt git -C "$repo_root" remote -v

# Record kernel, distribution, CPU, memory, and mount information.  These are
# read-only inputs that can affect JIT compilation and UMA decode performance.
capture_command uname.txt uname -a
capture_command os-release.txt cat /etc/os-release
capture_command cpu.txt lscpu
capture_command memory.txt free -h
capture_command mounts.txt findmnt -D

# Record the ROCm installation selected by the shell and the compiler version.
# Resolving symlinks exposes mixed ROCm installations before profiling begins.
capture_command rocm-links.txt readlink -f /opt/rocm
capture_command rocm-tree.txt find -L /opt/rocm -maxdepth 2 -type f -name 'hipcc' -o -type l -name 'libamdhip64.so*'
capture_command hipcc-version.txt /opt/rocm/bin/hipcc --version
capture_command rocprof-version.txt /opt/rocm/bin/rocprofv3 --version
capture_command rocm-packages.txt bash -lc "dpkg-query -W -f='\${Package}\t\${Version}\n' 'rocm*' 'hip*' 'rocprofiler*' 'llvm*' 2>/dev/null | sort"

# Record the active AMD device, dynamic power policy, and thermal state without
# attempting to change privileged DPM controls.
capture_command rocm-smi.txt rocm-smi --showproductname --showuniqueid --showmeminfo vram --showuse --showtemp --showclocks --showpower
capture_command dpm-policy.txt bash -lc "for f in /sys/class/drm/card*/device/power_dpm_force_performance_level /sys/class/drm/card*/device/pp_dpm_sclk; do printf '%s\n' \"### \$f\"; cat \"\$f\" 2>&1; done"

# Record only performance-relevant environment names.  Filtering avoids
# accidentally writing credentials or unrelated user environment variables.
capture_command performance-environment.txt bash -lc "env | LC_ALL=C sort | grep -E '^(ROCM|HIP|HSA|PYTORCH|TORCH|TRITON|LD_LIBRARY_PATH|PATH|FREETOKEN)=' || true"

# Ask the exact FreeToken virtual environment which HIP runtime and device it
# sees.  This detects a wheel whose embedded runtime differs from host ROCm.
if [[ -n "$venv_python" && -x "$venv_python" ]]; then
  capture_command pytorch-runtime.txt "$venv_python" -c "import json, torch; p=torch.cuda.get_device_properties(0); print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'cuda_available':torch.cuda.is_available(),'device':p.name,'gcnArchName':getattr(p,'gcnArchName',None),'total_memory':p.total_memory}, indent=2, sort_keys=True))"
  capture_command python-ldd.txt ldd "$venv_python"
else
  printf 'FreeToken virtual-environment Python not found. FREETOKEN_VENV=%s\n' "${FREETOKEN_VENV:-}" >"$output_dir/pytorch-runtime.txt"
  cp "$output_dir/pytorch-runtime.txt" "$output_dir/python-ldd.txt"
fi

# Record loaded-library resolution for the Python interpreter and rocprofv3.
# This is the primary evidence for a mixed LLVM or ROCm profiler environment.
capture_command rocprof-ldd.txt ldd /opt/rocm/bin/rocprofv3

# Hash optional comparison artifacts only when the caller supplied a readable
# path.  The explicit messages make missing input obvious in the manifest.
if [[ -n "$model_path" && -r "$model_path" ]]; then
  capture_command model-sha256.txt sha256sum "$model_path"
else
  printf 'Model path not supplied or unreadable: %s\n' "$model_path" >"$output_dir/model-sha256.txt"
fi

if [[ -n "$llama_binary" && -x "$llama_binary" ]]; then
  capture_command llama-binary-sha256.txt sha256sum "$llama_binary"
  capture_command llama-version.txt "$llama_binary" --version
else
  printf 'llama.cpp binary not supplied or not executable: %s\n' "$llama_binary" >"$output_dir/llama-binary-sha256.txt"
fi

# Create a deterministic inventory of every captured file and its SHA256.  The
# final line is a simple completion marker for automation and human review.
(cd "$output_dir" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum) >"$output_dir/SHA256SUMS"
printf 'Baseline capture complete: %s\n' "$output_dir"
