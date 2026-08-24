#!/usr/bin/env bash
# install-sm75.sh — FreeToken sm_75 (RTX 2080 Ti) install helper
#
# Targets: CUDA 12.8, torch cu128, sglang-kernel cu128, flashinfer cu12.
# Run from the repo root after `nix develop` (or any shell with CUDA 12.8
# and Python 3.10+ available).
#
# Usage:
#   bash scripts/install-sm75.sh          # full install including accel extras
#   bash scripts/install-sm75.sh --no-fi  # skip flashinfer (faster, Triton attn)

set -euo pipefail

NO_FI=0
for arg in "$@"; do
  case "$arg" in
    --no-fi) NO_FI=1 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
SGLANG_INDEX="https://docs.sglang.io/whl/cu128"
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu128/torch2.11/"

echo "══════════════════════════════════════════════════════════"
echo "  FreeToken sm_75 install — CUDA 12.8 + torch cu128"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── 1. Core package (torch resolved from cu128 index) ──────────────────────
echo "▸ Installing core package..."
uv pip install -e "." \
  --extra-index-url "$PYTORCH_INDEX"

# ── 2. sglang-kernel (cu128 build) ─────────────────────────────────────────
echo "▸ Installing sglang-kernel (cu128)..."
uv pip install "sglang-kernel==0.4.5" \
  --extra-index-url "$SGLANG_INDEX" \
  --extra-index-url "$PYTORCH_INDEX"

# ── 3. flashinfer (cu12 build, optional) ───────────────────────────────────
if [ "$NO_FI" -eq 0 ]; then
  echo "▸ Installing flashinfer (cu12)..."
  # flashinfer 0.6.x ships cu12 AOT wheels. sm_75 attention kernels may not be
  # in the AOT set (sm_80+ focused), but kernel/backend.py falls back to Triton
  # attention gracefully when flashinfer is installed but lacks sm_75 kernels.
  uv pip install "flashinfer-python[cu12]>=0.6,<0.7" \
    --extra-index-url "$FLASHINFER_INDEX" \
    --extra-index-url "$PYTORCH_INDEX" || {
    echo ""
    echo "  ⚠  flashinfer cu12 not available from the standard index."
    echo "     Triton attention fallback will be used automatically."
    echo "     Re-run with --no-fi to suppress this message."
    echo ""
  }
else
  echo "▸ Skipping flashinfer (--no-fi)"
fi

# ── 4. Build C++ extensions ─────────────────────────────────────────────────
echo "▸ Building C++ extensions (pinned_tensor, cpu_moe)..."
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
python setup.py build_ext --inplace

# ── 5. Verify ───────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Verify"
echo "══════════════════════════════════════════════════════════"
python - <<'PYEOF'
import torch, sys

# GPU info
if not torch.cuda.is_available():
    print("  ✗  No CUDA device found — check driver + LD_LIBRARY_PATH")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    cc  = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    name = torch.cuda.get_device_name(i)
    sm  = f"sm_{cc[0]}{cc[1]}"
    flag = "✓" if cc >= (7, 5) else "✗"
    print(f"  {flag}  GPU {i}: {name} ({sm}, {mem:.1f} GB)")

# FreeToken import
try:
    import freetoken
    print(f"  ✓  FreeToken {freetoken.__version__}")
except ImportError as e:
    print(f"  ✗  FreeToken import failed: {e}")
    sys.exit(1)

# arch.py
from freetoken.utils.arch import default_compute_dtype
dev = torch.device("cuda", 0)
dtype = default_compute_dtype(dev)
cc = torch.cuda.get_device_capability(0)
expected = torch.bfloat16 if cc >= (8, 0) else torch.float16
assert dtype == expected, f"default_compute_dtype returned {dtype}, expected {expected}"
print(f"  ✓  default_compute_dtype → {dtype} (correct for sm_{cc[0]}{cc[1]})")

# Kernel backends
from freetoken.kernel.backend import is_flashinfer_installed, is_sgl_kernel_installed
print(f"  {'✓' if is_sgl_kernel_installed() else '⚠'}  sgl_kernel: {'installed' if is_sgl_kernel_installed() else 'not installed (Triton fallback)'}")
print(f"  {'✓' if is_flashinfer_installed() else '⚠'}  flashinfer: {'installed' if is_flashinfer_installed() else 'not installed (Triton attn fallback)'}")

print("")
print("  All checks passed. Ready to serve.")
PYEOF

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Example: DeepSeek-V4-Flash on 4× 2080 Ti (hybrid mode)"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  ft serve --model /path/to/deepseek-v4-flash-q4_0 \\"
echo "           --tensor-parallel-size 4 \\"
echo "           --moe-backend hybrid \\"
echo "           --dtype float16 \\"
echo "           --max-model-len 32768"
echo ""
