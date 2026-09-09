#!/usr/bin/env bash
#
# Build the release wheels inside the pytorch manylinux_2_28 CUDA container, so the
# shipped .so get a glibc 2.28 floor (the same floor as torch's own cu130 wheels)
# instead of inheriting whatever glibc the build host runs.
#
# Host usage (CI runner or a dev machine with docker):
#   scripts/ci/manylinux-build.sh
#
# The script re-execs itself inside the container; everything below the
# FT_IN_CONTAINER guard runs in the container as root.
#
# Environment (host side):
#   FT_BUILDER_IMAGE   builder image (default: pytorch/manylinux2_28-builder:cuda13.0)
#   FT_CI_CACHE_DIR    persistent cache dir on the host, holds the uv binary and
#                      uv's package cache across builds (default: ~/.cache/freetoken-ci)
#   FT_OUT_DIR         host dir that receives the wheels (default: <repo>/dist)
#   FT_PYTHON_MATRIX   space-separated cp tags to build the runtime wheel for
#                      (default: cp312 -- the nightly/Desktop channel is cp312-only;
#                      the release lane passes "cp310 cp311 cp312 cp313")
#   FT_MANYLINUX_RETAG retag runtime wheels linux_x86_64 -> detected manylinux (default: 0).
#                      Release/PyPI lane only: shipped Desktops resolve the beta
#                      release's assets (copied from nightly) by name and expect linux_x86_64.
#   FREETOKEN_BUILD_NO_STAMP / _RELEASE / _DEV_STAMP / _STRIP and
#   FREETOKEN_KERNEL_CACHE_* are forwarded into the container. Other
#   FREETOKEN_BUILD_* vars are NOT: _CLEAN is set by this script per matrix
#   iteration, and the rest (_KEEP_TEMP, _NO_ISOLATION, _OUT_DIR, ...) only
#   make sense when driving build-release-wheels.sh directly.
set -euo pipefail

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

if [[ -z "${FT_IN_CONTAINER:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
  IMAGE="${FT_BUILDER_IMAGE:-pytorch/manylinux2_28-builder:cuda13.0}"
  CACHE_DIR="${FT_CI_CACHE_DIR:-$HOME/.cache/freetoken-ci}"
  OUT_DIR="${FT_OUT_DIR:-$ROOT/dist}"
  mkdir -p "$CACHE_DIR" "$OUT_DIR"

  say "building in $IMAGE"
  exec docker run --rm \
    -e FT_IN_CONTAINER=1 \
    -e FT_HOST_UID="$(id -u)" \
    -e FT_HOST_GID="$(id -g)" \
    -e FREETOKEN_BUILD_NO_STAMP="${FREETOKEN_BUILD_NO_STAMP:-}" \
    -e FREETOKEN_BUILD_RELEASE="${FREETOKEN_BUILD_RELEASE:-}" \
    -e FREETOKEN_BUILD_DEV_STAMP="${FREETOKEN_BUILD_DEV_STAMP:-}" \
    -e FREETOKEN_BUILD_STRIP="${FREETOKEN_BUILD_STRIP:-}" \
    -e FREETOKEN_KERNEL_CACHE_SPECS="${FREETOKEN_KERNEL_CACHE_SPECS:-}" \
    -e FREETOKEN_KERNEL_CACHE_VERBOSE="${FREETOKEN_KERNEL_CACHE_VERBOSE:-}" \
    -e FT_PYTHON_MATRIX="${FT_PYTHON_MATRIX:-}" \
    -e FT_MANYLINUX_RETAG="${FT_MANYLINUX_RETAG:-}" \
    -v "$ROOT:/workspace" \
    -v "$CACHE_DIR:/ci-cache" \
    -v "$OUT_DIR:/ci-out" \
    -w /workspace \
    "$IMAGE" bash scripts/ci/manylinux-build.sh
fi

# ---------------- inside the container (root) ----------------

# The mounted repo belongs to the host user; git refuses to touch it from root
# without this (and the version stamp both reads and restores via git).
git config --global --add safe.directory /workspace

# Wheels and the stamp-restored version.py are written as root into host-owned
# dirs; hand them back to the host user even when the build dies mid-way.
restore_ownership() {
  chown -R "$FT_HOST_UID:$FT_HOST_GID" /ci-out 2>/dev/null || true
  chown "$FT_HOST_UID:$FT_HOST_GID" \
    /workspace/python/freetoken/version.py /workspace/.git/index 2>/dev/null || true
}
trap restore_ownership EXIT

export PATH="/ci-cache/bin:$PATH"
if [[ ! -x /ci-cache/bin/uv ]]; then
  say "installing uv into the persistent cache"
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/ci-cache/bin sh -s -- --quiet
fi
export UV_CACHE_DIR=/ci-cache/uv

MATRIX="${FT_PYTHON_MATRIX:-cp312}"
RETAG="${FT_MANYLINUX_RETAG:-0}"

export FREETOKEN_BUILD_OUT_DIR=/ci-out
# One clean here instead of per-invocation: with several interpreters, each
# build-release-wheels.sh run would otherwise wipe the previous ABI's wheel.
rm -f /ci-out/freetoken-*.whl /ci-out/freetoken_kernel_cache-*.whl /ci-out/freetoken-kernel-cache-*.whl
export FREETOKEN_BUILD_CLEAN=0

# Build venvs are throwaway (recreated per build from the warm uv cache) so stale
# build deps can never linger; only the cache dir persists across builds.
first=1
for cptag in $MATRIX; do
  PYBIN="/opt/python/${cptag}-${cptag}/bin/python"
  [[ -x "$PYBIN" ]] || { echo "no such interpreter in the builder image: $PYBIN" >&2; exit 1; }
  VENV="/tmp/build-venv-$cptag"
  say "creating build venv ($cptag)"
  uv venv --quiet --python "$PYBIN" "$VENV"
  # Provenance pin: PyPI's torch 2.11.0 is itself the cu130 build, but this index
  # serves ONLY cu130 wheels, so the resolve can never pick a different-CUDA torch
  # and tag the kernel-cache wheel wrong; everything else is plain PyPI.
  uv pip install --quiet --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu130 "torch>=2.11,<2.12"
  uv pip install --quiet --python "$VENV/bin/python" \
    "setuptools>=77" wheel ninja "apache-tvm-ffi==0.1.13.post3"

  export FREETOKEN_BUILD_PYTHON="$VENV/bin/python"
  # The kernel-cache wheel is py3-none: build it once, with the first interpreter.
  export FREETOKEN_BUILD_SKIP_KERNEL_CACHE="$((1 - first))"
  first=0
  # No exec: the ownership trap above must still fire after the build returns.
  bash scripts/build-release-wheels.sh
done

# Release/PyPI lane only (see the header note on FT_MANYLINUX_RETAG). The glob
# leaves the kernel-cache wheel alone: freetoken_* does not match freetoken-*.
case " 1 true yes on " in *" $(printf '%s' "$RETAG" | tr '[:upper:]' '[:lower:]') "*)
  say "retagging runtime wheels to their detected manylinux policy"
  uv pip install --quiet --python "$VENV/bin/python" "auditwheel==6.6.0"
  found=0
  for whl in /ci-out/freetoken-*linux_x86_64.whl; do
    [[ -e "$whl" ]] || continue
    "$VENV/bin/python" scripts/ci/retag-manylinux.py "$whl"
    found=1
  done
  [[ "$found" == 1 ]] || { echo "FT_MANYLINUX_RETAG set but no linux_x86_64 runtime wheels in /ci-out" >&2; exit 1; }
  ;;
esac
