#!/usr/bin/env bash
#
# Build the FreeToken runtime wheel and the matching prebuilt kernel-cache wheel.
#
# Common use:
#   scripts/build-release-wheels.sh
#
# Useful environment:
#   FREETOKEN_BUILD_OUT_DIR       output directory (default: ./dist)
#   FREETOKEN_BUILD_PYTHON        python used by uv build (default: ./.venv/bin/python)
#   FREETOKEN_BUILD_NO_ISOLATION  pass --no-build-isolation when true (default: 1)
#   FREETOKEN_BUILD_CLEAN         remove old FreeToken wheels from out dir first (default: 1)
#   FREETOKEN_BUILD_KEEP_TEMP     keep setuptools/tvm build leftovers when true (default: 0)
#   FREETOKEN_BUILD_STRIP         strip debug symbols from the runtime wheel's .so (default: 1)
#   FREETOKEN_BUILD_NO_STAMP      skip the +g<sha> commit stamp -- dev builds only (default: 0)
#   FREETOKEN_BUILD_RELEASE       tagged-release mode: no stamp, and HEAD must be at the
#                                 tag v<version.py> exactly (default: 0)
#   FREETOKEN_BUILD_DEV_STAMP     stamp .dev<N> instead of +g<sha> -- PEP 440-legal without
#                                 a local segment, so TestPyPI rehearsals get a unique,
#                                 uploadable version per run (default: unset)
#   FREETOKEN_BUILD_SKIP_KERNEL_CACHE  build only the runtime wheel -- for the 2nd..Nth
#                                 interpreter of a matrix build; the kernel cache is
#                                 py3-none and only needs building once (default: 0)
#   FREETOKEN_KERNEL_CACHE_SPECS  optional comma-separated subset of kernel spec names
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

OUT_DIR="${FREETOKEN_BUILD_OUT_DIR:-$ROOT/dist}"
PYTHON_BIN="${FREETOKEN_BUILD_PYTHON:-$ROOT/.venv/bin/python}"
NO_BUILD_ISOLATION="${FREETOKEN_BUILD_NO_ISOLATION:-1}"
CLEAN="${FREETOKEN_BUILD_CLEAN:-1}"
KEEP_TEMP="${FREETOKEN_BUILD_KEEP_TEMP:-0}"
STRIP="${FREETOKEN_BUILD_STRIP:-1}"
NO_STAMP="${FREETOKEN_BUILD_NO_STAMP:-0}"
RELEASE="${FREETOKEN_BUILD_RELEASE:-0}"
DEV_STAMP="${FREETOKEN_BUILD_DEV_STAMP:-}"
SKIP_KERNEL_CACHE="${FREETOKEN_BUILD_SKIP_KERNEL_CACHE:-0}"

TRUE_VALUES=" 1 true yes on "

enabled() {
  case "$TRUE_VALUES" in
    *" $(printf '%s' "$1" | tr '[:upper:]' '[:lower:]') "*) return 0 ;;
    *) return 1 ;;
  esac
}

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[1;33m[warning]\033[0m %s\n' "$*" >&2; }

# A leaked arch override silently narrows the kernel-cache fatbin — a dev shell's
# TVM_FFI_CUDA_ARCH_LIST=9.0 once shipped an sm_90-only wheel that would driver-JIT
# (or worse) everywhere else. The build honors these vars, so shout when they're set.
warn_arch_override() {
  local var
  for var in TVM_FFI_CUDA_ARCH_LIST FREETOKEN_KERNEL_CACHE_ARCHES; do
    if [[ -n "${!var:-}" ]]; then
      warn "############################################################"
      warn "$var='${!var}' is set in this shell and OVERRIDES the"
      warn "default multi-arch list (8.0 8.6 8.9 9.0 10.0 12.0, see"
      warn "freetoken-kernel-cache/build_backend.py). The kernel-cache"
      warn "wheel will only carry SASS for the listed archs — do NOT"
      warn "release it unless the narrowing is intentional."
      warn "############################################################"
    fi
  done
}

# The `wheel` CLI (unpack/pack) — prefer the build venv's module, else run it via uv (present).
wheel_cli() {
  if "$PYTHON_BIN" -c 'import wheel.cli' >/dev/null 2>&1; then
    "$PYTHON_BIN" -m wheel "$@"
  else
    uvx --quiet wheel "$@"
  fi
}

# Strip debug symbols from the compiled extension modules in a wheel, in place. Wheels are
# already deflate-compressed zips, so downstream packagers (deb/pacman/AppImage/NSIS) cannot
# shrink them further — the two runtime .so ship UNSTRIPPED (~22 MiB) and dominate the bundle.
# We unpack → `strip --strip-unneeded` (drops .symtab/debug, KEEPS .dynsym so CPython's
# PyInit_* and any dlsym still resolve) → repack, which recomputes RECORD so the wheel stays
# valid. Runtime wheel only: the kernel-cache .so are the load-bearing prebuilt kernels and
# barely strip (~11%), so they're left untouched.
strip_wheel() {
  local whl="$1"
  enabled "$STRIP" || return 0
  command -v strip >/dev/null 2>&1 || { say "strip not found — leaving $(basename "$whl") unstripped"; return 0; }
  local before after tmp unpk n=0
  before="$(stat -c%s "$whl")"
  tmp="$(mktemp -d)"
  wheel_cli unpack --dest "$tmp/u" "$whl" >/dev/null
  while IFS= read -r -d '' so; do
    strip --strip-unneeded "$so" && n=$((n + 1))
  done < <(find "$tmp/u" -name '*.so' -print0)
  if [[ "$n" -gt 0 ]]; then
    unpk="$(find "$tmp/u" -maxdepth 1 -mindepth 1 -type d)"
    # dest-dir == the wheel's dir; `wheel pack` reproduces the identical filename → overwrites.
    wheel_cli pack --dest-dir "$(dirname "$whl")" "$unpk" >/dev/null
    after="$(stat -c%s "$whl")"
    say "stripped $n .so in $(basename "$whl"): $((before / 1024)) KiB -> $((after / 1024)) KiB"
  fi
  rm -rf "$tmp"
}

cleanup_generated() {
  enabled "$KEEP_TEMP" && return 0
  rm -rf \
    "$ROOT/build" \
    "$ROOT/python/freetoken.egg-info" \
    "$ROOT/freetoken-kernel-cache/build" \
    "$ROOT/freetoken-kernel-cache/freetoken_kernel_cache.egg-info" \
    "$ROOT/freetoken-kernel-cache/__pycache__" \
    "$ROOT/freetoken-kernel-cache/freetoken_kernel_cache/__pycache__" \
    "$ROOT/freetoken-kernel-cache/freetoken_kernel_cache/_build_meta.py" \
    "$ROOT/freetoken-kernel-cache/freetoken_kernel_cache/jit_cache"
}
# --- Release provenance stamp ------------------------------------------------
# A published wheel must be traceable to the commit it was built from, and its
# filename must CHANGE per build: the download URL is a cache key (uv caches by
# URL and does not revalidate by default), so republishing different bytes under
# an unchanged name keeps serving stale installs from warm caches forever. Stamp
# version.py with +g<sha> for the duration of the build; both wheels pick it up
# (the runtime via pyproject's dynamic attr, the kernel cache via
# build_backend.py, which merges its +cuNNN in front: 0.1.1+cu130.g<sha>).
VERSION_FILE="$ROOT/python/freetoken/version.py"
STAMPED=0
restore_version() {
  if [[ "$STAMPED" == 1 ]]; then
    git -C "$ROOT" checkout --quiet -- "python/freetoken/version.py" \
      || warn "could not restore python/freetoken/version.py -- it is still stamped; run: git checkout -- python/freetoken/version.py"
  fi
}
stamp_version() {
  if [[ -n "$DEV_STAMP" ]] && { enabled "$RELEASE" || enabled "$NO_STAMP"; }; then
    die "FREETOKEN_BUILD_DEV_STAMP cannot be combined with RELEASE or NO_STAMP modes"
  fi
  # Release mode: the runtime wheel must carry a bare PEP 440 version (PyPI rejects
  # any +local segment), so no stamp -- provenance comes from the tag instead, which
  # is verified to point at exactly this commit and this version.py. The kernel-cache
  # wheel then comes out as <version>+cu130 (no .g<sha>): build_backend.py appends
  # .g<sha> only when version.py already carries one.
  if enabled "$RELEASE"; then
    enabled "$NO_STAMP" \
      && die "FREETOKEN_BUILD_RELEASE and FREETOKEN_BUILD_NO_STAMP are mutually exclusive"
    if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
      die "working tree is not clean -- a release build must come from exactly the tagged commit."
    fi
    local version tag
    version="$(sed -nE 's/^__version__ = "([^"+]+)".*$/\1/p' "$VERSION_FILE")"
    [[ -n "$version" ]] || die "cannot read a version from $VERSION_FILE"
    tag="$(git -C "$ROOT" describe --exact-match --tags HEAD 2>/dev/null)" \
      || die "FREETOKEN_BUILD_RELEASE: HEAD is not at a tag (expected tag v$version)."
    [[ "$tag" == "v$version" ]] \
      || die "FREETOKEN_BUILD_RELEASE: HEAD tag is '$tag' but version.py says '$version' (expected tag v$version)."
    say "release build: $version (tag $tag)"
    return 0
  fi
  if enabled "$NO_STAMP"; then
    warn "FREETOKEN_BUILD_NO_STAMP is set -- building UNSTAMPED dev wheels (do not release)"
    return 0
  fi
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "not a git checkout -- cannot stamp the build commit. FREETOKEN_BUILD_NO_STAMP=1 builds unstamped."
  # A SIGKILLed build never runs the EXIT trap and leaves the stamp behind; recognize
  # exactly that leftover (version.py is the only change, and it carries a stamp) and
  # restore it instead of dying "not clean" at the operator.
  if [[ "$(git -C "$ROOT" status --porcelain)" == " M python/freetoken/version.py" ]] \
    && grep -qE '^__version__ = "[0-9][^"]*(\+g[0-9a-f]{7,}|\.dev[0-9]+)"' "$VERSION_FILE"; then
    git -C "$ROOT" checkout --quiet -- "python/freetoken/version.py"
    say "recovered a leftover version stamp from an interrupted build"
  fi
  # status --porcelain, not diff --quiet: it also catches UNTRACKED files -- setuptools
  # packages from the filesystem, so an un-added module would ship in a wheel claiming
  # to be <sha> -- and status refreshes stale stat info that makes diff --quiet report
  # phantom dirt (mtime-only touches).
  if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
    die "working tree is not clean -- a stamped wheel would lie about its commit. Commit/stash (and clean untracked files) first, or set FREETOKEN_BUILD_NO_STAMP=1 for an unstamped dev build."
  fi
  local suffix version
  if [[ -n "$DEV_STAMP" ]]; then
    [[ "$DEV_STAMP" =~ ^[0-9]+$ ]] || die "FREETOKEN_BUILD_DEV_STAMP must be a plain number (got '$DEV_STAMP')"
    suffix=".dev${DEV_STAMP}"
  else
    suffix="+g$(git -C "$ROOT" rev-parse --short=9 HEAD)"
  fi
  version="$(sed -nE 's/^__version__ = "([^"+]+)".*$/\1/p' "$VERSION_FILE")"
  [[ -n "$version" ]] || die "cannot read a version from $VERSION_FILE"
  # In-place edit, not overwrite: anything in version.py beyond the version line survives.
  sed -i -E "s/^(__version__ = \"[0-9][^\"+]*)\"/\1${suffix}\"/" "$VERSION_FILE"
  grep -qF "__version__ = \"${version}${suffix}\"" "$VERSION_FILE" \
    || die "failed to stamp $VERSION_FILE"
  STAMPED=1
  say "stamped version: $(sed -nE 's/^__version__ = "([^"]+)".*$/\1/p' "$VERSION_FILE")"
}
trap 'restore_version; cleanup_generated' EXIT

if ! command -v uv >/dev/null 2>&1; then
  die "uv not found. Install it from https://docs.astral.sh/uv/ and retry."
fi
if [[ "$PYTHON_BIN" == */* && ! -x "$PYTHON_BIN" ]]; then
  die "build python is not executable: $PYTHON_BIN"
fi
if [[ "$PYTHON_BIN" != */* ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  die "build python is not on PATH: $PYTHON_BIN"
fi

mkdir -p "$OUT_DIR"
if enabled "$CLEAN"; then
  rm -f \
    "$OUT_DIR"/freetoken-*.whl \
    "$OUT_DIR"/freetoken_kernel_cache-*.whl \
    "$OUT_DIR"/freetoken-kernel-cache-*.whl
fi

BUILD_ARGS=(build --wheel)
if enabled "$NO_BUILD_ISOLATION"; then
  BUILD_ARGS+=(--no-build-isolation)
fi
BUILD_ARGS+=(--python "$PYTHON_BIN" --out-dir "$OUT_DIR")

say "uv $(uv --version | awk '{print $2}')"
stamp_version
warn_arch_override
say "building freetoken runtime wheel"
uv "${BUILD_ARGS[@]}" .

# Select the wheel THIS interpreter just built by its cp tag -- a matrix build runs
# this script once per interpreter with CLEAN=0, so several freetoken-*.whl coexist.
cptag="$("$PYTHON_BIN" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
rt_whl="$(find "$OUT_DIR" -maxdepth 1 -name "freetoken-*-${cptag}-*.whl" -printf '%T@ %p\n' \
  | sort -n | tail -1 | cut -d' ' -f2-)"
[ -n "$rt_whl" ] || die "runtime wheel for $cptag not found in $OUT_DIR"
strip_wheel "$rt_whl"

if enabled "$SKIP_KERNEL_CACHE"; then
  say "skipping freetoken-kernel-cache wheel (FREETOKEN_BUILD_SKIP_KERNEL_CACHE)"
else
  say "building freetoken-kernel-cache wheel"
  warn_arch_override
  export FREETOKEN_KERNEL_CACHE_VERBOSE="${FREETOKEN_KERNEL_CACHE_VERBOSE:-1}"
  export FREETOKEN_KERNEL_CACHE_BUILD_DIR="${FREETOKEN_KERNEL_CACHE_BUILD_DIR:-$ROOT/build/freetoken-kernel-cache}"
  uv "${BUILD_ARGS[@]}" freetoken-kernel-cache
fi

say "wheels written to $OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -type f \
  \( -name 'freetoken-*.whl' -o -name 'freetoken_kernel_cache-*.whl' \) \
  -printf '  %p\n' | sort
