#!/usr/bin/env bash
#
# FreeToken engine installer (Linux, NVIDIA CUDA) — user-facing, wheel-based.
#
# Installs the `freetoken` runtime (the `ft` CLI) and its prebuilt kernel-cache
# wheel into a managed venv, then wires it up so FreeToken Desktop can find it.
# Dependencies come from PyPI via uv, except torch and sglang-kernel whose cu130
# wheels live on dedicated indexes (see CU_INDEX_ARGS below).
#
# Typical use (once a release exists):
#   curl -fsSL https://<host>/install.sh | bash
#
# Configurable via environment:
#   FREETOKEN_WHEEL               runtime wheel — local path OR URL. Defaults to the
#                                 pinned release asset ($DEFAULT_WHEEL_URL); if that is
#                                 empty and install.sh runs from a source checkout, the
#                                 wheels are built from source via
#                                 scripts/build-release-wheels.sh.
#   FREETOKEN_KERNEL_CACHE_WHEEL  prebuilt kernel-cache wheel — local path OR URL.
#                                 Defaults to $DEFAULT_KERNEL_CACHE_WHEEL_URL; if
#                                 unset and FREETOKEN_WHEEL is local, the script
#                                 auto-detects a sibling freetoken_kernel_cache-*.whl.
#   FREETOKEN_HOME                install root (default: ~/.freetoken); venv at $FREETOKEN_HOME/venv
#   FREETOKEN_PY_VERSION          python for the venv (default: 3.12 — must match the wheel tag)
#   FREETOKEN_BIN_DIR             where to symlink `ft` (default: ~/.local/bin)
#   FREETOKEN_ENV_DIR             environment.d dir (default: ~/.config/environment.d)
#
# NOTE: common TVM FFI kernels come from the kernel-cache wheel. A working CUDA
# toolkit (nvcc) is still needed when falling back to JIT for an uncovered kernel
# variant, a mismatched cache wheel, or development builds.
set -euo pipefail

# AppImage launchers prepend bundle-private libraries to LD_LIBRARY_PATH so the
# GUI can resolve its own dependencies. That environment is inherited here by
# system children such as curl; on newer distros, system libcurl can then bind
# an older bundled libnghttp2 with the same SONAME and fail before uv bootstrap.
# Strip only entries inside APPDIR: preserve user/system paths and leave every
# non-AppImage invocation unchanged.
_sanitize_appimage_ld_library_path() {
  [ -n "${APPDIR:-}" ] || return 0
  [ -n "${LD_LIBRARY_PATH:-}" ] || return 0

  local appdir="${APPDIR%/}"
  [ -n "$appdir" ] || appdir="$APPDIR"
  local entry cleaned="" rest="${LD_LIBRARY_PATH}:"
  local kept=0
  while [ -n "$rest" ]; do
    entry="${rest%%:*}"
    rest="${rest#*:}"
    case "$entry" in
      "$appdir"|"$appdir"/*) continue ;;
    esac
    if [ "$kept" -eq 1 ]; then
      cleaned="$cleaned:$entry"
    else
      cleaned="$entry"
      kept=1
    fi
  done

  if [ "$kept" -eq 1 ]; then
    export LD_LIBRARY_PATH="$cleaned"
  else
    unset LD_LIBRARY_PATH
  fi
}
_sanitize_appimage_ld_library_path
unset -f _sanitize_appimage_ld_library_path

DEFAULT_WHEEL_URL=""   # filled in once GitHub Releases are live
DEFAULT_KERNEL_CACHE_WHEEL_URL=""   # filled in once GitHub Releases are live

FT_HOME="${FREETOKEN_HOME:-$HOME/.freetoken}"
VENV="$FT_HOME/venv"
PY_VERSION="${FREETOKEN_PY_VERSION:-3.12}"
BIN_DIR="${FREETOKEN_BIN_DIR:-$HOME/.local/bin}"
ENV_DIR="${FREETOKEN_ENV_DIR:-$HOME/.config/environment.d}"
WHEEL="${FREETOKEN_WHEEL:-$DEFAULT_WHEEL_URL}"
KERNEL_CACHE_WHEEL="${FREETOKEN_KERNEL_CACHE_WHEEL:-$DEFAULT_KERNEL_CACHE_WHEEL_URL}"

# --yes / -y (or FREETOKEN_ASSUME_YES=1): run non-interactively — in particular, bootstrap uv
# without asking. FreeToken Desktop's in-app installer passes --yes (its stdout is piped into a
# modal, so there is no terminal to prompt at).
ASSUME_YES="${FREETOKEN_ASSUME_YES:-0}"
for _arg in "$@"; do
  case "$_arg" in
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) printf 'usage: install.sh [--yes]\n'; exit 0 ;;
  esac
done

# Colorize only when stdout is a real terminal — piped output (e.g. the Desktop modal) stays
# clean text instead of showing raw ANSI escapes.
if [ -t 1 ]; then
  C_CYAN=$'\033[1;36m'; C_YELLOW=$'\033[1;33m'; C_RED=$'\033[1;31m'; C_GREEN=$'\033[1;32m'; C_RESET=$'\033[0m'
else
  C_CYAN=''; C_YELLOW=''; C_RED=''; C_GREEN=''; C_RESET=''
fi
say()  { printf '%s==>%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf '%s[error]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

infer_kernel_cache_wheel() {
  [ -z "$KERNEL_CACHE_WHEEL" ] || return 0
  case "$WHEEL" in
    http://*|https://*) return 0 ;;
  esac
  [ -e "$WHEEL" ] || return 0

  local wheel_dir candidate found count
  wheel_dir="$(cd "$(dirname "$WHEEL")" && pwd -P)"
  found=""
  count=0
  for candidate in "$wheel_dir"/freetoken_kernel_cache-*.whl "$wheel_dir"/freetoken-kernel-cache-*.whl; do
    [ -f "$candidate" ] || continue
    found="$candidate"
    count=$((count + 1))
  done

  if [ "$count" -gt 1 ]; then
    die "multiple freetoken kernel-cache wheels found next to $WHEEL — set FREETOKEN_KERNEL_CACHE_WHEEL explicitly."
  fi
  if [ "$count" -eq 1 ]; then
    KERNEL_CACHE_WHEEL="$found"
    say "auto-detected kernel-cache wheel: $KERNEL_CACHE_WHEEL"
  fi
}

build_from_repo_if_needed() {
  [ -n "$WHEEL" ] && return 0
  local script_dir builder out_dir rt kc
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
  builder="$script_dir/scripts/build-release-wheels.sh"
  [ -f "$builder" ] || return 0

  out_dir="$script_dir/dist"
  say "no wheel given — building from source in $script_dir (compiles CUDA kernels; may take a while) ..."
  # Unstamped by default: this path serves tarball checkouts (no .git) and dev trees
  # (often dirty), where the release stamp's clean-git requirement would turn a working
  # install into a die -- and a wheel installed from a local path has no URL-cache
  # staleness to defend against. FREETOKEN_BUILD_NO_STAMP=0 forces a stamp anyway.
  FREETOKEN_BUILD_OUT_DIR="$out_dir" FREETOKEN_BUILD_NO_STAMP="${FREETOKEN_BUILD_NO_STAMP:-1}" bash "$builder"

  rt="$(ls -t "$out_dir"/freetoken-*.whl 2>/dev/null | head -1)" || true
  [ -n "$rt" ] || die "build finished but no freetoken-*.whl found in $out_dir"
  WHEEL="$rt"
  say "built runtime wheel: $WHEEL"
  if [ -z "$KERNEL_CACHE_WHEEL" ]; then
    kc="$(ls -t "$out_dir"/freetoken_kernel_cache-*.whl 2>/dev/null | head -1)" || true
    if [ -n "$kc" ]; then
      KERNEL_CACHE_WHEEL="$kc"
      say "built kernel-cache wheel: $KERNEL_CACHE_WHEEL"
    fi
  fi
}

# The FreeToken Desktop engine bundle ships the wheels in ./dist next to this script.
find_bundled_wheel() {
  [ -z "$WHEEL" ] || return 0
  local script_dir dist rt
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
  dist="$script_dir/dist"
  [ -d "$dist" ] || return 0
  rt="$(ls -t "$dist"/freetoken-*.whl 2>/dev/null | grep -Ev 'freetoken[_-]kernel[_-]cache-' | head -1)" || true
  [ -n "$rt" ] && { WHEEL="$rt"; say "found bundled runtime wheel: $WHEEL"; }
}

# --- 1. uv (bootstrap into the install tree when absent) -------------------
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  # Installing uv is a system side-effect — require consent unless --yes.
  if [ "$ASSUME_YES" != 1 ]; then
    if [ -e /dev/tty ]; then
      printf '%suv is not installed. Install it into %s (astral.sh)? [y/N] %s' "$C_YELLOW" "$BIN_DIR" "$C_RESET" >/dev/tty
      # -t 60: if /dev/tty exists but nobody answers (odd CI/automation), decline instead of
      # blocking forever. Interactive users have plenty of time to type y.
      read -r -t 60 _ans </dev/tty || _ans=''
      case "$_ans" in [yY]*) ;; *) die "declined — install uv from https://docs.astral.sh/uv/ and re-run, or pass --yes." ;; esac
    else
      die "uv not found and no terminal to confirm — pass --yes, or install uv from https://docs.astral.sh/uv/ first."
    fi
  fi
  command -v curl >/dev/null 2>&1 || die "need curl to bootstrap uv."
  say "bootstrapping uv into $BIN_DIR ..."
  mkdir -p "$BIN_DIR"
  UV_UNMANAGED_INSTALL="$BIN_DIR" curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$BIN_DIR/uv"
  [ -x "$UV" ] || die "uv bootstrap failed. Install uv from https://docs.astral.sh/uv/ and re-run."
fi
say "uv $("$UV" --version | awk '{print $2}')"

# Resolve the runtime wheel: explicit env → ./dist bundle → build from a source checkout.
find_bundled_wheel
build_from_repo_if_needed

[ -n "$WHEEL" ] || die "no runtime wheel to install — set FREETOKEN_WHEEL to a local path or URL, or run install.sh from a source checkout to build one."
infer_kernel_cache_wheel
[ -n "$KERNEL_CACHE_WHEEL" ] || die "no kernel-cache wheel to install — set FREETOKEN_KERNEL_CACHE_WHEEL to a local path or URL."

# --- 2. NVIDIA driver + CUDA toolkit -----------------------------------------
# CUDA major the kernel-cache wheel was built for, from its +cuNNN tag
# (cu130 -> 13; the stamped form +cu130.g<sha> parses the same). Empty when absent.
wheel_cuda_major() {
  case "$KERNEL_CACHE_WHEEL" in
    *+cu[0-9]*)
      local tag="${KERNEL_CACHE_WHEEL##*+cu}"
      tag="${tag%%[!0-9]*}"
      printf '%s' "${tag%?}"
      ;;
  esac
}

if command -v nvidia-smi >/dev/null 2>&1; then
  say "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
  warn "nvidia-smi not found — the runtime needs an NVIDIA GPU + CUDA-capable driver."
fi
if command -v nvcc >/dev/null 2>&1; then
  NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
  say "nvcc $NVCC_RELEASE (${CUDA_HOME:-toolkit found})"
  WANT_CUDA_MAJOR="$(wheel_cuda_major)"
  if [ -n "$WANT_CUDA_MAJOR" ] && [ "${NVCC_RELEASE%%.*}" != "$WANT_CUDA_MAJOR" ]; then
    warn "nvcc $NVCC_RELEASE does not match this install's CUDA $WANT_CUDA_MAJOR.x stack:"
    warn "JIT-built kernels would link libcudart.so.${NVCC_RELEASE%%.*}, which this venv"
    warn "cannot load. Install a CUDA $WANT_CUDA_MAJOR.x toolkit if you need the JIT fallback."
  fi
else
  warn "nvcc (CUDA toolkit) not found. Prebuilt kernel-cache coverage should handle"
  warn "common kernels, but uncovered variants or cache mismatches will fail without"
  warn "nvcc on PATH / CUDA_HOME set."
fi

# --- 3. Install the wheel into a managed venv ------------------------------
say "creating venv at $VENV (python $PY_VERSION) ..."
mkdir -p "$FT_HOME"
# Always a fresh venv (no reuse): --clear replaces any existing one non-interactively, so a
# re-install can't inherit a stale/mismatched torch (e.g. an old cu128 venv after a cu130 bump).
"$UV" venv "$VENV" --python "$PY_VERSION" --clear

# PyPI's torch 2.11.0 and sglang-kernel 0.4.5 are the same cu130 builds these indexes
# serve; the explicit indexes pin provenance to the cu130 channels. `unsafe-best-match`
# is needed because the pytorch index also mirrors stale copies of common deps (e.g.
# packaging<=24.1) that would shadow PyPI under uv's first-index strategy; all indexes
# here are trusted. [tool.uv.sources] does not survive into a built wheel, so the
# indexes it names must be repeated below.
# flashinfer JIT-compiles its kernels on first use (e.g. sampling softmax), which needs nvcc --
# breaking driver-only on a box with no CUDA toolkit. flashinfer-cubin + flashinfer-jit-cache ship
# those kernels PREBUILT (multi-arch), so nothing compiles at runtime. They live on flashinfer's
# own index (cubin arch-agnostic; jit-cache per-cuNNN). Large (~2 GiB) but downloaded once.
INSTALL_WHEELS=(
  "${WHEEL}[accel]"
  flashinfer-cubin
  flashinfer-jit-cache
  "$KERNEL_CACHE_WHEEL"
)
CU_INDEX_ARGS=(
  --index-strategy unsafe-best-match
  --extra-index-url https://download.pytorch.org/whl/cu130
  --extra-index-url https://docs.sglang.io/whl/cu130
  --extra-index-url https://flashinfer.ai/whl
  --extra-index-url https://flashinfer.ai/whl/cu130
)
say "installing $WHEEL + accel (flashinfer prebuilt + sglang-kernel) + $KERNEL_CACHE_WHEEL ..."
# --refresh-package: the engine wheels have been republished under unchanged URLs (the
# rolling `beta` release), and uv's URL-keyed cache does not revalidate by default -- a
# box that cached a wheel before a republish silently reinstalls the stale copy forever
# (the venv is fresh each time, the cache is not). Force revalidation of just our two
# packages; every other dependency keeps hitting the cache.
"$UV" pip install --python "$VENV" \
  --refresh-package freetoken --refresh-package freetoken-kernel-cache \
  "${CU_INDEX_ARGS[@]}" "${INSTALL_WHEELS[@]}"

FT_BIN="$VENV/bin/ft"
[ -x "$FT_BIN" ] || die "install finished but $FT_BIN is missing."

# --- 4. Wire up for PATH + FreeToken Desktop -------------------------------
mkdir -p "$BIN_DIR"
ln -sf "$FT_BIN" "$BIN_DIR/ft"
say "symlinked $BIN_DIR/ft -> $FT_BIN"

mkdir -p "$ENV_DIR"
printf 'FREETOKEN_FT_BIN=%s\n' "$FT_BIN" > "$ENV_DIR/50-freetoken.conf"
say "wrote $ENV_DIR/50-freetoken.conf (FREETOKEN_FT_BIN) — GUI picks it up after next login"

# --- 5. Self-check ---------------------------------------------------------
if "$FT_BIN" --help >/dev/null 2>&1; then
  say "self-check: \`ft --help\` OK"
else
  warn "self-check: \`ft --help\` returned non-zero — inspect with: $FT_BIN --help"
fi

cat <<EOF

${C_GREEN}FreeToken engine installed.${C_RESET}

  ft binary        $FT_BIN
  on PATH as       $BIN_DIR/ft   (ensure $BIN_DIR is on PATH)
  Desktop env      FREETOKEN_FT_BIN via environment.d (re-login to apply)

Run in this shell without re-login:
  export FREETOKEN_FT_BIN="$FT_BIN"
  ft serve --model <path> --port 1919

EOF
