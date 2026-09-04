#!/usr/bin/env bash
# Build pinned llama.cpp reference binaries outside FreeToken.
#
# Required source state: llama.cpp commit 7e4c0a968 (b10434). The script never
# changes source HEAD; it fails before configure when source identity differs.
#
# Env:
#   LLAMA_CPP_SRC    local llama.cpp checkout (required)
#   LLAMA_CPP_COMMIT exact commit (default 7e4c0a968)
#   LLAMA_BUILD_ROOT output root (default /tmp/llama-cpp-b10434)
#   LLAMA_BACKENDS   comma list: hip,vulkan (default hip,vulkan)

set -euo pipefail

EXPECTED_COMMIT="${LLAMA_CPP_COMMIT:-7e4c0a968}"
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-}"
LLAMA_BUILD_ROOT="${LLAMA_BUILD_ROOT:-/tmp/llama-cpp-b10434}"
LLAMA_BACKENDS="${LLAMA_BACKENDS:-hip,vulkan}"
MANIFEST="$LLAMA_BUILD_ROOT/provenance.txt"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$LLAMA_CPP_SRC" ] || die "set LLAMA_CPP_SRC to a local llama.cpp checkout"
[ -d "$LLAMA_CPP_SRC/.git" ] || die "LLAMA_CPP_SRC is not a git checkout: $LLAMA_CPP_SRC"
command -v cmake >/dev/null 2>&1 || die "cmake not found"

HEAD="$(git -C "$LLAMA_CPP_SRC" rev-parse HEAD 2>/dev/null)" || die "cannot read llama.cpp HEAD"
case "$HEAD" in
    "$EXPECTED_COMMIT"|"$EXPECTED_COMMIT"*) ;;
    *) die "llama.cpp HEAD=$HEAD does not match required $EXPECTED_COMMIT" ;;
esac

case ",${LLAMA_BACKENDS}," in
    *,hip,*)
        command -v hipcc >/dev/null 2>&1 || die "hipcc not found for HIP reference"
        ;;
esac
case ",${LLAMA_BACKENDS}," in
    *,hip,*|*,vulkan,*) ;;
    *) die "LLAMA_BACKENDS must contain hip and/or vulkan" ;;
esac

mkdir -p "$LLAMA_BUILD_ROOT"
{
    echo "source=$LLAMA_CPP_SRC"
    echo "source_head=$HEAD"
    echo "required_commit=$EXPECTED_COMMIT"
    echo "backends=$LLAMA_BACKENDS"
    echo "cmake=$(cmake --version | head -1)"
    echo "hipcc=$(command -v hipcc 2>/dev/null || echo unavailable)"
    echo "hipcc_version=$(hipcc --version 2>/dev/null | head -1 || echo unavailable)"
    echo "git_status=$(git -C "$LLAMA_CPP_SRC" status --short)"
} > "$MANIFEST"

build_backend() {
    local backend="$1"
    local build_dir="$LLAMA_BUILD_ROOT/$backend"
    local -a cmake_args=(
        -S "$LLAMA_CPP_SRC"
        -B "$build_dir"
        -DCMAKE_BUILD_TYPE=Release
    )
    case "$backend" in
        hip)
            cmake_args+=(
                -DGGML_HIP=ON
                -DCMAKE_HIP_ARCHITECTURES=gfx1100
            )
            ;;
        vulkan)
            cmake_args+=( -DGGML_VULKAN=ON )
            ;;
        *) die "unsupported backend: $backend" ;;
    esac
    printf 'configure:' >> "$MANIFEST"
    printf ' %q' cmake "${cmake_args[@]}"
    printf '\n' >> "$MANIFEST"
    cmake "${cmake_args[@]}"
    cmake --build "$build_dir" --target llama-server -- -j"${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
    local binary="$build_dir/bin/llama-server"
    [ -x "$binary" ] || binary="$build_dir/llama-server"
    [ -x "$binary" ] || die "llama-server missing after $backend build: $build_dir"
    printf 'binary_%s=%s\n' "$backend" "$binary" >> "$MANIFEST"
    printf 'version_%s=%s\n' "$backend" "$("$binary" --version 2>&1 | head -1)" >> "$MANIFEST"
    echo "built $backend: $binary"
}

IFS=',' read -r -a backends <<< "$LLAMA_BACKENDS"
for backend in "${backends[@]}"; do
    [ -n "$backend" ] || continue
    build_backend "$backend"
done

echo "manifest: $MANIFEST"
