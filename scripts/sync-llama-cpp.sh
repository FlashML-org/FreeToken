#!/usr/bin/env bash
#
# Compare the copied llama.cpp files with their pin, or merge a newer llama.cpp into them. See docs/ggml-kernels.md.
#
#   LLAMA_CPP=~/llama.cpp scripts/sync-llama-cpp.sh diff
#   LLAMA_CPP=~/llama.cpp scripts/sync-llama-cpp.sh merge <llama.cpp commit>
set -euo pipefail

LLAMA_CPP=$(realpath "${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp clone}")
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GGML=python/freetoken/kernel/csrc/ggml
GGUF=python/freetoken/third_party/gguf
# <here> <in llama.cpp>
DIRS="$GGML/ggml-cuda ggml/src/ggml-cuda
$GGML/ggml-cpu ggml/src/ggml-cpu
$GGML/src ggml/src
$GGML/include ggml/include
$GGUF gguf-py/gguf"
PIN=$(awk 'NR==1 {print $2}' "$GGML/VERSION")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

llama() { git -C "$LLAMA_CPP" "$@"; }

show_diff() {
  mkdir "$TMP/pin"
  llama archive "$PIN" ggml gguf-py | tar -x -C "$TMP/pin"
  echo "$DIRS" | while read -r here there; do
    diff -ru -x __pycache__ -x VERSION -x LICENSE "$TMP/pin/$there" "$here" | grep -v "^Only in $TMP/pin" || true
  done
}

merge() {
  local new date
  new=$(llama rev-parse --verify "$1^{commit}")
  llama cat-file -e "$PIN^{commit}"
  if ! git diff --quiet HEAD -- "$GGML" "$GGUF"; then
    echo "$GGML or $GGUF has uncommitted changes, commit or restore them first" >&2
    exit 1
  fi
  echo "$DIRS" | while read -r here there; do
    git ls-files "$here" | while read -r file; do
      path=$there/${file#"$here"/}
      # a file that llama.cpp does not have at the pin is ours
      llama show "$PIN:$path" > "$TMP/base" 2>/dev/null || continue
      if ! llama show "$new:$path" > "$TMP/theirs" 2>/dev/null; then
        echo "gone from llama.cpp: $file"
        continue
      fi
      git merge-file -L ours -L pin -L "$new" "$file" "$TMP/base" "$TMP/theirs" || echo "conflict: $file"
    done
  done
  date=$(llama show -s --format=%cs "$new")
  sed -i "1s/^llama.cpp [0-9a-f]\{40\} ([0-9-]*)/llama.cpp $new ($date)/" "$GGML/VERSION" "$GGUF/VERSION"

  echo
  echo "Added, renamed or removed in llama.cpp:"
  llama diff --name-status "$PIN" "$new" -- ggml/src/ggml-cuda ggml/src/ggml-cpu ':(glob)ggml/src/*' ggml/include gguf-py/gguf | grep -v '^M' || true
  echo
  echo "Not copied, but our code follows them, so read their diffs:"
  llama diff --stat "$PIN" "$new" -- ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml.c ggml/src/ggml-cpu/ggml-cpu.c ggml/src/CMakeLists.txt ggml/src/ggml-cpu/CMakeLists.txt
}

case "${1:-}" in
  diff) show_diff ;;
  merge) merge "${2:?usage: scripts/sync-llama-cpp.sh merge <llama.cpp commit>}" ;;
  *) echo "usage: scripts/sync-llama-cpp.sh diff | merge <llama.cpp commit>" >&2; exit 2 ;;
esac
