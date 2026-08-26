#!/usr/bin/env bash
# mirror-hf-configs.sh
#
# Mirror a model's official Hugging Face config files next to a local GGUF so
# FreeToken uses them instead of whatever the GGUF packager embedded:
#
#   ./mirror-hf-configs.sh <hf-repo-id> [gguf-file]
#
#   ./mirror-hf-configs.sh Qwen/Qwen3.6-35B-A3B
#   ./mirror-hf-configs.sh Qwen/Qwen3.6-35B-A3B /media/smk/Shared/Models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
#
# With no gguf argument, defaults to $FT_MODEL, else the single *.gguf in
# /media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Models or /media/smk/Shared/Models
# matching the repo's basename, else errors.
#
# Files fetched (when they exist upstream):
#   chat_template.jinja     — read by FreeToken's GGUF loader (overrides embedded)
#   generation_config.json  — default sampling params
#   tokenizer_config.json   — reference only (FreeToken builds its tokenizer from GGUF)
#
# To revert, delete the mirrored files — the GGUF metadata is never modified.

set -euo pipefail

REPO_ID="${1:?usage: mirror-hf-configs.sh <hf-repo-id> [gguf-file]}"
shift || true

if [ "$#" -ge 1 ]; then
    GGUF="$1"
else
    GGUF="${FT_MODEL:-}"
    if [ -z "$GGUF" ]; then
        base="$(basename "${REPO_ID##*:}")"
        cand="$(find /media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Models /media/smk/Shared/Models -maxdepth 1 -iname "*${base%%-*}*.gguf" 2>/dev/null | head -1)"
        [ -n "$cand" ] || { echo "ERROR: no gguf found; pass one explicitly" >&2; exit 1; }
        GGUF="$cand"
    fi
fi
[ -f "$GGUF" ] || { echo "ERROR: not a file: $GGUF" >&2; exit 1; }

DIR="$(dirname "$GGUF")"
echo "repo : $REPO_ID"
echo "into : $DIR"

PY="${PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv-rocm/bin/python}"

for f in chat_template.jinja generation_config.json tokenizer_config.json; do
    if "$PY" - "$REPO_ID" "$f" "$DIR" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo, fname, dest_dir = sys.argv[1:4]
try:
    p = hf_hub_download(repo_id=repo, filename=fname)
except Exception:
    sys.exit(1)
import shutil
shutil.copyfile(p, f"{dest_dir}/{fname}")
print(f"{fname}: OK")
EOF
    then
        :
    else
        echo "$f: not present upstream, skipped"
    fi
done

echo "done — restart the server to pick them up."
