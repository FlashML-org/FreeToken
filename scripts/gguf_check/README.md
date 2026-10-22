# gguf_check

Checks for GGUF support of a model family: does FreeToken load most of the GGUF files people publish for it, and does it read them right.
Run from the repo root with `PYTHONPATH=python` and transformers 5.18.

| Script | Needs | Checks |
|---|---|---|
| `fetch_headers.py` | network | nothing, writes header-only stubs of HF GGUF files |
| `probe_kernels.py` | GPU | which ggml ops this build runs per ggml type |
| `check.py` | CPU | config, dry load, expert banks, kernel coverage, tokenizer, chat template, and the vision tower of an mmproj next to the file, per file |
| `tok_compare.py` | CPU | GGUF tokenizer vs the model's HF `tokenizer.json` |

## Coverage: most published files load

```bash
# header-only stubs of every GGUF in the repos (about 15 MB each, weights read as zeros)
python scripts/gguf_check/fetch_headers.py ~/gguf-stubs --search GLM-4.5 --search GLM-4.7 unsloth/GLM-4.7-GGUF > fetch.jsonl

# once per build and GPU architecture
python scripts/gguf_check/probe_kernels.py kernels.json

python scripts/gguf_check/check.py --kernels kernels.json --out glm.jsonl ~/gguf-stubs
python scripts/gguf_check/check.py --report glm.jsonl
```

`check.py` builds the model on the meta device and runs the family's weight reader on meta matrices, so a worker needs no RAM for the weights; it falls back to real reads only for a reader that needs the values.
It prints `passed/total` per served GGUF architecture, the files of architectures FreeToken does not serve (draft models, ...), and the failures grouped by their first problem.
Every failure should be one of:

- the family's loader refuses the file with a clear error (a type or layout we do not serve), or
- the file is broken.

List the refused kinds under "Not covered" in the PR.
Passing means the server builds the model and finds every tensor and kernel. It does not read the weights or check the model fits in memory.

## Correctness: the files load right

On one or two real files per family, smallest and most common quant:

- `tok_compare.py <gguf> <HF model dir>`: 0 differing texts. The reference is the model's own tokenizer, not llama.cpp.
- `ft serve --model-path <file>`: a short eval (e.g. AIME problems) against llama.cpp on the same file, greedy outputs against llama.cpp, 48 staggered requests at concurrency 16 with no empty or corrupted output, and bs=1 tok/s.
- `pytest tests/models tests/moe tests/kernels/test_ggml.py tests/server`.
