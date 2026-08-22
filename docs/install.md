# Install

## Requirements

**Linux (CUDA — full engine)**

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

**macOS (Apple Silicon — Metal backend)**

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python >= 3.10
- One of the Metal engines: `mlx-lm` (recommended) or `llama.cpp`'s
  `llama-server` on PATH
- No CUDA, triton, flashinfer, sglang-kernel, or `flashlib` needed — the
  Metal path does not import FreeToken's CUDA stack at all

## Method 1: Install from PyPI (Linux/CUDA)

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

## Method 2: Install from source (Linux/CUDA)

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Method 3: macOS / Apple Silicon (Metal backend)

FreeToken's native engine is CUDA-only. On a Mac, `ft serve-metal` runs the
same API surface backed by Apple's own Metal runtimes instead, so nothing
needs porting: it launches Apple's `mlx_lm.server` or llama.cpp's
`llama-server` as the upstream engine and proxies the FreeToken wire surface.

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate

# Core package only — skips flashlib, triton, sglang-kernel and the CUDA-linked
# torch index pins, none of which have macOS wheels.
uv pip install -e .

# Then add one Metal engine:
uv pip install mlx-lm            # option A: MLX (Apple's framework)
brew install llama.cpp           # option B: llama.cpp's llama-server (Metal)
```

> **Why the CUDA deps are skipped:** `flashlib`, `triton==3.6.0`, and the cu130
> torch pins are marked `platform_system == 'Linux'` in pyproject.toml — they
> only serve the native scheduler, and the Metal path never imports them. A
> plain `uv pip install -e .` resolves everything else from PyPI.

```bash
# Verify (no model needed):
ft --version
ft serve-metal --help
```

### Troubleshooting: "no wheels with a matching Python implementation tag"

Your venv is an **x86_64 build running under Rosetta**, not native arm64 — no
arm64-only wheel (mlx, macOS torch) can ever install into it. This happens when
`uv venv` / `python3 -m venv` is run inside an x86_64 (Intel) terminal session.

Fix — recreate the venv natively:

```bash
deactivate 2>/dev/null; rm -rf .venv
# make sure the shell itself is native: arch -arm64 zsh   (or use an arm64 Terminal)
uv venv && source .venv/bin/activate
file .venv/bin/python   # must say: arm64   (not x86_64)
uv pip install -e . && uv pip install mlx-lm
```

Also run installs from the repo root — `uv pip install -e .` needs the
`pyproject.toml` in the current directory, which is why a stray
`error: Requesting extras requires a pyproject.toml ...` means you are in the
wrong directory (or asked for a nonexistent extra; the only extra is `dev`).

## Verify (Linux/CUDA)

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
