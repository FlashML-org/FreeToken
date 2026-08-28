# Install

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## sm_70 (Volta / Tesla V100) build

Torch 2.11 — the only torch line upstream FreeToken supports — dropped sm_70
kernels in every CUDA build, so the engine cannot run on Volta GPUs with the
stock dependency pins. This branch relaxes the torch floor to 2.10.x and
repoints the CUDA index to the cu128 build, which is the last torch line that
ships sm_70 cubins and runs on driver r570 (CUDA 12.8) — no driver upgrade and
no reboot required.

Build it on the target machine (needs a CUDA 12.x toolkit with `nvcc` on PATH
for the C++ extensions and the JIT kernels; CUDA 12.9 was used here):

```bash
git clone https://github.com/redoop/FreeToken.git
cd FreeToken
git checkout feat/support-sm_70
export PATH=/usr/local/cuda-12.9/bin:$PATH CUDA_HOME=/usr/local/cuda-12.9
uv venv --python 3.12 .venv-sm70
uv pip install -e . --python .venv-sm70/bin/python
```

Notes for the sm_70 build:

- Install the **base** package (no `[accel]`): flashinfer/sglang fused kernels
  target sm_80+ and are not usable on Volta; the runtime falls back to the
  pure-Triton kernels automatically.
- `freetoken_kernel_cache` (cu130) is skipped; JIT kernels compile on first use
  with the local nvcc.
- On 16 GiB cards (V100-16GB), lower the serve memory ratio so requests have
  headroom:
  `ft serve --model <dir> --memory-ratio 0.75` (default 0.9 OOMs).

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

## Method 2: Install from source

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
