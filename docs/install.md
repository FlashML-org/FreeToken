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

- Install the **base** package (**no `[accel]`**): flashinfer/sglang fused kernels
  target sm_80+/sm_100+ and ship CUDA 13 builds only, so they are not usable on
  Volta — if they are present the engine fails to load with `common_ops` /
  `libnvrtc.so.13` / `no kernel image` errors. The runtime falls back to the
  pure-Triton kernels automatically, which is what Volta uses.
- If you upgraded an existing cu130 venv in place (e.g. a FreeToken Desktop
  engine venv), also uninstall the accel native kernels and any leftover cu13
  libs so the cu12 runtime is the one loaded:
  `uv pip uninstall --python <venv>/bin/python sglang-kernel flashinfer-python`
  and remove the lingering `nvidia/cu13` package dir under the venv site-packages.
  The FreeToken Desktop engine venv (typically `~/.freetoken/venv` or
  `~/.freetoken-cli/.venv`) needs the same treatment.
- `freetoken_kernel_cache` (cu130) is skipped; JIT kernels compile on first use
  with the local nvcc.
- On 16 GiB cards (V100-16GB), lower the serve memory ratio so requests have
  headroom:
  `ft serve --model <dir> --memory-ratio 0.75` (default 0.9 OOMs).

## Serving on 16 GiB Volta with large system prompts

A 35B-class MoE on a V100-16GB with expert offload needs careful memory tuning
once the client sends a large system prompt (e.g. agent harnesses that inject
~8K tokens of runtime context into every request). Measured on
Qwen3.6-35B-A3B-FP8 + r570:

- `--memory-ratio 0.9` (default): OOMs on the 8K-token prefill (~2.3 GiB
  headroom is not enough for the prefill activations).
- `--memory-ratio 0.6`: fits the prefill but the smaller expert cache collapses
  decode to ~0.1-0.4 tok/s (expert fetch thrash over PCIe).
- **`--memory-ratio 0.75 --moe-cache-rate 0.2 --num-tokens 16384`**: ~4.1 GiB
  free after init; the 8099-token system prompt prefills at ~800 tok/s without
  OOM and decode stays at ~20 tok/s for normal prompts. This is the setting to
  run under a client that always streams with a big system prompt.
- `--moe-cache-rate` is a fraction of all experts (0.4 of the 30 GiB banks
  alone OOMs at startup on 16 GiB); 0.2 is the practical ceiling here.
- Streaming is OpenAI-compatible (progressive `data:` chunks and a terminal
  `data: [DONE]`), verified against the OpenAI SDK client; the earlier "stream
  hangs with one empty chunk" symptom was queue starvation, not the wire format.
- OpenAI-compatible gateway clients (e.g. a DSH `llm-pi-ai` provider route)
  must supply `api: openai-completions`, `baseURL: <host>:<port>/v1`, and an
  `apiKeyEnv` credential — pi-ai's adapter requires a key string even when the
  gateway ignores it.

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
