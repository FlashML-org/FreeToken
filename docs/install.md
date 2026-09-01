# Install

## Requirements

- Linux x86_64
  - **NVIDIA:** driver r580+ (CUDA 13), Ampere+
  - **AMD:** ROCm / HIP, `gfx1201` first (Radeon AI PRO R9700 / RX 9070 XT). See
    [amd.md](amd.md). Native HIP only — not ZLUDA.
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

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

## AMD / ROCm (Linux)

The default extras pull **CUDA 13** PyTorch. On an AMD box that wheel will not
run. Follow [amd.md](amd.md) for a ROCm torch install, `FREETOKEN_GFX_ARCH`,
and the on-box checks for a dual R9700 (`gfx1201`) workstation.
