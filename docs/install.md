# Install

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use. An explicit absolute `CUDA_HOME`
must contain an `nvcc` matching PyTorch's exact CUDA release; otherwise
FreeToken checks the matching `/usr/local/cuda-X.Y` toolkit and then `PATH`.
JIT and kernel-cache builds fail if no exact compiler is available. Toolkit
selection does not change the supported PyTorch, driver, or GPU matrix. A
complete prebuilt kernel cache does not require a compiler at server startup.

## Method 2: Install from source

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Method 3: Nightly wheels

Every night `main` is built into a wheel pair on the rolling
[`nightly` release](https://github.com/FlashML-org/FreeToken/releases/tag/nightly):
the `freetoken` runtime (CPython 3.12, Linux x86_64) and the matching
`freetoken-kernel-cache` with FreeToken's own CUDA kernels prebuilt.
Install both from the URLs on that release page:

```bash
uv pip install \
  "freetoken[accel] @ https://github.com/FlashML-org/FreeToken/releases/download/nightly/<runtime wheel>" \
  "https://github.com/FlashML-org/FreeToken/releases/download/nightly/<kernel-cache wheel>"
```

Filenames carry a `+g<sha>` build stamp and change every night, and the `nightly`
tag moves with them. Pin a wheel URL, never the tag. A local copy of the tag goes
stale: `git fetch` leaves it alone and `git fetch --tags` refuses to overwrite it;
refresh it with `git fetch --force origin tag nightly`. `engine-linux_x86_64.json`
next to the wheels names the current pair for scripts.

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
