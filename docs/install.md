# Install

## Requirements

- Linux x86_64 with either:
  - NVIDIA GPU, driver r580+ (CUDA 13), or
  - AMD RDNA3/RDNA4 GPU (`gfx1100`-`gfx1103`, `gfx1200`, or `gfx1201`) with ROCm 7.14
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

### AMD ROCm source install (experimental)

Use an official ROCm PyTorch image whose PyTorch version satisfies the project's
`torch>=2.11,<2.12` constraint. For RDNA4, the matching ROCm 7.14 image is:

```bash
VIDEO_GID="$(getent group video | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add="$VIDEO_GID" --group-add="$RENDER_GID" --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e PYTORCH_ROCM_ARCH=gfx1201 -e FREETOKEN_ROCM_ARCH=gfx1201 \
  -v "$PWD:/workspace/FreeToken" -w /workspace/FreeToken \
  rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0 bash
```

Inside the container, preserve the ROCm-enabled PyTorch already supplied by the
image and disable build isolation so it is also used to compile the extensions:

```bash
python -m pip install --no-build-isolation -e .
```

Set both architecture variables to `gfx1200` for RX 9060 family GPUs, or to the
actual target reported by `rocminfo`.

The optional native GGUF kernels also require Thrust headers. Install the generic
headers before the first GGUF kernel JIT build:

```bash
apt-get update && apt-get install -y --no-install-recommends libthrust-dev
```

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
