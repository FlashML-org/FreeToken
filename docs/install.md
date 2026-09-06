# Install

## Requirements

- Linux x86_64 with either an NVIDIA GPU, driver r580+ (CUDA 13), or an AMD
  GPU with a ROCm-enabled PyTorch build
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

### AMD ROCm source install (experimental)

ROCm support is hardware- and version-specific. Do not treat compilation or a
server boot as serving support; use the target matrix and validation commands
below. The reference container uses this tuple:

| component | reference tuple |
| --- | --- |
| PyTorch | 2.11.0 ROCm build |
| ROCm/HIP SDK | 7.14.x, matching `hipcc` |
| Triton | 3.7.x supplied by the ROCm image |
| target | actual `gcnArchName`, passed with `FREETOKEN_ROCM_ARCH` |

Recorded gfx1100 serving smoke used ROCm 7.2.1. The reference 7.14.x container
does not substitute for physical target evidence.

Example container for cross-compilation or hardware validation:

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

Inside container, preserve image's ROCm PyTorch and disable build isolation:

```bash
python -m pip install --no-build-isolation -e .
```

Replace `gfx1201` with target reported by `rocminfo`. Unsupported or untested
targets must remain explicit fallback/unsupported results.

Current ROCm target matrix:

| target family | targets | current status |
| --- | --- | --- |
| RDNA3 | gfx1100, gfx1101, gfx1102, gfx1103 | gfx1100 served GGUF smoke tested; other targets compile-only |
| RDNA3.5 | gfx1150, gfx1151 | compile-only |
| RDNA4 | gfx1200, gfx1201 | compile-only |

Currently only gfx1100 has end-to-end serving smoke evidence. Native
Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 dispatch is target-matrix validated and remains
fail-closed for unknown targets; gfx1150/1151/gfx1200/1201 still need physical
served-request, finite-logit, graph replay, and native/reference completion evidence.

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
