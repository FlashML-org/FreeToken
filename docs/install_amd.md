# AMD ROCm installation (WIP)

AMD support is experimental and remains a work in progress. See the
[AMD Support roadmap](https://github.com/FlashML-org/FreeToken/issues/541) for
the current integration and qualification status.

## Requirements

- Linux x86_64
- AMD RDNA3/RDNA4 GPU (`gfx1100`-`gfx1103`, `gfx1200`, or `gfx1201`)
- ROCm 7.14
- Python >= 3.10
- PyTorch 2.11–2.14 with Triton 3.6–3.8 (both stacks are supported; see below)

## Choose a ROCm PyTorch image

Use an official ROCm PyTorch image whose Torch/Triton versions fall in the ranges
above. Two known-good options for RDNA4 (`gfx1201`):

- **torch 2.11 / triton 3.7** (the original pin):
  `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`
- **torch 2.14 / triton 3.8** (latest stable; validated end-to-end on this branch):
  a ROCm 7.14 image carrying `torch 2.14.0+rocm7.14` and `triton-rocm 3.8.0`.

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

Set both architecture variables to the target reported by `rocminfo`
(`gfx1100` for W7900, `gfx1201` for R9700, `gfx1200` for the RX 9060 family).

## Build the extensions

Preserve the ROCm-enabled PyTorch already supplied by the image and disable build
isolation so it is also used to compile the extensions. On ROCm the tree has no
CUDA-only build dependencies, so also pass `--no-deps` (the dependency install is
curated below — a plain `pip install -e .` pulls CUDA packages; see the warning):

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Torch 2.14 headers require C++20; the build already sets `-std=c++20`. The
`_cpu_moe` extension additionally links `hiprtc` on ROCm (used for the CUDA-graph
flag handshake). If you edit any C/HIP source, rebuild in place with:

```bash
python setup.py build_ext --inplace
```

## Runtime dependencies

> **Do not run a plain `pip install -e .` (with dependencies) on ROCm.** The
> project's dependency set is authored for NVIDIA: `flashlib` pulls `numba` +
> `nvidia-cutlass-dsl`, and Torch's metadata pulls `cuda-bindings`. A full resolve
> wastes hundreds of MB and can replace the image's ROCm Torch with a CUDA build.

Install the runtime dependencies explicitly instead. These are pure-Python /
portable wheels that do not depend on Torch or CUDA:

```bash
python -m pip install \
  transformers tokenizers safetensors huggingface_hub \
  fastapi uvicorn pydantic openai httpx anyio \
  einops gguf msgpack pyzmq typer rich apache-tvm-ffi \
  partial-json-parser
```

Two components need special handling on ROCm:

```bash
# flashlib: import-only here (FreeToken reads slot-cache constants from it). Its
# numba / nvidia-cutlass-dsl deps are CUDA-only and unused on ROCm -> --no-deps.
python -m pip install --no-deps flashlib==0.3.0

# ninja: required by the QSA sparse-attention kernels, which JIT-compile a small
# runtime extension.
python -m pip install ninja
```

`flashinfer`, `sgl_kernel`, and `nvidia-cutlass-dsl` have **no ROCm builds** and are
intentionally left uninstalled — FreeToken gates them off and falls back to Triton
on ROCm. A handful of tests that hard-require them (the `fi` attention backend, the
sgl `topk_softmax` semantics) are expected to fail on ROCm.

## Running the tests

```bash
python -m pip install pytest pytest-timeout            # test harness
export PYTHONPATH="$PWD/python"
python -m pytest tests --ignore=tests/e2e/test_aime.py -q \
  --continue-on-collection-errors
```

`--continue-on-collection-errors` keeps a missing optional import (e.g. a server
test's parser dep) from aborting the whole collection. `tests/e2e/test_aime.py` is
excluded because it needs the full (multi-hundred-GB) checkpoint and multiple GPUs.

On ROCm 7.14 / torch 2.14 / triton 3.8 / `gfx1201` the suite reports ~1734 passed.
The remaining failures are **not** RDNA-specific bugs; they fall into:

- tests that require CUDA-only packages absent on ROCm (`flashinfer` / `sgl_kernel`
  attention/router backends; native-fp8 `sm_89+` references),
- inherent bf16-vs-fp32 precision / tie-break edges in the QSA and GLM-DSA sparse
  indexers,
- HF-reference models that are incompatible with the installed `transformers`.
