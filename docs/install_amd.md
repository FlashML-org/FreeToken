# AMD ROCm installation (WIP)

AMD support is experimental and remains a work in progress. See the
[AMD Support roadmap](https://github.com/FlashML-org/FreeToken/issues/541) for
the current integration and qualification status.

## Requirements

- Linux x86_64
- AMD RDNA3/RDNA4 GPU (`gfx1100`-`gfx1103`, `gfx1200`, or `gfx1201`)
- ROCm 7.14
- Python >= 3.10

## Install from source

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

## Optional kernel backends

On ROCm, FreeToken does not probe or select its CUDA-only FlashInfer,
`sgl_kernel`, or vLLM kernel integrations, even if those packages are installed.
This restriction concerns the kernels FreeToken imports, not whether the
packages offer other AMD implementations. Do not install the `[accel]`, `[fi]`,
or `[sgl]` extras to enable these paths on AMD.

Generic attention with `--attention-backend auto` selects Triton; activation and
normalization use their Triton/PyTorch fallbacks. NVFP4 linear and MoE selection
also supports the native Triton kernels. Explicitly requesting FlashInfer/SGL
attention, NVFP4 Marlin, or NVFP4 b12x fails early with a ROCm-specific error.
Model-specific attention requirements still apply; this does not make every
checkpoint ROCm-compatible. CUDA discovery and kernel preferences are unchanged.
