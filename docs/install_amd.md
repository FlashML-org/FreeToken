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
