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

FreeToken's FlashInfer, `sgl_kernel`, and vLLM kernel integrations are CUDA-only.
On ROCm, these paths use the built-in fallbacks; do not install the `[accel]`,
`[fi]`, or `[sgl]` extras. Forcing `--attention-backend fi`, `fa`, or `trtllm`,
or NVFP4 Marlin/b12x (e.g. `--quant-backend moe.nvfp4=marlin` or
`--quant-backend moe.nvfp4=b12x`), fails with a ROCm-specific error.

## CPU and hybrid MoE

`--moe-strategy cpu`, `hybrid` and `--moe-cpu-layers` sync the GPU with the CPU
expert pool through HIP stream memory operations. As on CUDA, BF16 inputs use
kernel IO when its startup probe passes: a publish kernel copies the inputs to
mapped host memory and notifies the CPU; a combine kernel reads the CPU output.
The publish kernel establishes the memory ordering before notification and
skips CPU work when no route needs it.

When kernel IO is unavailable or the input is not BF16, FreeToken uses copies.
With the memop handshake, this fallback records a system-release event after
the input copies and before notifying the CPU. The event is recorded as an
external graph node so that every replay retains the publication point. The
kernel IO path does not record this event. If the memop handshake is unavailable,
eager execution uses host callbacks instead. No custom HIP library is required.

Native memop capture is detected at runtime. If it is unavailable, FreeToken
adds explicit memop graph nodes. Before model capture, a numerical check
compares eager and Graph outputs against synchronous CPU execution, including
changing routes, skipped tasks and Hybrid output accumulation. If kernel IO
fails this check, FreeToken switches to the event + memcpy path and checks it
again. Startup checks only the model's activation dtype; BF16 can use kernel IO,
while FP16 and FP32 use copies. When decode graphs are disabled, the numerical
check runs eagerly without capturing a graph. If the Graph path is unsafe,
startup logs a warning and disables decode graphs for the engine. TP ranks
agree on this decision, so one unsafe rank disables graphs for all ranks.
A failed eager check, or a capture/runtime API error, stops startup rather
than allowing unverified output.

After updating the native CPU MoE sources, rebuild the extension with
`python setup.py build_ext --inplace` in the ROCm environment before serving.
