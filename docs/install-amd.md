# AMD GPU (ROCm) support

FreeToken targets Linux + NVIDIA CUDA by default. AMD (ROCm) is a supported, tested
configuration with a **single-GPU** milestone: correct functional path first, performance
recovered via HIP ports where safe. This page covers installing and running on RX 7000.

> Status: **experimental.** The default and best-tested path remains CUDA. AMD brings up a
> correct functional path (Triton attention + offload/CPU MoE + portable quant) and is
> recovering performance via the HIP kernel ports.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Linux x86_64 (Windows WDDM pinned-memory is a known edge, not supported yet) |
| GPU | AMD RX 7000 (RDNA 3, `gfx1100`); RX 9000 (`gfx1201`) is future work |
| ROCm | ROCm toolkit with `hipcc` (`/opt/rocm/bin/hipcc` or on `PATH`) |
| torch | ROCm build, e.g. `torch==2.5.1+rocm6.2` |

The build refuses to mix toolchains: it will **not** silently fall back to `nvcc`/`libcudart`
when only the ROCm toolkit is present, and vice versa.

## Install

```bash
# ROCm torch (PyTorch official ROCm wheels) -- must satisfy the repo's torch>=2.11,<2.12
# build pin, so use the rocm7.2 index (rocm6.2 only carries torch up to 2.5.1).
pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
    "torch==2.11.0+rocm7.2" torchvision triton-rocm==3.6.0

# FreeToken with the ROCm extra (builds the native extensions with hipcc)
uv pip install -e ".[rocm]" --no-build-isolation
```

`pip install ".[rocm]"` pulls ROCm-compatible `torch`/`triton`; the NVIDIA-only `[accel]`
packages (`flashinfer`, `sgl-kernel`, `triton_kernels`, Marlin) are **not** installed on AMD
and their backends are rejected with a clean error if requested.

## Verified feature matrix

| Feature | On AMD | Notes |
| --- | --- | --- |
| Attention | `--attention-backend triton` | flashinfer/fa/trtllm are NVIDIA-only and rejected |
| MoE | `--moe-backend offload / cpu / hybrid` | offload needs pinned host memory |
| Quant | BF16, MXFP4, GGUF (Q4_K/Q8_0), Triton inline-dequant NVFP4 | Marlin INT4 / native NVFP4 SASS unavailable |
| NVFP4 checkpoints with no MXFP4 variant | converted to MXFP4 on load (auto) | `--nvfp4-backend auto` → triton/MXFP4 |
| CUDA graphs (decode) | HIP graph capture **if** the capture probe passes | otherwise kernel-launch decode |
| Multi-GPU (RCCL) | out of scope (single-GPU milestone) | |

## CLI behavior on AMD

* `--nvfp4-backend marlin` / `flashinfer` → error (NVIDIA-only). Use `triton` / `auto`.
* `--attention-backend fi` / `fa` / `trtllm` → error (NVIDIA-only). Use `triton` / `auto`.
* `--moe-backend fused` → warning (fused MoE is CUDA-only; falls back to offload/cpu).
* `--nvfp4-backend auto` → resolves to the portable Triton inline-dequant path (or MXFP4
  for a converted checkpoint).

## Verify

```bash
ft version            # prints an AMD / ROCm banner
ft serve --model Qwen3.6-35B-A3B \
  --moe-backend offload --attention-backend triton --nvfp4-backend auto
```

`ldd` of the built `.so` should show `hiprt`/`amdhip64`, not `libcudart`.

## AOT kernel cache

Build the prebuilt `+rocm` kernel-cache wheel (no nvcc needed on the target):

```bash
scripts/build-release-wheels.sh   # on a ROCm torch + hipcc box; tags the cache +rocm
```

The runtime refuses to pair a `+rocm` cache with a `+cu130` runtime (and vice versa).

## Notes / limitations

* `nvtx_annotate` is a no-op on ROCm; roctx profiling is future work.
* FP8 / NVFP4-class formats: BF16 / MXFP4 / GGUF are the supported AMD matrix; performance
  parity vs CUDA is not guaranteed for NVFP4-class formats.
* Windows AMD is not yet supported (WDDM zero-copy semantics differ).

## Performance (gfx1100 / RX 7900 XTX, as of rocm-perf-parity)

Measured with `benchmarks/bench_decode_moe.py` on Qwen3.6-35B-A3B GGUF (Q4_K_M),
3-run medians, same prompt/sampling protocol throughout (see
`.plans/rocm-perf-parity/` for artifacts and the stage-time profiling notes):

| state | decode tok/s (median) |
| --- | --- |
| pre-plan baseline (kernel-launch decode, pure-torch router) | 34.69 |
| + in-repo fused triton router | 37.67 |
| + CUDA-graph decode capture (`prewarm` variant) | **45.09** |

What is enabled on AMD now:

- **Fused triton router**: `fused_topk` routes ROCm to the in-repo
  `kernel/triton/moe_router.fused_topk_softmax` (no `triton_kernels` install needed).
  The pure-torch fallback remains for CPU/Windows and is flagged by the
  `pure-torch router fallback` log line.
- **CUDA-graph decode**: the capture gate (`utils/graph_gate.py`) now probes capture
  *variants* — `default`, `rocblas` (`TORCH_BLAS_PREFER_HIPBLASLT=0`),
  `prewarm` (one GEMM before capture; the winner — hipBLASLt's lazy workspace
  allocation inside capture was the abort), and `rocblas+prewarm`. The winning
  variant's env is applied to the engine worker at spawn. The gate result is cached
  at `~/.cache/freetoken/freetoken_graph_gate.json`; delete it to re-probe after a
  driver/ROCm upgrade (the cache format invalidates itself automatically).
- Known-negative tuning results (documented, do not redo blindly):
  `FREETOKEN_GGUF_MMV_Y` (rows/warp for the GGUF MMVQ launches): 1/2/4/8 within run
  noise AND byte-identical outputs — keep 1. Decode attention kernel
  (`BLOCK_N=32/warps=4`): best of the swept grid; torch-SDPA gathered attention is
  4.5× slower on gfx1100.

Escape hatches for runtime bisection:

- `FREETOKEN_TORCH_PROFILE=<warm>:<steps>:<out.json>` — torch.profiler trace of a
  decode window (`scripts/profile-rocm-decode.sh` wraps this end-to-end).
- `FREETOKEN_GGUF_MMV_Y` — GGUF MMVQ rows-per-warp build knob (see above; JIT cache
  is keyed on the value).
- `--no-graph` on `benchmarks/bench_decode_moe.py`, or `--cuda-graph-max-bs 0`, to
  fall back to kernel-launch decode; `scripts/serve-qwen-moe.sh stop` clears a wedged
  server (a `kill -9` during a JIT rebuild leaves a stale torch-extension lock that
  `kernel/gguf.py` now clears automatically once older than 3 h).
