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
| MoE | `--moe-backend fused / offload / cpu / hybrid` | GGUF `fused` is native resident only when preflight fits; offload needs pinned host memory |
| Quant | BF16, MXFP4, GGUF (Q4_K/Q5_K/Q6_K/Q8_0), Triton inline-dequant NVFP4 | Marlin INT4 / native NVFP4 SASS unavailable |
| NVFP4 checkpoints with no MXFP4 variant | converted to MXFP4 on load (auto) | `--nvfp4-backend auto` → triton/MXFP4 |
| CUDA graphs (decode) | HIP graph capture **if** the capture probe passes | otherwise kernel-launch decode |
| Multi-GPU (RCCL) | out of scope (single-GPU milestone) | |

## CLI behavior on AMD

* `--nvfp4-backend marlin` / `flashinfer` → error (NVIDIA-only). Use `triton` / `auto`.
* `--attention-backend fi` / `fa` / `trtllm` → error (NVIDIA-only). Use `triton` / `auto`.
* `--moe-backend fused` → native GGUF resident path when allocation-free fit check passes;
  explicit fit failure is an error. `--moe-backend auto` selects native residency only when it
  fits, otherwise existing offload.
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

* `nvtx_annotate` is a no-op on ROCm; `FREETOKEN_ROCTX_MARKERS=1` enables low-overhead
  ROCTX ranges, and `scripts/profile-rocm-decode.sh PROFILE_MODE=rocprofv3` captures
  runtime, marker, and kernel traces when `rocprofv3` is installed.
* FP8 / NVFP4-class formats: BF16 / MXFP4 / GGUF are the supported AMD matrix; performance
  parity vs CUDA is not guaranteed for NVFP4-class formats.
* Windows AMD is not yet supported (WDDM zero-copy semantics differ).

## Performance (gfx1100 / RX 7900 XTX, as of qwen-moe-speed)

Measured with `benchmarks/bench_decode_moe.py` on Qwen3.6-35B-A3B GGUF (Q4_K_M),
3-run medians, same prompt/sampling protocol throughout (see
`.plans/rocm-perf-parity/` for historical artifacts and the stage-time profiling notes):

| state | decode tok/s (median) |
| --- | --- |
| pre-plan baseline (kernel-launch decode, pure-torch router) | 34.69 |
| + in-repo fused triton router | 37.67 |
| + CUDA-graph decode capture (`prewarm` variant) | **45.09** |

Current Qwen3.6 base-speed gate on current HEAD, using the exact FreeToken
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` file, one request, 512 generated tokens,
offload MoE, Triton attention, and graph replay:

| lane | measured decode tok/s (median) |
| --- | --- |
| FreeToken sampled, 10 runs | 62.295 |
| FreeToken greedy, 3 runs | 59.650 |
| FreeToken graph-off control, Inc2 baseline | 38.611 |
| Ollama sampled, 3 runs, client arrival | 80.225 |

Inc 4 native GPU-offload diagnostic (one exact 128-token graph run, not a promotion
median): **58.98 tok/s** API arrival, **61.19 tok/s** scheduler timing. Startup fell
from roughly 3.5 minutes for legacy Q8 conversion to roughly 19 seconds; decode did
not materially improve. This rejects loader conversion as sole cause and leaves
attention/dense/MoE dispatch and backend differences as the remaining speed gap.

This gate did not clear the 75 tok/s external floor or the 80 tok/s engineering
target. Ollama was configured without MTP, but its Q4_K_M model blob differs
from FreeToken's GGUF file; treat 80.225 tok/s as directional, not an exact
same-file claim. Full provenance and JSONL artifacts are in
`.plans/qwen-moe-speed/notes-baseline.md` and `notes-profile.md`.

### ROCm execution policy

`FREETOKEN_ROCM_BLAS=auto|hipblas|hipblaslt|rocblas` selects the ROCm BLAS
preference before workers start. `rocblas` is an alias for hipBLAS; explicit
requests fail when the installed PyTorch API cannot honor them. Startup logs and
benchmark JSON report requested and effective policy. Changing BLAS variables
after worker startup has no effect.

KV storage is explicit: `--kv-type bf16` (default), `fp16`, or opt-in `q8_0`.
Q8 uses the pinned 32-value row contract with FP16 scales and currently accepts
plain full-attention MHA groups only. Allocation, store, decode, prefill, and
pointer-generation metadata must all report q8 before a q8 benchmark row is
eligible. Roll back with `--kv-type bf16`.

ROCm GGUF JIT stages its `.cu` entrypoint into the torch extension cache before
PyTorch HIPify runs. This keeps generated HIP intermediates and tracked checkout
headers separate; delete only the relevant torch-extension cache after toolchain
upgrades if a rebuild is needed.

GGUF Q4_K/Q5_K/Q6_K/Q8_0 MoE uses MMVQ for decode. Native Q5_K/Q6_K rows retain
their source type; Q5_K cache rows carry zeroed Q6_K-sized tails and kernels receive
explicit expert and row strides. A grouped-MMQ ABI probe exists, but real-model route
validation exposed an illegal HIP access, so it is opt-in only via
`FREETOKEN_GGUF_GROUPED_PREFILL=1`; default prefill stays on proven vector dispatch.
The `gfx1100` MoE kernel is available only as an explicit, forced candidate; `legacy`
remains default because measured candidate throughput was slower.

Greedy/no-penalty decode may opt into fixed-address sampler graph capture:

```bash
FREETOKEN_GRAPH_SAMPLER=1 ft serve --model /path/to/model.gguf \
  --moe-backend offload --attention-backend triton
```

Dynamic temperature/top-k/top-p, penalties, and unsupported sampling modes use
the regular sampler path. Unset `FREETOKEN_GRAPH_SAMPLER` to retain default
behavior. Use `FREETOKEN_GGUF_MOE_IMPL=legacy` for explicit rollback:

```bash
export FREETOKEN_GGUF_MOE_IMPL=legacy
unset FREETOKEN_GRAPH_SAMPLER
ft serve --model /path/to/model.gguf --moe-backend offload --attention-backend triton
```

Final promotion requires ten accepted runs, exact 512-token completion, stable
sampling/output checks, a matching finite-logit/eager-graph probe, p02.5 bootstrap
lower bound above 75 tok/s, no run below 70 tok/s, and a matched full-file-SHA-256
llama.cpp/Ollama reference. Run `benchmarks/probe_qwen_moe_base.py` and pass its
JSON to `benchmarks/check_decode_gate.py --probe`; missing probe evidence fails
closed. The current 80.225 tok/s Ollama result lacks byte identity and is
directional only. Gate A covers sampled absolute throughput. Gate B is separate:
q8/q8 teacher-forced replay, matched IDs and route hashes, paired bootstrap delta.
Local CUDA compilation was unavailable (`nvcc` absent); NVIDIA compilation remains
a CI gate.

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
