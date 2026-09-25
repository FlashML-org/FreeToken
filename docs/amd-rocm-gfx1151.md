# FreeToken AMD ROCm on Radeon 8060S `gfx1151`

## Purpose

This branch ports the FreeToken serving runtime to native AMD ROCm and HIP on
the AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`).  The port preserves
the NVIDIA implementation as a separate runtime path.  It does not use Vulkan
or a CPU-only runner as a substitute for native GPU execution.

The intended first deployment host is LAN-223.  It serves the same local API
surface as upstream FreeToken, including OpenAI-compatible endpoints, while
using HIP-compiled extensions and AMD Triton kernels.

## Scope and parity contract

The port is complete only when the target model can load and serve through
`ft serve`, return a coherent streamed and non-streamed OpenAI-compatible
response, and exercise the applicable FreeToken cache and MoE paths.  The
initial full-model validation set is:

1. `Qwen/Qwen3.6-35B-A3B`, FreeToken's primary consumer-hardware MoE
   benchmark model.
2. The current Gemma 4 MoE GGUF accepted by FreeToken's native Gemma loader.

The project records correctness, stability, API behavior, GPU memory, host
memory, prefill throughput, decode throughput, TTFT, temperature, clocks, and
throttling.  NVIDIA GPU tokens per second are context, not an AMD acceptance
threshold: LAN-223 uses a shared-memory APU rather than discrete VRAM and
PCIe.

### Known scope boundary

This validation targets `gfx1151`; it is not a claim of generic AMD
architecture support. A separate public [gfx1011/ROCm 10.2 report](https://github.com/FlashML-org/flashlib/issues/24)
describes clean prefill followed by corrupted decode in a routed-expert
offload path and suspects `flashlib.kernels.slot_cache.lru_ensure`. That
report has not been reproduced on `gfx1151`, and its author did not establish
the kernel root cause. Until revision-scoped target tests exercise both
cache hits and misses, this branch does not claim general AMD cache-admission
correctness.

## What this branch changes

The code is deliberately gated at the narrowest possible boundary so CUDA
behavior stays unchanged.

- `setup.py` detects a ROCm PyTorch build and links the two native extensions
  to `libamdhip64` instead of `libcudart`.
- `kernel/csrc/hip_compat.h` maps the small CUDA Runtime API subset used by
  FreeToken's pinned-memory and CPU MoE extensions to HIP equivalents.
- CUDA JIT compilation removes NVCC-only flags on HIP and replaces CUDA-only
  launch behavior with compatible HIP launch behavior.
- Triton paths avoid NVIDIA PTX inline assembly, Hopper Programmatic Dependent
  Launch controls, and CUDA tile assumptions when PyTorch reports HIP.
- CUDA-only optional package probes are suppressed on HIP.  The pure Triton
  implementations remain the portable GPU fast path.
- NVIDIA SM feature gates reject ROCm before numerical capability comparison.
  This matters because PyTorch presents HIP devices under `torch.cuda` for
  compatibility, and `gfx1151` must never be interpreted as a new NVIDIA SM.

## Clean LAN-223 installation

**Qualification status: partial; gfx1151 runtime remains unqualified.** On
2026-09-25, the current candidate dependency set installed in a disposable
Linux/x86_64 Python 3.14 environment on LAN-215 using AMD's ROCm 7.14.0 index
and `gfx1151` device extras. `rocm-sdk init` exposed HIP 7.14.60850, and
`pip install --no-build-isolation -e '.[rocm]'` built the pinned-tensor,
CPU-MoE, and row-store native extensions. `pip check`, native-extension
imports, the `flashlib` slot-cache import, and nine no-hardware install/setup
contract tests passed. The GPU was hidden during the build and test/import
checks; LAN-215 is a `gfx1150` host, and setup builds only host C++ extensions.
After making HIP capability-query failure handling explicit and checking stream
callback-enqueue errors, the three native extensions rebuilt with zero compiler
warnings. `fast_index_copy.cuh` also compiled through ROCm 7.14 `hipcc` for the
`gfx1151` target as a temporary host object; no FreeToken device kernel was
emitted or run. This validates the candidate package/install and host-extension
path, not FreeToken HIP JIT/device-kernel execution on `gfx1151` or end-to-end
parity. The commands below remain experimental until those target-runtime
gates pass.

Do not install into system Python, an existing llama.cpp environment, or the
existing vLLM environment.  The reference layout is intentionally isolated:

```text
/home/david/freetoken-amd/
  source/       this Git checkout
  .venv/        Python 3.12, ROCm PyTorch, AMD Triton, FreeToken
  artifacts/    commands, environment manifests, tests, logs, telemetry
  models/       optional links to read-only local model storage
```

The candidate PyTorch pair is `torch==2.11.0+rocm7.14.0` and
`torchvision==0.26.0+rocm7.14.0` from AMD's multi-architecture wheel index.
The explicit `device-gfx1151` extras select AMD's matching device packages.
The matching SDK, including its development tools and headers, must also be
installed in this same isolated environment. Do not point the build at a
different system ROCm installation: LAN-223 currently has ROCm 10.0 and
7.2.4 trees, which are not the candidate 7.14 SDK.

```bash
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1151]==2.11.0+rocm7.14.0" \
  "torchvision[device-gfx1151]==0.26.0+rocm7.14.0" \
  "rocm[libraries,devel,device-gfx1151]==7.14.0"
rocm-sdk init
export ROCM_HOME="$(rocm-sdk path --root)"
export PATH="$(rocm-sdk path --bin):$PATH"
export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
hipcc --version
python -c "import torch; assert torch.version.hip, torch.version.hip"
```

FreeToken's upstream CUDA package set must not be installed on AMD:
`flashinfer`, `sglang-kernel`, CUDA-indexed Torch wheels, and the CUDA
kernel-cache wheel are NVIDIA binaries. Standard pip does not read uv's source
mapping, so do not expect `pip install "freetoken[rocm]"` to select the ROCm
index automatically.

The initial build command is run from `source` only after the isolated Python
environment has a working HIP PyTorch import:

```bash
python -m pip install -e . --no-build-isolation
```

`--no-build-isolation` is intentional: the validated HIP Torch ABI already
exists in this isolated environment and must be the ABI used for the native
extensions. FreeToken's ordinary runtime dependencies still install, but Torch
is not a base dependency and is therefore not re-resolved. Do not substitute
the CUDA `accel` extra for this sequence.

Maintainers can verify the metadata selection independently with
`bash scripts/verify-accel-resolver.sh`. It creates a disposable directory,
performs resolver dry-runs only, and rejects the invalid ROCm-plus-CUDA-extra
selection; it neither installs packages nor changes a service.

Use `hipcc --version`, `rocminfo`, and a small PyTorch HIP allocation before
the FreeToken build. Record outputs in `artifacts/environment/`, with secrets
and access tokens removed.

## Persistent GGUF HIP JIT cache

The native Gemma GGUF extension is compiled once per combination of FreeToken
source, PyTorch and HIP version, compiler flags, Python ABI, and GPU target.
`torch.utils.cpp_extension` reuses the resulting shared object on later
process starts.  Normal serving must not delete that cache.

The default cache is `$HOME/.cache/torch_extensions/`.  For a deliberate,
portable installation-specific location, set this before every `ft serve`
launch and keep the directory across reboots and service restarts:

```bash
export TORCH_EXTENSIONS_DIR=/home/david/freetoken-amd/cache/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
```

After an intentional FreeToken source or ROCm toolchain update, one rebuild is
expected.  Deleting this directory is a recovery action only.  It was cleared
during the original port investigation to force revised HIP sources to build;
that development step is not part of normal operation.

## Required validation sequence

1. Verify the host's `gfx1151` device, HIP runtime, PyTorch HIP build, and
   AMD Triton version.
2. Build and import `_pinned_tensor` and `_cpu_moe` from the isolated
   environment.
3. Run the ROCm gate unit tests plus the relevant CPU and Triton tests.
4. Run Qwen3.6-35B-A3B through `ft serve` on a non-conflicting local port.
5. Test `/v1/models`, non-streaming `/v1/chat/completions`, and streamed
   `/v1/chat/completions` with fixed requests.
6. Run `ft bench bw` on LAN-223.  Treat its recommendation as a measured
   candidate, then verify it with full serving workloads.
7. Repeat the same API and stability checks for the supported Gemma 4 MoE
   GGUF.
8. Save raw command output, service logs, request responses, profiler output,
   and hardware telemetry under `artifacts/`.

No llama-swap service, model configuration, or existing port is modified by
these commands.  Service packaging happens only after the full validation set
passes.

## Provenance

This branch incorporates the focused current-main ROCm work from FreeToken
pull request #241, preserving its commits and authorship.  It adds explicit
`gfx1151` safety coverage and project-specific validation documentation.
Upstream review should receive a focused pull request containing code plus
tests.  LAN-223 environment reports and benchmark artifacts belong in this
fork unless the upstream maintainers request them.

The completed 2026-08-28 native HIP validation, exact LAN-223 environment,
API evidence, command shapes, and known limitations are documented in
[`lan223-rocm-validation-2026-08-28.md`](lan223-rocm-validation-2026-08-28.md).
