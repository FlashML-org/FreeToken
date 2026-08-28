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

Do not install into system Python, an existing llama.cpp environment, or the
existing vLLM environment.  The reference layout is intentionally isolated:

```text
/home/david/freetoken-amd/
  source/       this Git checkout
  .venv/        Python 3.12, ROCm PyTorch, AMD Triton, FreeToken
  artifacts/    commands, environment manifests, tests, logs, telemetry
  models/       optional links to read-only local model storage
```

The exact PyTorch ROCm wheel must be selected after validating its compatible
Triton build on LAN-223.  FreeToken's upstream CUDA package set must not be
installed on AMD: `flashinfer`, `sglang-kernel`, CUDA-indexed Torch wheels, and
the CUDA kernel-cache wheel are NVIDIA binaries.

The initial build command is run from `source` only after the isolated Python
environment has a working HIP PyTorch import:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

Use `hipcc --version`, `rocminfo`, and a small PyTorch HIP allocation before
the FreeToken build.  Record outputs in `artifacts/environment/`, with secrets
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
