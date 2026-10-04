# FreeToken AMD ROCm on Radeon 8060S `gfx1151`

## Purpose

This branch ports the FreeToken serving runtime to native AMD ROCm and HIP on
the AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`).  The port preserves
the NVIDIA implementation as a separate runtime path.  It does not use Vulkan
or a CPU-only runner as a substitute for native GPU execution.

The intended first deployment target is a `gfx1151` Strix Halo system. It serves the same local API
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
threshold: the target uses a shared-memory APU rather than discrete VRAM and
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
- `kernel/csrc/include/freetoken/hip_compat.h` maps the small CUDA Runtime API subset used by
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

## ROCm 10 system and Python environment

**Qualification status: partial; HIP device execution and parity remain open.**
The qualified target class reports ROCm `10.0.0`, `/opt/rocm` selecting
`/opt/rocm-10.0`, and AMD ROCm packages at release `10.0.0-4`. The validated
Python runtime is PyTorch `2.13.0+rocm10.0.0`; its HIP component reports
`7.15.26333`, which is component metadata inside the ROCm 10 environment, not
a separate installed ROCm 7.x tree. AMD's release history identifies ROCm
10.0.0 as the current production release ([release history](https://rocm.docs.amd.com/en/latest/release/versions.html)).

Use only the pre-provisioned ROCm 10 system stack and a matching isolated HIP
Python environment. Do not install a ROCm 7.x SDK, wheel, index, or runtime
package, and do not use the generic `rocm` package extra to provision the
toolchain. The `rocm` extra is intentionally empty so dependency resolution
cannot silently replace the supported ABI; provision ROCm 10 and its matching
PyTorch/Triton stack before building FreeToken. Keep CUDA-only `flashinfer`,
`sglang-kernel`, CUDA-indexed Torch wheels, and CUDA kernel-cache wheels out of
the HIP environment.

Keep source, Python environment, test artifacts, and models separate under a
caller-selected `${FREETOKEN_WORK_ROOT}`:

- `${FREETOKEN_WORK_ROOT}/source/` — the exact FreeToken checkout under test.
- `${FREETOKEN_WORK_ROOT}/.venv/` — Python and the matching ROCm 10 HIP stack.
- `${FREETOKEN_WORK_ROOT}/artifacts/` — revision, package, test, and runtime evidence.
- `${FREETOKEN_WORK_ROOT}/models/` — optional read-only links to local model storage.

Build from the exact candidate source with build isolation disabled and package
dependencies unchanged; otherwise PEP 517 or a resolver may replace the
pre-provisioned ROCm 10 PyTorch ABI. Record the source fingerprint, `/opt/rocm`
target, ROCm package release, PyTorch/HIP versions, `gfx` target, build output,
and dependency resolution in the artifact manifest. Confirm each target's actual
architecture separately (`gfx1151` and `gfx1150`); do not infer target execution from a host-only build or a masked
device test. See [AMD ROCm installation notes](install_amd.md) for the
pre-provisioned environment boundary and build acceptance checks.

## Persistent GGUF HIP JIT cache

The native Gemma GGUF extension is compiled once per combination of FreeToken
source, PyTorch and HIP version, compiler flags, Python ABI, and GPU target.
`torch.utils.cpp_extension` reuses the resulting shared object on later
process starts.  Normal serving must not delete that cache.

The default cache is `$HOME/.cache/torch_extensions/`.  For a deliberate,
portable installation-specific location, set this before every `ft serve`
launch and keep the directory across reboots and service restarts:

```bash
export FREETOKEN_WORK_ROOT="${FREETOKEN_WORK_ROOT:-$HOME/freetoken-amd}"
export TORCH_EXTENSIONS_DIR="${FREETOKEN_WORK_ROOT}/cache/torch_extensions"
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
6. Run `ft bench bw` on the target system. Treat its recommendation as a measured
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
tests. Environment reports and benchmark artifacts remain private unless
sanitized and explicitly approved for upstream publication.

The completed 2026-08-28 native HIP validation, exact private environment,
API evidence, command shapes, and known limitations are documented in
[`lan223-rocm-validation-2026-08-28.md`](lan223-rocm-validation-2026-08-28.md).
