# LAN-223 native ROCm validation, 2026-08-28

## Result

This validation passed the first release gate for the AMD port.  FreeToken
served both required MoE models through the OpenAI-compatible API on LAN-223's
Radeon 8060S (`gfx1151`) using a native HIP and ROCm execution path.

This is not a CPU fallback or a Vulkan result.  The serving process uses the
ROCm PyTorch wheel, HIP-compiled native extensions, and Triton GPU kernels.
CUDA graphs were deliberately disabled for this validation because the MVP
needs correctness before graph capture tuning.

## Reproducibility record

| Item | Value |
| --- | --- |
| Host | LAN-223, `david-Gmktec-x2-2` |
| GPU | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| System ROCm installation | ROCm 10.0 at `/opt/rocm-10.0` |
| PyTorch wheel | `2.13.0+rocm10.0.0` |
| HIP reported by PyTorch | `7.15.26333` |
| FreeToken branch | `amd-rocm-gfx1151` |
| Validation commit | `065d806` |
| API exposure | loopback-only ports, not llama-swap |

The isolated validation layout was `/home/david/freetoken-amd/`; no existing
llama-swap service, model configuration, or production endpoint was changed.

## Models and API evidence

| Model | Source revision | Backend selection | Non-streaming result | Streaming result |
| --- | --- | --- | --- | --- |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | vendor model snapshot used for this run | Triton attention, MoE offload, native Triton NVFP4, serial expert load | HTTP 200, `AMD ROCm FreeToken ready.` in 1.54 s | HTTP 200, SSE chunks and `[DONE]` |
| `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` | `d1c082be9cf3c8a514acf63b8761f4b41935842e` | Triton attention, MoE offload, serial expert load, HIP GGUF JIT | HTTP 200, `native hip api works` in 341.304 ms | HTTP 200, SSE chunks and `[DONE]` |

Raw evidence remains on LAN-223 in these isolated artifact directories:

```text
/home/david/freetoken-amd/artifacts/qwen36-nvfp4-serial-hip-prefill/
/home/david/freetoken-amd/artifacts/gemma4-q4-rocm-thrust-system/
```

The Gemma telemetry captured immediately after the API tests identified the
same `gfx1151` device, 33 percent GPU utilization, 46 percent allocated VRAM,
and a 40 C edge temperature.  The model uses the APU's shared-memory design;
the tool's VRAM label is therefore only its standard telemetry label.

## Warm single-request throughput

The following measurements use one fixed 733-token prompt, greedy sampling,
and a one-sentence answer that produced 26 completion tokens.  `TTFT` is the
client-observed time to the first non-empty SSE text chunk.  Prompt throughput
is the end-to-end prompt-token count divided by TTFT, so it includes normal
API and scheduler overhead.  Output throughput is completion tokens divided
by the interval from that first chunk through `[DONE]`.

| Model | Prompt tokens | Completion tokens | TTFT | Prompt TPS | Generation interval | Output TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B NVFP4 | 733 | 26 | 4.976 s | 147.3 | 0.899 s | 28.9 |
| Gemma 4 26B A4B Q4_0 GGUF | 733 | 26 | 3.244 s | 226.0 | 0.581 s | 44.8 |

These are warm, single-request measurements, not concurrency or maximum
throughput claims.  The Qwen configuration uses the native Triton serial
NVFP4 prefill route selected for ROCm correctness.  Its approximately
seven-minute cold initialization is expert-bank preparation and cache
allocation, not inference time.

## Same-model llama.cpp Vulkan comparison

To compare the usable Strix Halo serving baseline rather than an unrelated
model, the exact Gemma GGUF was served by llama.cpp Vulkan build `b10141`
(`0d47ea742`) on a separate loopback port.  Both servers used one slot,
8,192-token context, greedy sampling, and the same repeated scheduler prompt.
The model SHA-256 was
`3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.

| Runtime | GPU backend | Prompt tokens | Completion tokens | TTFT | Client prompt TPS | Client output TPS | Runtime prompt TPS | Runtime output TPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FreeToken | ROCm/HIP | 733 | 26 | 3.244 s | 226.0 | 44.8 | not exposed | not exposed |
| llama.cpp `b10141` | Vulkan | 758, 7 template tokens cached | 128 | 0.855 s | 886.7 | 63.2 | 1,078.4 | 61.7 |

For this isolated, single-request Gemma workload, llama.cpp Vulkan reached
first output about 3.8 times sooner, delivered about 3.9 times the
client-observed prompt rate, and delivered about 1.4 times the client-observed
generation rate.  llama.cpp's internal timing excludes ordinary API and
scheduler overhead, so its 1,078.4 prompt TPS and 61.7 output TPS must not be
compared directly with FreeToken's client-observed rates.

The completion lengths differ because llama.cpp exposed Gemma's reasoning
stream and consumed the 128-token cap, whereas FreeToken's parser emitted the
final concise answer and stopped at 26 tokens.  That makes the output-rate
comparison useful as a warm streaming rate, but not a quality or exact
end-to-end task comparison.  The raw llama.cpp evidence is retained under
`/home/david/freetoken-amd/artifacts/llamacpp-vulkan-gemma4-q4-tps/` on
LAN-223.

## Same-model ROCm 10 and HIP comparison

The Vulkan baseline above answers a practical deployment question, but it is
not a backend-for-backend comparison.  This follow-up rebuilt the same
llama.cpp source revision, `b10141` (`0d47ea742`), with HIP for `gfx1151` and
ran it under the same ROCm 10 installation used by FreeToken.  The compiler
was ROCm 10 HIP `7.15.26333` with AMD Clang 23.0.0.  At runtime, llama.cpp's
`libamdhip64`, `libhipblas`, `librocblas`, `libamd_comgr`, and HSA runtime
libraries all resolved from `/opt/rocm-10.0`, not the older ROCm installation.

Both runners used the identical 14 GB Gemma 4 26B A4B Q4_0 GGUF, SHA-256
`3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`, one
request at a time, an 8,192-token context, greedy sampling, `max_tokens: 128`,
and a 48-times repeated scheduler prompt.  Each measurement used a distinct
nonce, preventing prompt-cache reuse.  The token totals differ by one because
the two runners tokenize and render Gemma's chat template differently.

| Runtime | HIP and ROCm stack | Prompt tokens | Completion tokens | TTFT | Client prompt TPS | Client output TPS | Runtime prompt TPS | Runtime output TPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FreeToken, steady state | PyTorch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, native HIP GGUF extension | 772 | 20 | 2.863 s | 269.6 | 46.1 | not exposed | not exposed |
| llama.cpp `b10141` | ROCm 10 HIP, `gfx1151` | 771 | 128 | 0.850 s | 906.6 | 58.3 | 1,011.6 | 56.2 |

On this uncached, single-request workload, llama.cpp ROCm 10 reached first
text about 3.4 times sooner, supplied about 3.4 times the client-observed
prompt rate, and supplied about 1.3 times the client-observed output rate.
llama.cpp's internal numbers exclude HTTP, SSE, and scheduling overhead and
therefore are only comparable to another internal timing source, not directly
to FreeToken's client values.

The FreeToken request that triggered a fresh GGUF HIP extension build is kept
as a separate cold-start measurement: 768 prompt tokens, 21 completion tokens,
109.938 s TTFT, 6.99 client prompt TPS, and 27.47 client output TPS.  It
contains HIP compilation and must not be presented as inference throughput.
The subsequent steady-state run above was made after the extension completed,
using a fresh nonce and no prompt cache hit.  FreeToken's extension compiler
was `/opt/rocm-10.0/bin/hipcc` targeting `gfx1151`, and its runtime libraries
came from the ROCm 10 PyTorch SDK packages.  Its existing JIT command also
passed `/opt/rocm-7.2.4/include` as a supplemental include path.  That does not
change the ROCm 10 compiler or loaded runtime libraries, but it prevents this
FreeToken build from being described as a strictly ROCm 10-only header build.

The llama.cpp response used all 128 allowed tokens because it exposed Gemma
reasoning text.  FreeToken stopped after a concise 20-token answer.  This
makes the output-rate comparison a useful streaming measurement, but it is
not an exact answer-quality or equal-completion-length evaluation.

Raw artifacts are retained only on LAN-223:

```text
/home/david/freetoken-amd/artifacts/llamacpp-rocm10-gemma4-q4-tps/
/home/david/freetoken-amd/artifacts/freetoken-rocm10-gemma4-q4-tps/
```

## AMD TPS optimization campaign

The first configuration optimization pass used the same warm AIME-25 problem
and a 128-token greedy completion for both FreeToken and the ROCm 10 HIP build
of llama.cpp `b10141`.  Each runner received the identical user message, used
a warm identical request before the measured request, and ran one stream at a
time.  Both rendered 63 prompt tokens; FreeToken's measured request reused 62
prompt tokens and llama.cpp's reused 58.

| Runtime and candidate | Decode TPS | TTFT | Result |
| --- | ---: | ---: | --- |
| FreeToken, offload, eager | 54.89 | 267.9 ms | Baseline |
| FreeToken, offload, HIP graph capture at batch size 1 | 55.73 | 259.3 ms | Best observed safe configuration |
| FreeToken, HIP graph plus experimental `-ffast-math` GGUF extension | 55.65 | 261.6 ms | Rejected: no gain, despite matching output hash |
| FreeToken, final target-specific `gfx1151` GGUF extension plus graph capture | 55.44 | 263.6 ms | Validated shipping configuration; normal run-to-run variation |
| llama.cpp `b10141`, ROCm 10 HIP | 60.42 client, 58.88 internal | 128.6 ms | Matched reference |

The graph configuration removes approximately 1.5 percent of the eager decode
cost, but FreeToken still trails llama.cpp by 7.8 percent using client TPS and
by approximately 5.4 percent compared with llama.cpp's internal decode timing.
The requested criterion of meeting or exceeding llama.cpp is therefore **not
met** by the first configuration pass.

The best verified FreeToken command shape is:

```bash
export ROCM_PATH=/opt/rocm-10.0
export HIP_PATH=/opt/rocm-10.0
export TORCH_EXTENSIONS_DIR=/home/david/freetoken-amd/cache/torch_extensions

ft serve --model-path /home/david/freetoken-amd/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
  --attention-backend triton --moe-backend offload --moe-cache-auto \
  --memory-ratio 0.50 --max-running-requests 1 --max-seq-len-override 8320 \
  --cuda-graph-max-bs 1
```

The port now derives and exports `PYTORCH_ROCM_ARCH=gfx1151` before the GGUF
extension is compiled when the operator did not set an explicit architecture.
This avoids compiling for unnecessary visible targets and makes the extension
cache target-specific.  It does not itself increase steady-state TPS because
the original HIP build already selected `gfx1151` on this single-GPU host.

The remaining gap is not an untested cache or residency setting: Gemma's GGUF
adapter only supports the native Q4_0 offload implementation, and the automatic
cache selected all 3,840 routed-expert slots.  Closing the gap requires a
profile-guided improvement to the HIP GGUF decode kernels or another proven
ROCm attention or quantized-linear implementation.  The available ROCm 10
`rocprofv3` installation could not yet provide that kernel breakdown: attach
mode reports that the PyTorch process has no `rocp-bg-attach` registration
thread even when launched with `ROCP_TOOL_ATTACH=1`, while launch mode aborts
before FreeToken starts with LLVM's duplicate `spirv-expand-step` option.  The
full error evidence is retained in `rocprof-gfx1151*/` and
`rocprof-launch-gfx1151-v2/` under the raw artifact directory.  This is a
toolchain issue, not a FreeToken performance result, so no profiler-derived
optimization claim is made here.  A temporary high-performance DPM governor
test could not be run because the non-root LAN-223 account cannot write
`power_dpm_force_performance_level`; automatic mode was unchanged.

Raw campaign artifacts are retained on LAN-223:

```text
/home/david/freetoken-amd/artifacts/amd-optimization-2026-08-28/
```

## GGUF extension reuse validation

The first Gemma request after the original source change built the native HIP
GGUF extension.  A subsequent complete server restart retained the existing
Torch extension cache.  Its first API request returned HTTP 200 and Ninja
reported `no work to do`, proving the compiled shared module was reused.
Torch still runs a lightweight hipify and dependency check before loading the
cached module; it did not run `hipcc` compilation or shared-library linking.
See the persistent-cache operating procedure in
[`amd-rocm-gfx1151.md`](amd-rocm-gfx1151.md#persistent-gguf-hip-jit-cache).

## Commands used

Qwen was started in the isolated environment with this functional shape:

```bash
ft serve --model-path /home/david/freetoken-amd/models/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen3.6-35b-a3b-nvfp4-amd --host 127.0.0.1 --port 18501 \
  --attention-backend triton --moe-backend offload --nvfp4-backend triton \
  --expert-load serial --moe-cache-auto --memory-ratio 0.35 \
  --max-seq-len-override 8192 --kv-reserve-tokens 2048 \
  --cuda-graph-max-bs 0 --disable-pynccl --disable-moe-prefill-overlap
```

Gemma used the native GGUF model file and its own loopback port:

```bash
ft serve --model-path /home/david/freetoken-amd/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
  --served-model-name gemma-4-26b-a4b-q4-amd --host 127.0.0.1 --port 18502 \
  --attention-backend triton --moe-backend offload --expert-load serial \
  --moe-cache-auto --memory-ratio 0.50 --max-seq-len-override 8192 \
  --kv-reserve-tokens 2048 --cuda-graph-max-bs 0 --disable-pynccl
```

The API checks used `/v1/models` and `/v1/chat/completions`, both with normal
JSON responses and with `stream: true`.  The front-end port can answer before
the worker finishes loading, so the successful tests waited for the server log
line `API server is ready to serve` before submitting requests.

## AMD-specific corrections verified here

1. ROCm detection is explicit, preventing `gfx1151` from being treated as an
   NVIDIA SM 11.5 capability.
2. CUDA-only optional backends are not selected on HIP.
3. DLPack and fast indexed-copy tensor handling accepts HIP tensors.
4. HIP avoids the unsafe grouped NVFP4 prefill kernel and uses the native
   Triton serial expert implementation instead.  This trades prompt prefill
   speed for correctness on the current Strix Halo stack.
5. The Gemma GGUF JIT discovers a system Thrust include directory when the
   PyTorch wheel omits Thrust.  It passes that path as a compiler system
   include, avoiding an attempted hipify write into the ROCm installation.
6. The same JIT adds a system ROCm library directory only when the wheel SDK
   lacks the unversioned `libamdhip64.so` linker name.  On LAN-223 this allowed
   the native `gfx1151` object and shared module to compile and link.

## Known limitations and follow-up work

- This is a functional API validation, not a performance benchmark.  The
  recorded request timings include the chosen small fixed requests and are not
  tokens-per-second claims.
- CUDA graph capture remains disabled for the HIP MVP.
- Qwen's HIP prefill deliberately uses the safe serial Triton route instead of
  the grouped NVFP4 prefill route that produced an HSA aperture violation on
  this machine.
- The first Gemma request compiles its GGUF HIP extension and has a substantial
  cold-start cost.  Later requests use the cached module.
- llama-swap integration is intentionally outside this release gate.

## Local checks completed

```bash
python -m compileall -q python
git diff --check
```

The port's HIP gate tests are retained under `tests/utils/test_rocm_runtime.py`.
The live end-to-end checks above are the required full-model validation for
this change.
