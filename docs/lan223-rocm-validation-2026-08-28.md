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
