# DeepSeek-V4.1 KV cache quantization proposal

Status: design record, 2026-09-12. The accepted implementation names the native
mixed mode `fp8-fp4`; see [current behavior and validation](deepseek-v41.md#native-fp8-fp4-kv-storage).
The estimates below predate GPU validation. The experimental NVFP4 window mode
is outside the accepted implementation scope.

Target: one RTX 5090 (32 GiB), the current V4.1 branch, and
`LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4` at revision
`dfce15b92ed1fa76e80e2a46ba847e5b5451f12c`, including image inputs.

## Recommendation

Implement packed storage for the model's existing quantization first: FP8 for
the sliding window, FP4 for compressed attention KV, and MXFP4 for index keys.
The current implementation already rounds these values to those formats, then
stores the reconstructed values in BF16. Keeping the original codes and scales
can avoid additional quantization error relative to this branch's BF16 baseline.
Readers must reconstruct the same BF16 values before the existing arithmetic;
end-to-end equivalence still needs tests.

Add NVFP4 window storage as a separate experimental option after that baseline
works. It introduces an additional numerical change. Its incremental benefit is
small with a small window pool, so it should not delay the first implementation.

This reduces GPU KV memory. It does not shrink the host expert banks or Engram
tables, and requires no model download or checkpoint conversion. Lower memory
traffic may help attention, but decoding packed values also costs work. Overall
generation speed, including CPU expert offload, must be measured.

## Formats and proposed configuration

The accepted CLI value is `fp8-fp4`, distinct from the generic `fp8` and `nvfp4`
layouts. Keep BF16 as the default until validation completes.

| Proposed option | Window | Compressed attention KV | Index keys |
| --- | --- | --- | --- |
| `--kv-cache-dtype bf16` | Existing BF16 storage | Existing BF16 storage | Existing BF16 storage |
| `--kv-cache-dtype fp8-fp4` | Native FP8 packed storage | Native FP4 packed storage | Native MXFP4 packed storage |
| `--kv-cache-dtype nvfp4` (future only) | Additional NVFP4 quantization | Same native FP4 storage | Same native MXFP4 storage |

Report all three resolved formats and their bytes in startup diagnostics.

| Tier | Width | Codes and scale representation | Bytes/row, including scales |
| --- | ---: | --- | ---: |
| BF16 window / compressed KV | 512 | BF16 values | 1,024 |
| BF16 index keys | 128 | BF16 values | 256 |
| Native window | 512 | E4M3 codes; one UE8M0 scale per 32 values | 528 |
| Native compressed KV | 512 | Packed E2M1 codes; one E4M3 scale per 16 values; implicit global scale 1 | 288 |
| Native index keys | 128 | Packed E2M1 codes; one UE8M0 scale per 32 values | 68 |
| Experimental NVFP4 window | 512 | Packed E2M1 codes; E4M3 scale per 16 values; FP32 scale per row | 292 |

V4.1 uses a shared K/V latent; do not multiply these figures by two for K and V.
There are 40 window pools but only four compressed-KV/index-key owners, at layers
2, 8, 14 and 20, with compression ratios 2, 2, 2 and 1. Eight layers compute index
queries, which does not require eight index-key pools.

The fixed checkpoint's [reference model](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4/blob/dfce15b92ed1fa76e80e2a46ba847e5b5451f12c/inference/model.py)
uses these native formats: window at line 707, compressed KV at line 760, and
index queries/keys at lines 546/552. Its
[reference quantization kernel](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4/blob/dfce15b92ed1fa76e80e2a46ba847e5b5451f12c/inference/kernel.py)
also exposes packed FP4 output. These files were inspected, not executed.

Uniform FP8 storage is less attractive: it uses more bytes for already-FP4 tiers
and can round compressed values again. For example, an E2M1 value of 1.5 times an
E4M3 scale of 1.125 reconstructs to BF16 1.6875, which E4M3 cannot represent
exactly. Preserve each tier's native representation instead.

## Memory estimates

These totals cover pool-owned tensors, including scales, compression state,
scratch rows and the full-to-window map. They assume 128-token pages, two running
requests, three scratch rows per source, and one extra dummy full page. K and M
mean 1,024 and 1,048,576 tokens. Capacity is shared across requests, not available
independently to each of the two requests.

| Shared KV capacity | Window pool ratio | Current BF16, GiB | Native mixed storage, GiB | NVFP4 window plus native compressed/index, GiB |
| --- | ---: | ---: | ---: | ---: |
| 32K | 0.20 | 0.353405 | 0.159638 | 0.101120 |
| 128K | 0.20 | 1.397678 | 0.630562 | 0.399868 |
| 1M | 0.02 | 3.937756 | 1.293732 | 1.109177 |
| 1M | 0.20 | 11.173664 | 5.041100 | 3.196675 |

For 1M with a 0.02 window pool ratio, native storage saves about 2.64 GiB;
quantizing the window further saves another 0.18 GiB. At the current nominal
32K/0.20 setting, native storage saves about 198 MiB. The live automatic planner
can round to a different page count, so these are not exact live-process totals.

The window pool ratio controls retained window pages, not the model's 128-token
attention window. Reducing this ratio is a separate cache-retention decision.
Neither these estimates nor the model's context limit establishes usable 1M
throughput on this machine.

For reproduction, let `T` be usable shared token capacity, `F = T + 128`,
`w = ceil(r * (T / 128 + 1))`, and `C = 2.5 * F`. For the configurations above,
the window working-set floor is below `w`. With row byte sizes `bw`, `bc`, `bi`:

```text
window bytes     = 40 * 128 * w * bw
compressed bytes = (C + 12) * bc
index bytes      = (C + 12) * bi
FP32 state bytes = 3 * (2 * w + 1) * 1024 * 4
map bytes        = (F + 1) * 8
```

The BF16 totals were checked against the current pure-Python cost functions and
an independent shape calculation; all four cases matched exactly. Packed totals
use the same shapes with the proposed row sizes. Alignment or padding added by
the implementation must be reflected in the estimates.

Page tables, free lists and fixed position arrays add approximately 0.38, 1.51,
12.04 and 12.05 MiB respectively. At 1M, a two-request eager decode snapshot adds
16 MiB, with another 8 MiB temporary index-selection tensor during its creation.
Query/selection/attention workspaces, prefix-cache metadata and CUDA allocator
headroom are additional and workload dependent. Freed capacity can be used for
longer contexts or more GPU expert caching; it is not automatically all spare
VRAM if the planner reallocates it.

## Implementation sequence

### 1. Preserve quantization results and expose a storage layout

Extend `kernel/triton/dsv41/quant.py` with packed encode/decode helpers and a CPU
reference. Capture codes and scales where `_project`, `Indexer.keys` and
`_publish` currently perform roundtrips in `models/deepseek_v41/attention.py`.
Do not derive new scales from already reconstructed BF16 values in native mode.
Keep query quantization and RoPE ordering unchanged, including the existing
quantization of the RoPE channels.

Keep a logical BF16 computation dtype and an explicit storage layout. Each packed
uint8 row contains its code bytes followed by its scale bytes. The inherited BF16 dtype assertion cannot simply be
changed to FP8: the three tiers have different storage types and strides. Centralize
row-size and scale-shape definitions so allocation and byte accounting agree.

### 2. Allocate and write every tier consistently

Extend `kvcache/dsv41_paged_pool.py` to allocate the packed code/scale rows for
each owner, including dummy and request-specific scratch rows. Keep the unfinished
compression state rings in FP32; they hold reduction inputs and scores across
partial pairs, not finished quantized KV values.

Override the V4.1 backend's compressed write route. The inherited
`attention/dsv4_compress.py::scatter_compressed` directly calls
`index_copy_(..., kv.to(pool.dtype))`; changing only a pool writer would miss this
path and cast values to integer bytes instead of encoding them. Route window,
compressed attention and index-key writes through the appropriate codec in both
prefill and decode. Preserve the current page addressing and request-isolated
scratch destinations.

### 3. Decode only selected tiles inside attention and the indexer

Add V4.1-specific readers through `attention/dsv41_sparse.py`. The current shared
`kernel/triton/dsv4/sparse_attn.py` assumes identical BF16 window/compressed
strides. Both normal and split-K attention need mixed-format readers. Keep the
V4 BF16 path unchanged; factor shared pieces only when their behavior is identical.

Reconstruct a selected tile to the same BF16 values as the old roundtrip, then
use the existing FP32 arithmetic. In particular, multiplying FP4 codes and scales
directly into FP32 attention without the intermediate BF16 rounding changes the
baseline. Do not simultaneously switch to FP8/FP4 tensor-core attention.

Preserve the single-KV-tile staging design: the existing kernel documents an RTX
5090 shared-memory constraint. Loading two complete window/compressed tiles and
selecting afterwards can exceed it. Inspect compiled shared memory, registers and
temporary allocations on sm120. Generic NVFP4 readers assume scalar base pointers;
mixed per-column pool selection requires an adapted reader.

In `kernel/triton/dsv41/indexer.py`, decode MXFP4 keys within the existing bounded
tiles. Preserve score rounding, candidate reuse and deterministic tie ordering.
Verify selected IDs, not just score closeness. Do not create a full-context BF16
key copy, which would erase the memory benefit at long contexts.

### 4. Complete accounting, cache lifecycle and configuration gates

Update exact bytes, unit bytes, automatic budget estimates and the page solver in
`kvcache/dsv41_cost_model.py` using the same layout definitions as allocation.
Include every scale byte in reported bytes and rebuild validation. Keeping scales
inside the original slabs lets inherited cleanup free codes and scales together;
preserve the existing idle-only resize/rebind sequence.

Codes and scales must share physical row identity through prefix reuse, window
eviction, page reuse, dummy accesses and request cancellation. Reinitialize safe
dummy values and do not permit stale scales to accompany new codes. Preserve
graph-safe writes and recapture behavior wherever the baseline supports graphs.

Only after writers, readers and accounting pass tests, enable V4.1's matching
capability flags and pool checks in `attention/__init__.py`, `engine/engine.py`
and `kvcache/__init__.py`. Keep rejection for other unsupported architectures.
Update CLI help and `docs/deepseek-v41.md` with the actual mixed-tier semantics.

### 5. Add the experimental NVFP4 window option

Initially encode the existing FP8-roundtrip BF16 window values into NVFP4, so this
option is explicitly one additional compression step relative to native mode.
Use a separate FP32 scale per stored row; a changing cache-wide scale would
invalidate previously stored prefixes. Existing `kernel/triton/kv_nvfp4.py`
provides useful packing and row-scale patterns. Its quantizer must not replace
the native compressed-KV codec, whose global scale is implicitly one.

NVIDIA describes NVFP4 as E2M1 data with 16-value E4M3 block scales and a second
FP32 scale in its [format explanation](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
As checked on 2026-09-12, TensorRT-LLM's
[hardware matrix](https://nvidia.github.io/TensorRT-LLM/features/quantization.html#hardware-support-matrix)
does not list NVFP4 KV support for sm120, though it lists it for sm100/103.
This proposal therefore uses FreeToken's own packed storage and tile decoding;
it does not assume a TensorRT-LLM attention kernel is available on the RTX 5090.

## Validation and acceptance

Extend existing tests rather than adding a parallel test framework:

- `tests/kernels/test_dsv41_quant.py`: independent code/scale/reference checks,
  zero and tiny values, outliers, rounding boundaries and FP4 tie cases; native
  decode must reproduce the current BF16 roundtrip exactly.
- `tests/kernels/test_dsv41_indexer.py`: score precision and exact selected IDs,
  ties, candidate restriction, source sharing and long-key tile boundaries.
- `tests/models/test_dsv41_attention.py`: identical-mode BF16 comparisons for
  normal and split-K attention, sinks, all-masked selections, ratio 0/1/2,
  odd chunk boundaries, 128-token page boundaries and two-request decode.
- `tests/kvcache/test_dsv41_pool.py`: allocation bytes equal cost estimates,
  solver maximal fit, fixed-window overrides, scratch/dummy rows, page reuse,
  prefix reuse, eviction and rebuild replacing all code/scale pointers.
- `tests/engine/test_kv_quant_config.py` and
  `tests/engine/test_attention_backend_matrix.py`: accept completed V4.1 profiles
  and retain rejection for unsupported pool/backend combinations. Cover graph
  replay and resize recapture where supported by the baseline.
- RTX 5090 integration: text, Japanese/code prompts, image prompts and mixed
  multi-turn requests. Check independent requests cannot reuse each other's
  scale or unfinished compression state. Confirm no full-context BF16 expansion.

Benchmark the same pinned checkpoint, prompts, context lengths and batch sizes
against this branch's pre-change BF16 implementation. Record TTFT, prefill rate,
decode tokens/s, actual KV allocation, peak VRAM and temporary memory. Separate a
fixed expert-cache comparison from a planner-retuned comparison that spends the
saved KV bytes on experts. Warm kernels and Engram pages consistently.

Repository policy requests performance comparisons against `main`; if `main`
cannot run this V4.1 checkpoint, report that limitation explicitly rather than
substitute a different model as an equivalent baseline. Test shared-kernel
regressions on supported models if shared code changes.

Native mode must first pass exact codec reconstruction and index-selection tests;
then check model-level differences with matching launch modes and documented
floating-point tolerances. NVFP4 window mode additionally needs teacher-forced
logit/perplexity comparisons and long-context retrieval, Japanese/code and image
quality evaluation. Establish acceptance thresholds before promoting it. Short
arithmetic and image-caption smoke tests alone are insufficient for that decision.

At the time of the initial proposal, no GPU quantized-KV tests, quality evaluations
or speed benchmarks had been run. Current results belong in `deepseek-v41.md`.
The first deliverable should be validated native mixed storage;
the further NVFP4 window compression remains opt-in until its benefit and quality
are measured.
