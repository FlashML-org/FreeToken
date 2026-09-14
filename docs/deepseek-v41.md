# DeepSeek-V4.1 Flash NVFP4

This branch adds experimental text and image inference for
[`LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4),
validated at revision `dfce15b92ed1fa76e80e2a46ba847e5b5451f12c`.
The earlier `s-zaizen/DeepSeek-V4.1-Flash-NVFP4` checkpoint with FP8 Engram
tables remains supported.
The implementation uses the checkpoint's native names and shapes. It does not
instantiate model code downloaded from the Hub.

## Requirements on one RTX 5090

The launch example targets one 32 GiB RTX 5090 and approximately 480 GiB of WSL
memory on a 512 GiB host. The LibertAIDAI checkpoint occupies about 429 GB
(400 GiB). Routed expert banks require 285.04 GiB of host memory; the 97.28 GiB
Engram tables remain packed on disk and in the OS page cache. Only requested
rows are decoded. Tables are required for text as well as images, and are never
fully pinned or expanded to BF16.

With every Engram page cached, expert banks plus tables total 382.32 GiB,
leaving about 98 GiB of the WSL memory allocation for runtime overhead and other
uses. The earlier FP8 Engram tables required 188.83 GiB; their FP4 replacements
save 91.56 GiB. GPU-resident weight geometry is unchanged. These are byte-count
estimates, not a guarantee for every context length or concurrent workload.

The publisher describes the expert transcode as lossless but the Engram FP8 to
FP4 conversion as lossy, and has not published end-to-end quality evaluations.
Successful smoke tests cannot establish equivalent answer quality.

Allow SSD space for the approximately 429 GB checkpoint and runtime caches, plus
another approximately 429 GB if creating an FTW copy. Download the revision above
with a Hugging Face client, then pass the completed local snapshot directory to
`--model` in the launch command below.

The launch example uses a 32,768-token context, 1,024-token prefill chunks and two
concurrent requests. Adjust `--max-seq-len-override`, `--kv-reserve-tokens`,
`--max-prefill-length` and `--max-running-requests` for your workload. Increasing
the context limit alone does not reserve the corresponding KV capacity.
Full-checkpoint long-context throughput and memory limits still need measurement.

## Direct launch

After [installing FreeToken](install.md), run this command with a downloaded
snapshot or standalone FTW directory. The current default streams vision blocks
from host memory (`--mm-encoder-weights host`). Historical validation below used
resident vision weights, so its GPU allocation measurements do not describe this
default placement:

```bash
ft serve --model /path/to/snapshot \
  --served-model-name LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4 \
  --moe-strategy offload --quant-backend moe.nvfp4=triton --expert-load parallel \
  --moe-cache-auto --disable-moe-prefill-overlap \
  --cuda-graph-max-bs 0 --kv-cache-dtype fp8-fp4 \
  --max-seq-len-override 32768 --kv-reserve-tokens 32768 \
  --max-prefill-length 1024 --max-running-requests 2 \
  --swa-full-tokens-ratio 0.2 \
  --host 0.0.0.0 --port 1919
```

On the validation host, the launch configuration explicitly loaded expert banks in
parallel. The expert reader excludes the two large Engram shards; its banks and
temporary shard buffers fit within this host's RAM. The generic automatic check
counts the whole checkpoint and would choose serial loading. Use
`--expert-load serial` for a lower-memory host. Prefill overlap is disabled so the
GPU cache does not require two complete expert layers, and the planner sizes the
cache from measured free GPU memory.

The engine automatically selects `dsv41_sparse` attention. WSL's existing CPU
expert fallback remains available when its CUDA pinned-memory quota cannot hold
the expert banks. CPU and hybrid execution use the same NVFP4 expert values as
the offload strategy. The Triton NVFP4 expert kernel supports this model's clamped
SwiGLU; `moe.nvfp4=marlin` and `moe.nvfp4=b12x` are rejected.

## Sizing a million-token context

The following configuration is sized from tensor shapes and covered by budget
tests. It has not been validated with a full million-token checkpoint run.

```bash
ft serve --model /path/to/snapshot \
  --served-model-name LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4 \
  --moe-strategy offload --quant-backend moe.nvfp4=triton --moe-cache-auto \
  --disable-moe-prefill-overlap \
  --cuda-graph-max-bs 0 --kv-cache-dtype bf16 \
  --max-seq-len-override 1048576 --kv-reserve-tokens 1048576 \
  --swa-full-tokens-ratio 0.02 \
  --max-prefill-length 1024 --max-running-requests 2
```

Historical validation with GPU-resident vision weights measured 11.35 GiB of
resident model parameters. The current default streams vision blocks from host
memory through two GPU buffers, so its resident allocation differs. The BF16 KV
pool for 1,048,576 tokens requires 3.94 GiB at a window ratio of 0.02; the default 0.2
ratio requires 11.17 GiB. A ratio of 0.01 reduces KV storage to 3.54 GiB. These
ratios control retained window pages and prefix reuse, while the model's attention
window remains 128 tokens.

One layer of 384 cached experts requires 7.13 GiB. Disabling prefill overlap permits
this minimum; the automatic planner can assign additional expert slots from the
measured free memory. With the historical resident-vision layout and ratio 0.02,
the minimum persistent model, expert and KV allocation is 22.42 GiB, plus about
24 MiB of GPU Engram staging for a 1,024-token chunk. Activations, CUDA workspaces
and allocator overhead need additional memory.
The two-buffer overlap path requires at least 768 expert slots and may exceed
the default memory budget at this context length.

Chunked prefill keeps Engram staging proportional to the chunk size. Increasing
the context limit does not allocate a million rows of Engram staging. The host
expert banks remain about 285 GiB; Engram tables occupy about 97 GiB on disk and
are read through the reclaimable OS page cache. All figures describe allocation
geometry, not measured full-model throughput or answer quality.

## Image requests

OpenAI chat accepts ordered `text` and `image_url` parts, Anthropic messages accept
base64 or URL `image` blocks, and Responses accepts `input_image.image_url`.
Use a base64 image data URL for local files; public HTTP(S) image URLs are also
supported. Uploaded Responses file IDs are not supported.

```python
import base64
from pathlib import Path
from openai import OpenAI

image = base64.b64encode(Path("example.png").read_bytes()).decode()
client = OpenAI(base_url="http://localhost:1919/v1", api_key="local")
response = client.chat.completions.create(
    model="LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
    ]}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

Image spans include start, newline and end tokens and may cross prefill chunk
boundaries. The shared multimodal path hashes the processed patches, grid and
span layout into image-specific placeholder IDs, allowing matching image
prefixes to reuse the prefix cache. Repeated images share encoded embeddings
while those embeddings are needed by active requests; a complete prefix hit can
skip image encoding. Legacy `media` and precomputed `mm_embeds` requests remain
isolated from prefix reuse. Current input limits are 16 images per request,
32 MiB per image and 64 megapixels per image.

The native chat encoder also preserves ordered text and images in Anthropic
`tool_result` blocks and Responses `function_call_output`. Image support is
enabled by default; `--text-model-only` or `--mm-disable vision` disables the
tower and rejects image inputs. Vision block weights use host streaming by
default; select `--mm-encoder-weights gpu` to keep them resident.

`--image-max-tokens` limits the entire image span, including start, row-newline
and end tokens. `--image-min-tokens` is rejected; use
`--mm-processor-kwargs '{"vision_min_pixels": 295936}'` to set the minimum pixel
area before the maximum-token resize. The processor also accepts
`vision_max_n_token` and `vision_max_wh_ratio`; `vision_max_n_token` overrides
`--image-max-tokens` when both are supplied.

## Native FP8-FP4 KV storage

Use `--kv-cache-dtype fp8-fp4` to keep the model's existing quantized KV values
packed in GPU memory. This mode uses `dsv41_sparse` attention with BF16 compute;
`--attention-backend auto` selects it. Generic `fp8` and `nvfp4` KV modes remain
unsupported for V4.1 because their layouts differ.

The launch example selects `fp8-fp4` explicitly. Use `--kv-cache-dtype bf16` to
select the previous representation, then restart the server. The CLI default
remains `bf16`.

| Stored tier | Native format | Bytes per row, including scales | BF16 bytes per row |
| --- | --- | ---: | ---: |
| Window, 512 values | E4M3 + UE8M0 scale per 32 values | 528 | 1,024 |
| Compressed KV, 512 values | E2M1 + E4M3 scale per 16 values | 288 | 1,024 |
| Index keys, 128 values | E2M1 + UE8M0 scale per 32 values | 68 | 256 |

Each physical row contains codes followed by scale bytes, so page reuse and
cache rebuilds move them together. The writer keeps codes from the existing
quantization step, without recalculating scales from reconstructed BF16 values.
Attention and indexer kernels restore selected tiles to BF16 at the original
rounding point. Unfinished compression state remains FP32. The full cache is
never expanded back to BF16 for attention.

This mode avoids additional storage quantization relative to this branch's BF16
baseline. It does not remove the model's existing quantization or establish
equivalence to a differently quantized checkpoint. Model weights, Engram tables
and host RAM requirements are unchanged; no new download is required.

For a nominal shared 32K pool at window ratio 0.2 and two concurrent requests,
the pool-owned tensors decrease from 379,465,736 to 171,409,848 bytes. A shared
1M pool at ratio 0.02 decreases from 4,228,132,872 to 1,389,134,264 bytes. These
include scales, pending state, scratch rows and the full-to-window map; query
workspaces and allocator overhead are additional. The automatic planner may
choose different page counts or spend the savings on more cached experts.
Long-context throughput still needs separate measurement.

The mode is selected at startup. Changing between `bf16` and `fp8-fp4` requires
a server restart; resizing an idle cache preserves its storage format. Further
NVFP4 quantization of the window is not part of this implementation.

### Packed KV validation on RTX 5090

On 2026-09-12 JST, 319 focused tests passed in the CUDA 13 development container
with PyTorch 2.11.0+cu130, Triton 3.6.0 and the workspace sources installed.
They covered codecs, sparse attention,
index scores and selection, pool allocation/rebuild, model prefill/decode/images,
engine configuration and the shared page manager. A subsequent focused CUDA
test also passed for two requests crossing odd compression and page boundaries,
with identical outputs and shared index IDs between BF16 and packed storage.

The equivalent focused test command from the repository root is shown below.
The recorded run invoked `python -m pytest` inside a private development
container; the container configuration is not part of this repository.

```bash
uv run python -m pytest \
  tests/kernels/test_dsv41_quant.py \
  tests/kernels/test_dsv41_sparse.py \
  tests/kernels/test_dsv41_indexer.py \
  tests/models/test_dsv41_attention.py \
  tests/models/test_deepseek_v41_model.py \
  tests/engine/test_deepseek_v41_engine.py \
  tests/kvcache/test_dsv41_pool.py \
  tests/engine/test_kv_quant_config.py \
  tests/engine/test_attention_backend_matrix.py \
  tests/scheduler/test_dsv4_generic_manager.py -q -p no:cacheprovider
```

The native codecs reproduce the BF16 roundtrip values exactly. Index scores and
selected IDs also matched at the model's 32 index heads. Attention with 64 heads
showed small floating-point differences from the BF16 kernel: three of 32,768
output elements in one decode case and 36 of 1,048,576 in one prefill case, both
with maximum absolute difference `6.103515625e-5`. Packed storage therefore does
not imply bit-identical model logits for every launch shape.

An isolated attention comparison used batch 1, 64 heads, dimension 512, window
128 and 640 selected rows, five warmups and 31 CUDA-event samples:

| Attention call | BF16 median | Packed median | Packed / BF16 |
| --- | ---: | ---: | ---: |
| Decode, one query, five splits | 0.153664 ms | 0.143968 ms | 0.937 |
| Prefill, 32 queries, no splits | 0.123488 ms | 0.275744 ms | 2.233 |

These are CUDA-event elapsed times around Python attention-wrapper calls,
including possible host enqueue gaps and output/scratch allocation effects;
they are not isolated kernel execution times or end-to-end generation throughput.
Prefill calls were slower despite using less KV storage. The packed kernel requires 100,352 bytes
of shared memory against this GPU's 101,376-byte limit, using one pipeline stage.
It retains BF16 tiles and promotes them separately for each FP32 dot operation;
promoting too early or increasing pipeline stages exceeds the device limit.
Rerun the 512-dimensional regressions after compiler/kernel changes.

The baseline is this branch's existing V4.1 BF16 implementation. The repository's
unmodified base at `04d4621` lacks this V4.1 model path and cannot serve as a
same-checkpoint baseline.

The private validation image was
`sha256:d6f7429ed8939fd981d2e26359f377deddaf1e926839c8ccfe3e09403b2d4ad3`
and used the model settings in [Direct launch](#direct-launch). Image digests
identify local validation artifacts; they are not published images or build
instructions. The private Dockerfile, Compose configuration and download helper
are outside the scope of this repository.

All 390 Python source files in the image matched the tested workspace. The
server became ready at 21:27:26 UTC on September 11, using the pinned LibertAIDAI
checkpoint and one RTX 5090. The automatic planner resolved the following
allocations; pool byte counts include its owned tensors:

| Allocation | Previous BF16 image | FP8-FP4 image |
| --- | ---: | ---: |
| KV pool | 363.06 MiB | 176.40 MiB |
| Usable full pages, 128 tokens each | 259 | 279 |
| Usable window pages | 51 | 55 |
| Cached experts | 818 | 830 |
| Free GPU memory after initialization | 2.98 GiB | 2.97 GiB |

The allocated KV pool decreased by 51.4% while page and expert counts increased.
This is the planner's actual allocation, not a comparison at fixed capacity;
freed bytes were reassigned, so total free GPU memory did not increase.

Before API checks in each mode, both compressed Engram tables were read once
with a 64 MiB buffer and `mincore` reported 100% residency. Identical streaming
chat requests used `temperature: 0`, `thinking: {"type": "disabled"}` and
`stream_options: {"include_usage": true}`. The three requests followed a short
`2 + 2` warmup. Each cell below is request wall time / first content-token time:

| Request | BF16 | FP8-FP4 first pass | FP8-FP4 repeat |
| --- | ---: | ---: | ---: |
| Japanese, 22 input / 42 output tokens | 37.22 / 17.46 s | 49.69 / 18.56 s | 24.58 / 15.08 s |
| Corn image, 198 input tokens | 31.15 / 18.60 s | 52.43 / 21.41 s | 24.71 / 15.48 s |
| Multi-turn arithmetic, 27 input / 2 output tokens | 17.93 / 17.69 s | 18.66 / 18.42 s | 15.37 / 15.13 s |

The Japanese prompt was `日本語で、空が青く見える理由を短い一文で説明してください。`
with `max_tokens: 48`. Its answer was identical in all three runs. The image
request used the same local `corn.jpeg` and `Describe this image briefly.`, also
with `max_tokens: 48`. Both modes correctly described three ears of corn, one
partially husked; wording differed, producing 42 BF16 tokens versus 41 packed
tokens. Multi-turn history was user `What is 4 + 5?`, assistant `9`, then user
`Double that number. Reply with only the number.` with `max_tokens: 16`. All runs
answered `18` and ended normally. Both packed passes produced identical text.
Health remained `ok` after the checks.

These single samples do not establish a throughput improvement or equivalent
model quality. Initial packed requests were slower, and repeated requests were
faster. The indexer specializes its key width during eager decode, so new
positions can compile new kernels; packed mode uses separate specializations
from BF16. This is a likely contributor to first-pass latency, not an isolated
measurement of compilation cost. Different expert/page allocations and cache
warming also affect these results. Prefill attention remains slower in the
isolated comparison above, and unseen context lengths can incur more compilation.

## Implementation and current limits

- V4.1 has a separate model registration and nested-config parser. It implements
  CSA2 ratio-1/ratio-2 compressed attention, shared source layers, two-stage index
  selection, shifted mHC mixing, image-aware routing and Engram lookup.
- Resident projections use FP8 weights with 32x32 UE8M0 scales. Routed experts use
  the checkpoint's NVFP4 values, per-16 E4M3 scales and global scales in W4A16
  execution. This activation precision differs from the donor reference and is
  not a claim of bit-identical logits or unchanged model quality.
- Window, compressed and index keys retain their required FP8/FP4 quantization.
  `--kv-cache-dtype bf16` stores reconstructed values; `fp8-fp4` stores their
  original codes and scales. Additional NVFP4 window quantization is not implemented.
- Eager execution is required. Explicit CUDA graph capture is rejected because
  the variable-length index search is not ready for practical long-context graphs.
- Equal index scores select earlier positions deterministically. This preserves
  chunk consistency but can differ from the reference's unspecified top-k ties.
- MTP weights are excluded; speculative decoding is not implemented.
- Engram supports FP8 E4M3 and packed FP4 E2M1, both with block-32 UE8M0 scales.
  FP4 values use the low nibble first. This is distinct from the experts'
  NVFP4 block-16 E4M3 scales and global scales.
- FTW conversion streams Engram tables to a standalone `engram/` directory and
  retains their format in a version-2 manifest and nested configuration. Legacy
  version-1 FP8 manifests remain readable. The source snapshot is not
  needed when serving the completed FTW directory.

## Validation

### LibertAIDAI FP4 Engram checkpoint

The pinned checkpoint has 143,317 tensors across 48 shards. All 138,240 backbone
expert components passed the native header validator, and all 1,480 resident
parameters matched the meta model without missing names, extra names or shape
differences. Its 46,080 backbone expert global scales range from 2^-7 to 2^-3;
all are positive, finite and exactly representable in the loader's FP16 storage.
An independent arithmetic E2M1 decoder matched the native BF16 lookup exactly
for 1,002 sampled rows in each real Engram table, including the first/last rows,
all 16 E2M1 codes, block scales, duplicate IDs and reordered queries. These tests
ran against the installed code in image
`sha256:a9525a2fdd6f7c8217266c15a6bb93c1bc85a64e86113f01c06b2607de86e7f2`;
all 388 Python source files in that image matched the tested workspace.

All 48 shards (429,406,627,896 bytes) matched their SHA256 download etags during
Docker import. The final cache passed size checks for all 90 repository files
and safetensors/index checks for all 143,317 tensors. This was checkpoint
verification in the private validation environment, not a repository download
or import workflow.

On 2026-09-11 UTC, the private container setup started the image on the same
RTX 5090/driver 591.86 host, using the [Direct launch](#direct-launch) settings
with `--kv-cache-dtype bf16`. The API became ready at 20:23:03 UTC, 169 seconds
after process startup; parallel expert reading took 144 seconds. The resolved
cache retained 818 experts and 259 KV pages, with 14 CPU-decode layers and 26
GPU-offload layers. The engine reported 2.98 GiB of free GPU memory after
initialization. The model process used 289.95 GiB RSS with no process swap after
the first text request; this excludes unmapped Engram file-cache pages.

With `thinking: {"type": "disabled"}` and `temperature: 0`, the first text
request (`What is 2 + 2? Reply with only the number.`, `max_tokens: 16`) returned
`4` in 64.27 seconds, with 18 input and two output tokens. The corn image request
shown below (`max_tokens: 32`) returned the same correct description as the
earlier checkpoint in 48.30 seconds, with 201 input and 29 output tokens.
These two prompts are functional checks, not a general quality comparison or
a throughput benchmark.

After these requests, both Engram shards were read once with a 64 MiB buffer
while the model remained running. This took 72.18 seconds. `mincore` then
reported 100% residency for both tables: all 104,451,107,600 compressed bytes
(97.28 GiB) were in Linux page cache. Container `memory.current` was 390.73 GiB,
and Windows still reported 73.76 GiB of free physical memory. WSL swap usage
remained zero before and after warming.

WSL `MemAvailable` was 180.99 GiB, which includes reclaimable Engram pages;
subtracting the entire Engram cache gives approximately 83.71 GiB as a
conservative remaining-memory estimate with those pages retained. Docker's usual
stats display was about 292 GiB because it subtracts inactive file cache. Do not
interpret that display as the total with all Engram pages cached. Page residency
is an observation at measurement time; the OS can reclaim cached pages later.
The same text prompt still returned `4` after warming (18 input/two output
tokens, 25.48 seconds). This measurement does not isolate cache-warming benefits
or establish improved generation throughput.
After that generation, both tables still had 100% page residency, container
memory was 390.67 GiB, model RSS was 290.28 GiB, and both process and WSL swap
usage were zero. The API remained healthy.

The compatibility change passed 37 configuration/weight tests and a separate
37-test Engram/model/Engine regression run. After the final CPU-device guard,
the Engram suite passed 28 tests. These runs overlap. They cover independent
E2M1 decoding, requested-row-only reads, packed FTW roundtrips, legacy FP8
manifests, CUDA staging and image prefill/decode in the small engine.

### Earlier s-zaizen FP8 Engram checkpoint

The earlier s-zaizen checkpoint at revision
`179b7cda25486efbaaf8637d696759d9a791d8bd` was loaded and served successfully on
2026-09-11 UTC with one RTX 5090, NVIDIA driver 591.86, CUDA 13 and
PyTorch `2.11.0+cu130`. The tested Docker image was
`sha256:87805e7660460773ef16d6018bc547fafe783b98d019d7c8b5158ac08da7a18f`.
All 48 checkpoint shards, totaling 527,293,384,648 bytes, matched their SHA256
etags; all 188,245 tensor entries matched the checkpoint index.

The private container setup used the [Direct launch](#direct-launch) settings
with `--kv-cache-dtype bf16`, the local s-zaizen snapshot as `--model`, and
`s-zaizen/DeepSeek-V4.1-Flash-NVFP4` as `--served-model-name`.

This run used a 32,768-token context, 1,024-token prefill chunks, two concurrent
request slots, parallel expert loading, automatic MoE cache sizing and disabled
prefill overlap. Startup ran from 19:11:16 to API readiness at 19:14:14 UTC
(178 seconds); the parallel expert read took 151 seconds. The resolved cache had
818 expert slots and 259 KV pages. RAM use after initialization was 290.7 GiB;
the engine reported 2.94 GiB of free GPU memory. The 285.04 GiB expert banks used
14 CPU-decode layers and 26 GPU-offload layers, matching the WSL pin-budget plan.

HTTP chat requests used `thinking: {"type": "disabled"}` and `temperature: 0`:

| Request | Input / output tokens | Wall time | Result |
|---|---:|---:|---|
| First text request, `max_tokens: 16` | 18 / 2 | 73.15 s | `4` |
| Image request, `max_tokens: 32` | 201 / 29 | 49.14 s | Description below |
| Same text request after the image, `max_tokens: 16` | 18 / 2 | 17.78 s | `4` |

The text prompt was `What is 2 + 2? Reply with only the number.` The image request
used the model repository's `corn.jpeg` with
`Describe this image in one short sentence.` Its response was:

> Three ears of fresh corn with green husks are arranged on a white background, with the central ear partially peeled to reveal bright yellow kernels.

Health checks succeeded, and `/v1/models` reported the configured served model ID
and context length 32768. These are limited functional smoke tests. Request wall
times include all request processing; the cause of the first-request delay was
not isolated. General text/image quality, full million-token inference, sustained
performance and FTW startup with this complete checkpoint remain unvalidated.

Automated tests compare operations with independent CPU/dequantization references
and exercise a complete small V4.1 model, including NVFP4 experts, Engram, vision,
chunked prefill, decode and request isolation. A separate 1,048,576-key index
search test checks bounded temporary GPU allocation.

```bash
uv run pytest tests/models/test_deepseek_v41* tests/models/test_dsv41_attention.py \
  tests/kernels/test_dsv41* tests/kvcache/test_dsv41_pool.py \
  tests/scheduler/test_multimodal_chunks.py -q
```

The broader suite run reported 1,858 passes and six failures. One missing AOT
registration introduced by this change was fixed, followed by 30 passing targeted
tests. Four GLM failures also reproduced at the unchanged HEAD (`04d462149e21`). An intermittent
device-to-device copy test failed in the broader run and passed five isolated
reruns; this does not establish a clean full-suite result. Separate focused runs
passed nine tests in the private validation image.
These counts describe separate runs and are not added together.

The adapted reference components retain their upstream MIT attribution in
`python/freetoken/models/deepseek_v41/NOTICE`.
