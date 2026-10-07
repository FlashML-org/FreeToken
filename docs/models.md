# Supported models

FreeToken loads HF safetensors checkpoints directly. The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4.1-Flash | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.3-Flash | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.8 / Qwen3.6 dense | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)), [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Qwen3-VL | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| MiniMax-M3 | [nvidia/MiniMax-M3-NVFP4](https://huggingface.co/nvidia/MiniMax-M3-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

### Image input

These families accept image input by default; pass `--text-model-only` to skip the vision encoder. The flags are described in the
[CLI reference](cli.md#image-input); each family reads them in its own units.

| Family | Image tokens | `--image-min-tokens` / `--image-max-tokens` | `--mm-processor-kwargs` example |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash (ViT tower, streamed under `--mm-encoder-weights host`) | one feature per 42x42 pixels, plus a learned row break per row and two delimiters; checkpoint cap 1024 total span tokens | maximum caps the entire span; minimum maps to a pixel-area floor before the cap; checkpoint minimum area is 544x544 | `{"max_image_tokens": 512, "min_pixels": 295936}` |
| Qwen3.6 (both variants, every listed weight format), Qwen3.8-Flash-Next, Qwen3-VL | one token per 32x32 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 64 to 16384 tokens | `{"size": {"longest_edge": 1048576}}` |
| Gemma-4 26B-A4B, 31B (`gemma4`: ViT tower, streamed under `--mm-encoder-weights host`) | one of the soft-token budgets 70 / 140 / 280 / 560 / 1120, every image scaled to its budget as far as the aspect ratio allows | the maximum picks the largest budget within it, below 70 is refused at start-up; the minimum has no effect | `{"max_soft_tokens": 1120}` |
| Gemma-4 12B (`gemma4_unified`: linear patch embedder, resident under either placement) | same budgets, one 48x48 super-patch per soft token | same as the tower releases | same |
| GLM-5.3-Flash (`glm5_next`: ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution on a canvas zero-padded to a 28-multiple | token counts, passed through as the processor's `min_image_tokens` / `max_image_tokens`; checkpoint defaults 16 to 8000 tokens | `{"max_image_tokens": 2048}` |
| Muse-Glimmer-30B (windowed ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, aspect ratio kept under a token cap; checkpoint default 4096 tokens | the maximum is the cap (`max_image_tokens`); the minimum has no effect | `{"max_image_tokens": 1024}` |
| MiniMax-M3 (`minimax_m3`: CLIP-style ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 4 to 576 tokens | `{"size": {"longest_edge": 1048576}}` |

## MoE strategies

`ft serve --moe-strategy {auto,fused,offload,cpu,hybrid}` (`--moe-backend` is the deprecated old spelling):

- **fused** — experts resident on GPU (needs the VRAM). Serves every expert
  format except GGUF and DeepSeek-V4's.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it. Unified-memory GPUs differ, see below.

### Unified-memory GPUs (GB10 / DGX Spark)

The GPU and the CPU share one DRAM, so offload only copies experts between two
names for the same memory. On these GPUs `auto` resolves every MoE model to
`fused`: the default is the resident path only. GGUF and DeepSeek-V4 experts
have no resident path and stay on `offload`. Offload-only flags
(`--moe-cpu-layers`, `--moe-cache-size` / `-rate` / `-auto`,
`--moe-prefill-hit-d2d`, `--disable-moe-prefill-overlap`) are ignored with a
warning; pass `--moe-strategy offload` to use them.

### NVLink-C2C hosts (GH200 / GB200)

The host link is ~450 GB/s per direction instead of PCIe's ~32-64 GB/s. The
offload expert gather reads host memory zero-copy and is latency-bound, so it
needs a wider grid to keep enough loads in flight. Set
`FREETOKEN_H2D_BLOCKS_PER_BANK`:
```
    FREETOKEN_H2D_BLOCKS_PER_BANK=32 ft serve --model <model> --moe-strategy offload
```
On GH200 this raises the gather from ~220 to ~410 GB/s (`PCIe-gather` in
`ft bench bw`), matching the DMA ceiling. Run `ft bench bw` with the variable
set so the hybrid split is calibrated against the faster gather. Leave it
unset on PCIe GPUs: wider grids add no bandwidth there.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- An FTW converted by an older build with `--moe-backend fused` or `triton` keeps
  its experts as dense weights and no longer loads; reconvert it with `ft checkpoint`.
- FTW files converted by builds before the quantization refactor may fail to load;
  see [ftw-hotfix.md](ftw-hotfix.md) for the affected checkpoints and the repair tool.
- An FTW converted before its family served images holds no vision encoder: `ft serve`
  refuses it unless started with `--text-model-only` (or `--mm-disable vision`); reconvert it
  with `ft checkpoint`, or add the encoder in place with [scripts/ftw_hotfix.py](ftw-hotfix.md).
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- DeepSeek-V4.1-Flash serves text and images from the HF checkpoint as shipped (fp8 block-32
  dense weights, fp4 experts on the offload cache, 890 B/token global KV in packed
  fp4/fp8 pools). Its two 98 GiB Engram tables stream from the checkpoint shards on
  demand (keep them on a fast NVMe; the 6 GiB of table scales stay in host RAM), using
  bounded pinned staging buffers; an FTW conversion copies the shards that hold them next
  to the checkpoint.
  SWA bounded replay (DeepSeek_V41_Tech_Report.pdf, shipped in the checkpoint, §3.2.2): a replayed
  segment recomputes only the SWA KV of its last `n_win` tokens and truncates each query's window to
  the segment. *Encoder* replay rebuilds the encoder SWA KV behind a prefix hit from the global KV
  alone (approximate; the report's fallback when the SWA KV of a hit has been evicted). *Decoder*
  replay runs the decoder layers on each prompt's last `n_win` tokens only; their SWA KV is never
  prefix-cached and post-training simulated it. FreeToken keeps encoder SWA KV in the radix cache (no
  encoder replay); `--swa-decoder-replay bounded` (default) is the report's decoder replay, with a
  prefix hit stopping at least `n_win` tokens before the prompt end so those tokens are prefilled;
  `exact` runs the decoder on every prompt token (reference numerics).
  Bounded output differs from exact by construction and does not depend on the prefill chunk size.
  Under expert offload the prefill time is bounded by streaming each layer's experts, so decoder
  replay saves decoder-layer compute, not prefill time.
  Run `ft bench bw --model dsv4.1-flash` to measure local PCIe and CPU bandwidth before
  choosing a MoE strategy; `--moe-strategy hybrid` splits expert misses using those
  measurements. With `--moe-cache-auto`, `--kv-reserve-tokens` sets the token capacity reserved
  before the expert cache is allocated, subject to the attention pool's structural minimum; size it
  for the intended context and concurrency. `--max-extend-length` controls the prefill chunk size
  and its workspace. Image input uses the shared multimodal processor and encoder cache, including
  chunked prefill and prefix replay; `--text-model-only` skips the vision weights. DSpark
  speculative decoding is not served.
  `reasoning_effort` takes `low`, `high` or `max`; the 1-100 integer budget goes through the
  template kwargs with thinking on, `"chat_template_kwargs": {"enable_thinking": true,
  "reasoning_effort": 37}`, and the checkpoint's encoder validates it. Runtime window-cache
  controls use 128-token pages.
- Qwen3.8-Flash-Next keeps a 47.7 GiB PLE n-gram table pinned in host RAM.
