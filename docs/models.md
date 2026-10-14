# Supported models

FreeToken loads HF safetensors checkpoints directly. The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too. The MiMo v2.6 Flash row is a recent port and is not yet
covered by that validation; its capabilities section below states exactly what has and has not been verified.

| Model | HF checkpoints |
|---|---|
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
| MiMo v2.6 Flash (text-only) | `MiMoV2ForCausalLM` safetensors checkpoints (BF16 or ModelOpt routed W4A16_NVFP4) |

### MiMo v2.6 Flash capabilities

MiMo v2.6 Flash is supported at tensor parallel size 1. The loader serves only the
text tower; vision, audio, speech-embedding, and MTP weights are intentionally
excluded. BF16 checkpoints may use an index or a single safetensors file without
one. Routed experts may be BF16 resident weights or ModelOpt `MIXED_PRECISION`
experts explicitly listed as `W4A16_NVFP4` with group size 16. The NVFP4 expert
path uses the ModelOpt `weight_scale_2` global-scale layout and the offload backend;
other quantization dialects, quantized dense projections, shared experts, and TP>1
are not supported by this family loader.

Validation status: unlike the families above, this port has not been validated end
to end on real hardware. The repository includes synthetic config/weight tests and
eager vs cached-attention numerical comparisons; their presence is not evidence
that they ran on this revision or on checkpoint weights. Supplied external NVFP4
metadata (`config.json`, `model.safetensors.index.json`) was inspected, not loaded
as a complete model. Unverified: loading the full release
checkpoint on a GPU, quantized expert numerics through the offload backend with
real weights, scheduler reuse, graph replay, and performance. The upstream modeling
code and design notes this port was derived from are not distributed with the
repository; code comments cite them only as provenance for ported behavior, not
exact numerical equivalence. In particular, the external `MiMoV2MoEGate` requests
unsorted expert top-k and this port uses sorted top-k; ordering and full-model
equivalence are not claimed. The reproducible parser/template contract and its
separate external corroboration limits are described in
[`tests/server/mimo_chat_contract.md`](../tests/server/mimo_chat_contract.md).

Checkpoint facts, read directly from the NVFP4 release files (not executed):
`config.json` declares `architectures: ["MiMoV2ForCausalLM"]`, `model_type: "mimo_v2"`,
`dtype: "bfloat16"`, 48 layers split by `hybrid_layer_pattern` into 9 full-attention
and 39 SWA layers (`sliding_window` 128), a dense layer 0 followed by 47 MoE layers
with 256 routed experts, top-8 sigmoid `noaux_tc` routing, no shared experts,
`head_dim` 192 with `v_head_dim` 128, dual RoPE (`rope_theta` 10000000.0,
`swa_rope_theta` 10000.0, `partial_rotary_factor` 0.334), `attention_value_scale`
0.707, and a `quantization_config` (`quant_method: "modelopt"`,
`quant_algo: "MIXED_PRECISION"`, `group_size: 16`) whose `quantized_layers` map
lists exactly 36096 modules (47 layers x 256 experts x 3 projections), each
`W4A16_NVFP4`. `model.safetensors.index.json` maps 109114 tensors across 65 shard
files (`metadata.total_size` 193964338432, about 181 GiB); every routed-expert
projection ships `weight`, `weight_scale`, and `weight_scale_2` tensors, and the
index also carries the MTP, vision, audio, and speech-embedding tensors the loader
drops. The checkpoint's own `conversion_metadata.source_revision` is
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`. This is a metadata claim, not an
independently authenticated checkpoint, modeling-code, or tokenizer/template revision.

### Image input

These families accept image input by default; pass `--text-model-only` to skip the vision encoder. The flags are described in the
[CLI reference](cli.md#image-input); each family reads them in its own units.

| Family | Image tokens | `--image-min-tokens` / `--image-max-tokens` | `--mm-processor-kwargs` example |
| --- | --- | --- | --- |
| Qwen3.6 (both variants, every listed weight format), Qwen3.8-Flash-Next, Qwen3-VL | one token per 32x32 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 64 to 16384 tokens | `{"size": {"longest_edge": 1048576}}` |
| Gemma-4 26B-A4B, 31B (`gemma4`: ViT tower, streamed under `--mm-encoder-weights host`) | one of the soft-token budgets 70 / 140 / 280 / 560 / 1120, every image scaled to its budget as far as the aspect ratio allows | the maximum picks the largest budget within it, below 70 is refused at start-up; the minimum has no effect | `{"max_soft_tokens": 1120}` |
| Gemma-4 12B (`gemma4_unified`: linear patch embedder, resident under either placement) | same budgets, one 48x48 super-patch per soft token | same as the tower releases | same |
| GLM-5.3-Flash (`glm5_next`: ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution on a canvas zero-padded to a 28-multiple | token counts, passed through as the processor's `min_image_tokens` / `max_image_tokens`; checkpoint defaults 16 to 8000 tokens | `{"max_image_tokens": 2048}` |
| Muse-Glimmer-30B (windowed ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, aspect ratio kept under a token cap; checkpoint default 4096 tokens | the maximum is the cap (`max_image_tokens`); the minimum has no effect | `{"max_image_tokens": 1024}` |
| MiniMax-M3 (`minimax_m3`: CLIP-style ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 4 to 576 tokens | `{"size": {"longest_edge": 1048576}}` |

## MoE strategies

`ft serve --moe-strategy {auto,fused,offload,cpu,hybrid}` (`--moe-backend` is the deprecated old spelling):

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- FTW files converted by builds before the quantization refactor may fail to load;
  see [ftw-hotfix.md](ftw-hotfix.md) for the affected checkpoints and the repair tool.
- An FTW converted before its family served images holds no vision encoder: `ft serve`
  refuses it unless started with `--text-model-only` (or `--mm-disable vision`); reconvert it
  with `ft checkpoint`, or add the encoder in place with [scripts/ftw_hotfix.py](ftw-hotfix.md).
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Qwen3.8-Flash-Next keeps a 47.7 GiB PLE n-gram table pinned in host RAM.
