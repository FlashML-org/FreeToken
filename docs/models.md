# Supported models

FreeToken loads HF safetensors checkpoints directly, plus native GGUF for the
architectures listed under [GGUF](#gguf) below. The checkpoints below are
known-good — the prebuilt kernels are tuned for them; other checkpoints of the
same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.6 dense | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## GGUF

Native GGUF, meaning the block-quantized weights are kept packed and dequantized inside
the kernels rather than expanded to bf16 at load.

| GGUF `general.architecture` | Covers |
|---|---|
| `gemma4` | Gemma-4 |
| `qwen3moe` | Qwen3 MoE (e.g. Qwen3-235B-A22B, Qwen3-30B-A3B) |
| `qwen35moe` | Qwen3.5 / Qwen3.6 MoE (e.g. Qwen3.6-35B-A3B, Qwen3.5-122B-A10B) |
| `qwen35` | Qwen3.5 / Qwen3.6 dense (e.g. Qwen3.6-27B, Qwen3.5-9B) |

Split checkpoints load: point `--model` at any shard of a `-00001-of-000NN` set, or at the
directory holding them. Metadata, config and tokenizer are read from shard 1 (later shards
carry only `split.*` keys), and the tensor tables are aggregated across the set. A missing
shard raises with the index named rather than loading a partial model.

Quant types follow what the vendored kernels in `csrc/gguf/` implement:

- Standard and K-quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K through Q6_K) use MMQ for
  prefill and MMVQ for decode.
- I-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS) have no
  MMQ kernel, so prefill dequantizes and runs a plain matmul; decode uses MMVQ.

Two constraints worth knowing before picking a file:

- A MoE checkpoint's routed-expert banks must use one ggml type across every layer. The GPU
  slot pool is a single allocation with a single row stride, so a bank that changes type
  between layers cannot be served and the load fails with the offending layers named.
  llama.cpp's `_M` and `_XXS` levels raise the precision of the first few layers'
  `ffn_down_exps` and hit this; `llama-quantize --pure` produces a checkpoint that loads.
  Dense models have no expert banks and are unaffected.
- GGUF paths are TP=1 only, and a NextN/MTP block in the checkpoint is dropped (served
  text-only, no speculative decoding).

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

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
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Multimodal checkpoints are served text-only.
