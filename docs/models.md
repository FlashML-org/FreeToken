# Supported models

FreeToken loads HF safetensors checkpoints directly (plus native GGUF for
Gemma-4, Qwen3.5-MoE/Ornith, and Laguna). The checkpoints below are known-good —
the prebuilt kernels are tuned for them; other checkpoints of the same architectures
work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Ornith 1.5 35B-A3B | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) (native Q4_K_M GGUF) |
| Qwen3.6 dense | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| Poolside Laguna-S 2.1 | compressed-tensors INT4 safetensors (including its BF16 expert tail), native GGUF |
| NVIDIA Nemotron 3 Super | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

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
- Laguna-S INT4 needs the `offload` backend. On WSL, FreeToken automatically
  keeps enough layers on CPU when the mixed INT4/BF16 banks exceed the CUDA
  pinned-memory budget.
- A single-session 200K Laguna configuration on a 16 GB GPU should reserve the
  minimum 256 expert slots, use INT4 KV, and keep the SWA pool near its working-set
  floor: `--max-running-requests 1 --max-seq-len-override 200000 --num-tokens 200000
  --kv-cache-dtype int4 --moe-cache-size 256 --disable-moe-prefill-overlap
  --swa-full-tokens-ratio 0.006 --memory-ratio 0.95`.
- For Ornith Q4_K_M at 200K on a 16 GB GPU, use one request, Q4_0 KV, 5,000
  expert slots, and the default 8K prefill chunks: `--max-running-requests 1
  --max-seq-len-override 200000 --num-tokens 200000 --kv-cache-dtype q4_0
  --moe-backend offload --moe-cache-size 5000 --max-prefill-length 8192
  --memory-ratio 0.95`. On the RTX 2000 Ada/WSL test host, cold 32K TTFT was
  51.1 s at 8K chunks versus 54.6 s at 16K; `--moe-prefill-hit-d2d` was slower
  on this stack and should remain disabled. Install the optional SGLang kernel
  (`freetoken[sgl]`) for faster expert-route alignment. FreeToken's Q4_0 path matches
  llama.cpp's block quantizer and is validated with normal answers, OpenAI tool calls,
  and a 55.6K-token Claude Code Bash-tool round trip. The sm_89 attention tuning reduces
  a synthetic 200K full-attention layer from 2.42 ms to 0.92 ms; the live 55.9K decode
  ran at 36--48 tok/s after warmup. A coherent 169.9K-token live generation completed
  in 425.8 s of prefill and decoded at 27--35 tok/s. `int4` remains an alias for `q4_0`.
- On Blackwell (sm_120, e.g. RTX 5080 16 GB) the same command serves the **full
  262,144-token window**: `--attention-backend triton --max-seq-len-override 262144
  --num-tokens 262144 --kv-cache-dtype q4_0 --max-running-requests 1
  --moe-backend offload --moe-cache-auto --max-prefill-length 8192`. Pass the
  backend explicitly: sm_120 auto-resolves to FlashInfer, which cannot read the
  quantized KV pool. The attention launch tables are
  architecture-aware: the sm_120 Q4_0 decode launch (64 splits, 64-token tiles)
  runs a synthetic 262K full-attention layer in 0.36 ms versus 0.82 ms with the
  sm_89 tuning, and the extend/prefill kernels drop to 4 warps (1.12x on long-Q4
  prefix extension, 2x on cold chunks). BLOCK_N=16 silently corrupts the packed
  Q4 loader on sm_120 exactly as on sm_89 and stays excluded. On sm_120 the
  Q4_K/Q6_K GGUF matmuls (dense prefill and large routed-expert batches) run on
  llama.cpp's int8-tensor-core MMQ (vendored under `kernel/csrc/gguf_mmq/`,
  JIT-built on first use): ~13x over the DP4A kernels and ~1.3x over transient
  dequant+cuBLAS at 8K-token chunks, with the same lossless packed weights.
- Nemotron 3 Super uses its native hybrid Mamba-2 / full-attention / latent-MoE
  architecture. The NVFP4 release needs about 60 GiB of host RAM for expert banks and
  10.3 GiB of resident GPU weights. FreeToken currently serves one concurrent Nemotron
  session. On WSL, `--moe-pageable-gpu` keeps the pin-budget overflow banks pageable,
  stages only their routed misses through a small pinned buffer, and still executes every
  ReLU² expert on GPU. This eager path disables CUDA graphs and prefill overlap. A minimal
  all-GPU-compute launch is:
  `ft serve --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  --max-running-requests 1 --moe-backend offload --moe-cpu-layers 0
  --moe-pageable-gpu --moe-cache-auto`.
