<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/issues/482"><b>Community WeChat</b></a> |
</p>

## Windows NVFP4 Fork Enhancements

This fork adds Windows-focused support for serving large Qwen3.8 and Gemma4 NVFP4/GGUF checkpoints on consumer hardware.

### Engine 0.2 performance on Blackwell

Engine 0.2 brings substantial performance improvements on NVIDIA Blackwell GPUs. Depending on the model and serving setup, some models may reach speeds of up to 100 tokens per second. In current testing, Qwen3.8-Flash-Next NVFP4 reaches about 10 tokens per second, with room for further improvement. Actual throughput varies with the GPU, checkpoint, context length, and serving configuration.

### Disk-tier MoE offload

The Windows disk tier keeps only a configurable prefix of routed experts in host RAM and fetches the remaining NVFP4 expert rows directly from the original safetensors files when needed. This avoids materializing the full expert bank in memory or requiring a page file large enough for the entire model.

Example for a 128 GiB RAM system:

```cmd
ft serve --model "%USERPROFILE%\.freetoken\models\Qwen3.8-Flash-Next-Uncensored-NVFP4" --port 1919 --moe-strategy offload --moe-disk-tier on --expert-ram-experts 128 --disk-fetch-workers 16 --disable-moe-prefill-overlap --cuda-graph-max-bs 0 --max-running-requests 4 --memory-ratio 0.95 --host 0.0.0.0 --cors-origins "*"
```

The disk-tier prototype currently targets Qwen4Exp/Qwen3.8 native NVFP4 checkpoints, GPU offload, disabled MoE prefill overlap, and disabled CUDA graphs. It uses Windows-compatible file reads and does not depend on Linux `io_uring`, `O_DIRECT`, or `preadv`.

### Checkpoint and Windows fixes

- Native Q8_0 Gemma4 GGUF embeddings, projections, and routed experts.
- Qwen3.8 FP8 GDN loading with preserved weight scales.
- BF16 Qwen3.8 vision-tower loading alongside the quantized text tower.
- Windows Qwen4Exp disk PLE fallback to the compatible legacy native store.
- Windows-safe expert-bank residency and stale compiled-module shadow handling.
- GGUF tokenizer, metadata, and model-repair utilities for local checkpoints.

## My model collection

This list reflects the checkpoints in my local `~/.freetoken/models` collection, rather than every architecture supported by FreeToken. The Hugging Face source links identify the matching checkpoint when its local card or GGUF metadata recorded one.

**These specific model builds and the Windows launch workflows documented here are supported by this fork, not by the upstream/origin FreeToken repository.** Upstream does not include all of this fork's checkpoint compatibility fixes and Windows-specific features, so do not expect these models to run there with the commands or configuration shown here.

| Local model folder | Hugging Face checkpoint / source | Upstream base |
|---|---|---|
| `gemma-4-26B-A4B-it-abliterated-Q4_0-GGUF` | [huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated](https://huggingface.co/huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated) (local Q4_0 GGUF) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `gemma-4-26B-A4B-it-ultra-uncensored-heretic-BF16-nvfp4-with-metadata` | [llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic) (local NVFP4 export) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `gemma-4-26B-A4B-it-ultra-uncensored-heretic-GGUF` | [mradermacher/gemma-4-26B-A4B-it-ultra-uncensored-heretic-GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-ultra-uncensored-heretic-GGUF); source safetensors: [llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `Gemma-4-26B-A4B-it-Uncensored-EXPERTS-NVFP4` | Local mixed GGUF: routed expert weights are NVFP4 and other tensors remain BF16; metadata marks the HF-NVFP4 layout. | Gemma 4 26B A4B Instruct family: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `Gemma-4-26B-A4B-it-Uncensored-FP8` | [orcarouter/Gemma-4-26B-A4B-it-Uncensored-FP8](https://huggingface.co/orcarouter/Gemma-4-26B-A4B-it-Uncensored-FP8) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `Gemma-4-26B-A4B-it-Uncensored-TRC-nvfp4` | Local mixed NVFP4/BF16 GGUF (11,921 NVFP4 and 12,277 BF16 tensors in the local file). | Gemma 4 26B A4B Instruct family: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `Ornith-1.5-35B-A3B-FULL-OBLITERATED-NVFP4` | [jhone888/Ornith-1.5-35B-A3B-FULL-OBLITERATED-NVFP4](https://huggingface.co/jhone888/Ornith-1.5-35B-A3B-FULL-OBLITERATED-NVFP4), derived from [jhone888/Ornith-1.5-35B-A3B-FULLY-OBLITERATED](https://huggingface.co/jhone888/Ornith-1.5-35B-A3B-FULLY-OBLITERATED) | [huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated](https://huggingface.co/huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated), from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| `Ornith-1.5-35B-A3B-uncensored-NVFP4` | [ressl/Ornith-1.5-35B-A3B-uncensored-NVFP4](https://huggingface.co/ressl/Ornith-1.5-35B-A3B-uncensored-NVFP4) | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| `Qwen3.6-Fallen-Fabulist-35B-A3B-heretic` | [Cyclone-Labs/Qwen3.6-Fallen-Fabulist-35B-A3B-heretic](https://huggingface.co/Cyclone-Labs/Qwen3.6-Fallen-Fabulist-35B-A3B-heretic) | Merge of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Gryphe/WorldSim-Opus-3.6-35B-A3B](https://huggingface.co/Gryphe/WorldSim-Opus-3.6-35B-A3B), [ReadyArt/Melody1437-35B-A3B](https://huggingface.co/ReadyArt/Melody1437-35B-A3B), and [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| `Qwen3.6-Fallen-Fabulist-35B-A3B-heretic-GGUF` | [mradermacher/Qwen3.6-Fallen-Fabulist-35B-A3B-heretic-GGUF](https://huggingface.co/mradermacher/Qwen3.6-Fallen-Fabulist-35B-A3B-heretic-GGUF); source: [Cyclone-Labs/Qwen3.6-Fallen-Fabulist-35B-A3B-heretic](https://huggingface.co/Cyclone-Labs/Qwen3.6-Fallen-Fabulist-35B-A3B-heretic) | Same merged bases as the safetensors checkpoint above |
| `Qwen3.8-Flash-Next-ABLITERATED-NVFP4` | [dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4) | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| `Qwen3.8-Flash-Next-Uncensored-FP8` | [orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8) | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| `Qwen3.8-Flash-Next-Uncensored-NVFP4` | [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| `supergemma4-e4b-abliterated-i1-GGUF` | [mradermacher/supergemma4-e4b-abliterated-i1-GGUF](https://huggingface.co/mradermacher/supergemma4-e4b-abliterated-i1-GGUF); source: [Jiunsong/supergemma4-e4b-abliterated](https://huggingface.co/Jiunsong/supergemma4-e4b-abliterated) | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |

### Converting an HF checkpoint to FTW

The HF safetensors directory is the input to [`ft checkpoint`](./docs/cli.md#ft-checkpoint); GGUF models stay GGUF and are not converted with this command. For example, in PowerShell, download the exact customized checkpoint and convert it to a separate FTW directory:

```powershell
hf download dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 --local-dir "$HOME\.freetoken\hf-source\Qwen3.8-Flash-Next-ABLITERATED-NVFP4"
ft checkpoint --model "$HOME\.freetoken\hf-source\Qwen3.8-Flash-Next-ABLITERATED-NVFP4" --out "$HOME\.freetoken\models\Qwen3.8-Flash-Next-ABLITERATED-NVFP4-FTW"
ft serve --model "$HOME\.freetoken\models\Qwen3.8-Flash-Next-ABLITERATED-NVFP4-FTW"
```

To keep a fine-tuned, merged, or quantized variant, use that exact variant's HF repository as the `hf download` source. Downloading and converting only its foundation model (for example, `Qwen/Qwen3.8-Flash-Next`) produces the unmodified base, not the customized model in this list. See the [`ft checkpoint` reference](./docs/cli.md#ft-checkpoint) for options and FTW caveats. HF safetensors can also be served without conversion.

### My Gemma GGUF quantization workflow

The locally assembled Gemma NVFP4 files are GGUFs, not HF safetensors checkpoints to pass directly to `ft checkpoint`. The recorded workflow starts with a quantizer-produced tensor-only GGUF, uses the matching HF safetensors directory as the source of original BF16 tensors, and then adds the Gemma model, tokenizer, and chat-template metadata. The tracked helpers are [make_mixed_gemma4_gguf.py](./scripts/make_mixed_gemma4_gguf.py) and [repair_gemma4_gguf.py](./scripts/repair_gemma4_gguf.py). In PowerShell, their post-quantization steps are:

```powershell
python .\scripts\make_mixed_gemma4_gguf.py `
  .\quantizer-output.gguf `
  "$HOME\.freetoken\hf-source\Gemma-4-26B-A4B-it-Uncensored" `
  .\mixed-bf16-nvfp4.gguf

python .\scripts\repair_gemma4_gguf.py `
  .\mixed-bf16-nvfp4.gguf `
  "$HOME\.freetoken\hf-source\Gemma-4-26B-A4B-it-Uncensored\config.json" `
  "$HOME\.freetoken\hf-source\Gemma-4-26B-A4B-it-Uncensored\tokenizer.json" `
  .\chat-template.txt `
  .\Gemma-4-26B-A4B-it-Uncensored-NVFP4.gguf
```

The expert-only variant keeps NVFP4 on routed expert weights and retains BF16 for the other weights; the TRC variant has a wider mixed NVFP4/BF16 tensor set. For a reproducible rebuild, use the exact customized HF safetensors source (not just Google's base checkpoint), and record the quantizer version, quantization rules, and calibration inputs alongside the output.

### Qwen3.6 experts-only NVFP4 GGUF recipe

For the local `Qwen3.6-Fallen-Fabulist-35B-A3B-heretic-GGUF`, the recorded quantizer settings are `llama-quantize --type nvfp4` with these first-match rules in order:

```text
model\.language_model\.layers\..*\.mlp\.experts\.(gate_up_proj|down_proj)=nvfp4
model\.language_model\..*=bf16
```

The expert rule must precede the BF16 catch-all. This quantizes the stacked expert gate/up and down projections while leaving the other language-model tensors BF16; the recorded output had 164 NVFP4 and 862 BF16 tensors. The resulting tensor-only GGUF was repaired with [repair_qwen_gguf.py](./scripts/repair_qwen_gguf.py) before serving.

## Engine 0.2 and fork updates

The fork is based on [FreeToken Engine 0.2.0](https://github.com/FlashML-org/FreeToken/releases/tag/v0.2.0) and includes subsequent upstream improvements. Highlights include:

- **More model formats and families:** GGUF serving expanded across Gemma 4, Qwen 2/3/3.5, Llama, Mistral, GLM, MiniMax, Muse-Glimmer, and DeepSeek-V4. Depending on the architecture, GGUF checkpoints can use CPU or hybrid MoE experts and can accept image input through a companion `mmproj` file.
- **Faster quantized MoE execution:** NVFP4 and MXFP4 experts can use expert-bank execution; NVFP4 prefill/decode and hybrid cache paths received performance and memory improvements.
- **Engine and API improvements:** improved rotary sizing and checkpoint overrides, DeepSeek-V4 vision support, cache management endpoints, and fixes for sampling, memory accounting, and request handling.
- **Broader hardware support:** experimental ROCm support for AMD RDNA3/RDNA4 GPUs, alongside NVIDIA optimizations; see [AMD installation](docs/install_amd.md).

### Fork-specific Windows and checkpoint work

- A Windows disk tier fetches Qwen3.8 NVFP4 expert rows from the original safetensors shards while keeping a configurable subset of experts in host RAM.
- Qwen3.8 checkpoint loading fixes cover quantized dense-weight remapping/fusion, hyperconnection weights, FP8 GDN weights, BF16 vision weights, and PLE host-table loading.
- Gemma 4 GGUF fixes cover Q8_0 tensor scales and tokenizer handling; additional repair tools help adapt local Gemma and Qwen GGUF files.
- Windows deployment fixes include a compatible Qwen4Exp PLE fallback, safer expert-bank residency, and guarded handling of stale native-extension modules.
- NVFP4 and Q4_0 MoE paths and quantization-config handling have been extended for the checkpoints in [my model collection](#my-model-collection).

### Windows site-library snapshots

[`freetoken-0.2.0+g425dc7bf1-site-lib.zip`](./freetoken-0.2.0+g425dc7bf1-site-lib.zip) is a snapshot of the `freetoken` package from the Windows Desktop app venv at `%LOCALAPPDATA%\FreeToken\venv\Lib\site-packages`. It contains the Python package and active CPython 3.12 Windows native extensions, but not a full venv or third-party dependencies. Bytecode caches and disabled `.pyd.disabled` modules are excluded.

[`freetoken-0.1.3+g52322e984-site-lib.zip`](./freetoken-0.1.3+g52322e984-site-lib.zip) is a historical snapshot, not the current Engine 0.2 package. Use it only when intentionally restoring that older version.

To restore the current snapshot, stop the FreeToken daemon and server, then extract it into the app venv's `site-packages`. Preview stale native extensions before applying the cleanup:

```powershell
$sitePackages = Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Lib\site-packages'
Expand-Archive -LiteralPath .\freetoken-0.2.0+g425dc7bf1-site-lib.zip -DestinationPath $sitePackages -Force
.\scripts\disable-stale-pyd.ps1 -SitePackages $sitePackages
```

Review the preview, then apply it with:

```powershell
.\scripts\disable-stale-pyd.ps1 -SitePackages $sitePackages -Apply
```

The helper uses a fixed allowlist of the 17 module paths already disabled in the current app venv; it does not scan for or disable any other `.pyd` files. It verifies that each target still has a same-name `.py` source and renames only matching ABI-tagged extensions to `.pyd.disabled`, without deleting them. Restart the daemon or server after cleanup.


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](./docs/install.md)
- [Quick start](./docs/quickstart.md)
- [Supported models](./docs/models.md)
- [CLI reference and checkpoint conversion](./docs/cli.md#ft-checkpoint)
- [Repairing old FTW checkpoints](./docs/ftw-hotfix.md)
- [Diagnosing a checkpoint that fails to load](./scripts/checkpoint_diagnose/README.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
