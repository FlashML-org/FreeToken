# GMKtec EVO-X2 284B capacity manifest

This is a read-only capacity snapshot for the GMKtec EVO-X2 Strix Halo system.
It is not a claim that a 284B model fits or serves interactively.

## Observed platform

| Field | Observed value |
| --- | --- |
| GPU | AMD Radeon 8060S Graphics |
| GFX target | `gfx1151` |
| Dedicated VRAM reported by ROCm SMI | 2,147,483,648 bytes (2 GiB) |
| Dedicated VRAM currently used | 368,336,896 bytes |
| System memory total | 59 GiB |
| System memory available at capture | 18 GiB |
| Swap configured | 127 GiB |
| Swap used at capture | 2.2 GiB |
| GPU power | 12.048 W |
| GPU temperature | 30.0 C |
| GPU utilization | 0 percent |
| Performance policy | auto |

The reported 2 GiB VRAM is not the whole unified-memory budget.  Conversely,
the 59 GiB system-memory total is not proof that the GPU can safely allocate
59 GiB for model weights and KV cache.  A valid capacity claim must measure
GPU-visible allocations, runtime reservations, model weights, expert storage,
KV cache, and swap behavior under the exact model configuration.

## Model inventory

A read-only search of the configured model directory found no file or directory
matching `284B`, `280B`, or `235B`.  No 284B capacity test was therefore run.

## Latest live refresh

A second read-only probe on 2026-09-05T12:01:56Z confirmed the capacity
boundary while the protected Qwen service was healthy. The host reported 59
GiB total memory, 18 GiB available, 127 GiB configured swap with 2.0 GiB in
use, and 769 GiB free on the model filesystem. ROCm reported the same 2 GiB
dedicated VRAM and `gfx1151`, with 367,693,824 bytes currently used, 32.0 C,
28.042 W, and zero percent GPU utilization. The configured model payloads were
approximately 22 GiB for Qwen NVFP4, 67 GiB for the Qwen safetensors source,
and 15 GiB for Gemma Q4. No DeepSeek or 284B model payload was found in the
configured FreeToken model directory. Source-code references and archived
configuration names are not model payloads and are not treated as admission
evidence.

## Qualification result

**INCOMPLETE.**  The current manifest establishes the memory and GPU baseline,
but it does not qualify a 284B model.  The next test requires the exact model
artifact, quantization, context length, expert-loading policy, KV reservation,
and a clean-memory run with process-scoped swap telemetry.

## Primary-paper capacity facts

The paper's primary text identifies DeepSeek-V4-Flash as a 284B-parameter MoE
with approximately 13B active parameters per token and six selected experts
from 256 routed experts across 43 layers. It states that the deployed FP4
configuration requires roughly 140 GB of expert weights and presents the
interactive demonstration on a 32 GB RTX 5090-class GPU. The paper also
explains that only the active computation fits in the GPU budget while the
complete expert pool resides in host-side storage and moves through the
CPU-GPU path as needed.

These facts explain why the local 2 GiB dedicated-VRAM reading alone does not
decide feasibility, but they also show why the missing exact payload and a
measured host-memory and bandwidth budget are mandatory before claiming that
the GMKtec EVO-X2 can reproduce the paper result. Source: [FreeToken paper,
arXiv:2608.16157](https://arxiv.org/abs/2608.16157), especially the model and
hardware description in the introduction and evaluation setup.

## Evidence source

The raw values were collected from read-only `free -h`, `swapon --show
--bytes`, `rocm-smi`, DRM memory-info files, and a model-directory inventory.
No production service, model file, ROCm setting, power policy, or kernel state
was changed.
