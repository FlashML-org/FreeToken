# Install

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

## Method 2: Install from source

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).

---

## Turing GPU support (RTX 2080 Ti / sm_75)

This fork adds support for RTX 20-series (Turing, sm_75) GPUs. The following
table covers what works and what doesn't — read it before loading a model.

### Format support on sm_75

| Format | Status | Notes |
|---|---|---|
| Q4_0 GGUF + `--moe-backend hybrid` | ✅ Full support | Best mode for 22+ GB VRAM |
| BF16 + `--moe-backend hybrid` | ⚠️  Slow | Turing has no BF16 ALUs — software emulated, ~1.5× slower than FP16 |
| FP16 + `--moe-backend hybrid` | ✅ Full support | Preferred dense dtype on Turing |
| AWQ/GPTQ INT4 (Marlin stages=2) | 🚧 Planned | FP16 activation only — see issue tracker |
| NVFP4 / MXFP4 / DS-FP4 | ❌ Not supported | e2m1 compute requires sm_80+ (Ampere) |
| FP8 activation | ❌ Not supported | Requires sm_89+ (Ada) |

### Hardware requirements

| Resource | Minimum | Recommended (DeepSeek-V4-Flash) |
|---|---|---|
| VRAM | 11 GB (1× 2080 Ti) | 88 GB (4× 2080 Ti 22 GB modded) |
| System RAM | 32 GB | 192 GB+ (Q4_0 weights ≈ 145 GB) |
| Driver | r525 | r553 (CUDA 12.8 max for 2080 Ti) |
| CUDA toolkit | 12.0 | 12.8 |

### NixOS install (unstable / 26.05)

The repo ships a `flake.nix` that provides a fully wired CUDA 12.8 dev shell:

```bash
git clone https://github.com/JohnieBraaf/FreeToken_2080ti && cd FreeToken_2080ti
nix develop          # enters the CUDA 12.8 shell (gcc12, nvcc 12.8, NCCL)
bash scripts/install-sm75.sh
```

The shell sets `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0` automatically.

### Manual install (Ubuntu / other distros)

```bash
git clone https://github.com/JohnieBraaf/FreeToken_2080ti && cd FreeToken_2080ti
uv venv && source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
bash scripts/install-sm75.sh
```

### Running DeepSeek-V4-Flash on 4× 22 GB cards

```bash
# Hybrid mode: GPU-resident hot experts, CPU absorbs cold overflow.
# NVLink pairs (GPU 0+1, GPU 2+3) share experts at ~100 GB/s.
# Recommended env for 4× 2080 Ti with 2× NVLink:
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0

ft serve --model /path/to/deepseek-v4-flash-q4_0 \
         --tensor-parallel-size 4 \
         --moe-backend hybrid \
         --dtype float16 \
         --max-model-len 32768
```

For a smaller model (Qwen3-30B-A3B) that fits entirely on-GPU:

```bash
ft serve --model /path/to/Qwen3-30B-A3B-Q4_0 \
         --tensor-parallel-size 4 \
         --moe-backend offload \
         --dtype float16
```

### Expected performance (4× 2080 Ti 22 GB, validated by PocketLLM on identical hardware)

| Model | Mode | Prefill | Decode |
|---|---|---|---|
| DeepSeek-V4-Flash Q4_0 | TP4 hybrid | ~401 tok/s @ 32K–64K | ~3.7 tok/s |
| Qwen3.8-27B FP8→FP16 | TP4 GPU-resident | ~416 tok/s @ 512 tok | ~35 tok/s |

FreeToken's hybrid backend gains over a plain TP4 executor through:
- Global LRU expert cache (hot experts never re-cross PCIe)
- CUDA graph capture with `cuStreamWriteValue64` handshake (~6 ms/step saved on a 75-layer model)
- Double-buffered prefill (H2D weight streaming overlapped with GPU GEMM)
- `hybrid_fetch_fraction` adaptive split (PCIe fetch and CPU GEMV overlap perfectly)

### Known limitations on sm_75

- Triton MoE configs for the 2080 Ti are seed estimates (`num_stages=2` throughout,
  since Turing lacks `cp.async`). Run `freetoken tune-moe` on your hardware and
  replace the JSON files in `python/freetoken/moe/configs/triton_3_5_1/` with the
  profiled results for a meaningful throughput uplift.
- flashinfer cu12 AOT wheels are sm_80+ focused. The attention backend falls back
  to Triton (`--attn-backend triton`) automatically on sm_75.
- BF16 works but is emulated in software on Turing. Always use `--dtype float16`
  or `--dtype auto` (which this fork resolves to float16 on sm_75).

