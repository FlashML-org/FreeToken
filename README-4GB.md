# FreeToken on a 4-GiB GPU (Qwen3.6-35B-A3B-NVFP4)

This branch (`lowvram-4gb`, based on the `v0.1.2` tag — **not** upstream `main`) runs
`nvidia/Qwen3.6-35B-A3B-NVFP4` (35B MoE, 256 experts, top-8, 40 layers, GDN
linear-attention hybrid) on a laptop RTX 3050 with 4 GiB VRAM, with a 64K-token
context pool and a single request tracked to 65,349 tokens of actual context.

## Target hardware (tested)

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, 4096 MiB |
| CPU | AMD Ryzen 5 5600H (6C/12T) |
| RAM | 32 GiB |
| OS | Debian 13 (trixie), driver 610.43.02, CUDA 13.3 |

## What the four changes do

1. **`--kv-dtype fp8_e4m3`** — stores the KV cache in FP8 (compute stays BF16). The
   65,536-token pool costs 0.62 GiB instead of ~1.24 GiB. Also threads a separate
   query dtype through the flashinfer backend and prices the pool slab with the
   storage dtype.
2. **`FREETOKEN_CPU_EMBED=1`** — moves the 970 MiB `embed_tokens` table to host RAM;
   lookup runs on CPU and only the (hidden-size) activations cross the bus.
3. **CPU-MoE cache-size fix** — stock 0.1.2 unconditionally overrode an explicit
   `--moe-cache-size` to `2 * num_experts` (256 slots silently became 512) and forced
   prefill overlap back on. Now an explicit size is authoritative, and the default
   respects `--disable-moe-prefill-overlap` (`2*E` with overlap, `E` without).
4. **Diagnostics** — `[VRAM-DIAG]` checkpoints at every init stage plus a
   `[WEIGHT-DIAG]` unique-GPU-tensor report; the flashinfer workspace floor is also
   right-sized (32 MiB instead of 256 MiB where geometry allows), saving ~224 MiB.

## Serve command

```bash
export FREETOKEN_CPU_EMBED=1
ft serve \
  --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --kv-dtype fp8_e4m3 \
  --nvfp4-backend auto \
  --moe-backend cpu \
  --moe-cache-size 256 \
  --expert-load serial \
  --disable-moe-prefill-overlap \
  --cuda-graph-max-bs 0 \
  --cache-type naive \
  --max-prefill-length 1024 \
  --num-tokens 65536 \
  --max-running-requests 1 \
  --moe-cpu-threads 8 \
  --max-output-tokens 32768
```

### Light 16K variant

```bash
ft serve ... --kv-reserve-tokens 16384 --max-seq-len-override 16384 \
  --num-tokens 16384 --max-output-tokens 8192
```

## Measured results (65,536-token KV pool, cpu MoE, cache 256, fp8_e4m3 KV)

VRAM ledger (device-free MiB after each init stage):

```
after_weights 931.8 -> after_moe_cache 897.8 -> after_kv_pool 817.8
-> after_linear_state 757.8 -> after_full_init 717.8   (runtime settles ~570)
```

- Prefill: ~206–230 tok/s in 1024-token chunks (first chunk of a request is warm-up)
- Decode: ~10–14 tok/s sustained from 10K to 65K context
- Weights: ~2.65 GiB on GPU; experts served from host RAM by 8 pinned CPU threads
  (AVX2, NVFP4 format)

## Mandatory flag interactions (crash catalog)

- `FREETOKEN_CPU_EMBED=1` **requires** `--cuda-graph-max-bs 0`: CPU tensors copied
  during graph capture raise `RuntimeError: Cannot copy between CPU and CUDA tensors
  during CUDA graph capture unless the CPU tensor is pinned`.
- Keep `--max-prefill-length 1024`: 16K chunks OOM in the GDN transient
  (`new_empty(B,T,H,K)`) when free VRAM is tens of MiB.
- A request longer than `--num-tokens` is silently dropped — size the pool for your
  longest input+output (65,536 covers 20734-token inputs plus 32K output with room).
- Run exactly one server: leftover `ft serve` processes cause port-1919 clashes,
  wrong VRAM readings, and spurious init OOMs.
- `--kv-reserve-tokens` does not cap the pool when `--num-tokens` is explicit.
- `FREETOKEN_FI_WORKSPACE_MB` / `TRITON_BENCH_CACHE_MB` do nothing on 0.1.2 (no
  references in the codebase); this branch's flashinfer workspace sizing is built in.

## Installing this branch

```bash
git clone -b lowvram-4gb https://github.com/BenMohStem/FreeToken.git
cd FreeToken
uv pip install -e ".[accel]"
```

See [docs/lowvram-4gb-notes.md](docs/lowvram-4gb-notes.md) for the full engineering
notes, memory arithmetic, and the failure catalog behind every flag.
