# 4-GiB Low-VRAM Engineering Notes (Qwen3.6-35B-A3B-NVFP4 on RTX 3050)

The engineering log behind the `lowvram-4gb` branch. Everything below was measured
on the reference laptop (Ryzen 5 5600H, RTX 3050 4 GiB, 32 GiB RAM, Debian 13,
driver 610.43.02 / CUDA 13.3) unless stated otherwise.

## 1. The budget problem

`nvidia/Qwen3.6-35B-A3B-NVFP4` is a 35B-parameter MoE: 256 experts per layer, top-8
routing, 40 layers, hidden 2048 / intermediate 512 per expert, NVFP4-quantized
experts, GDN (gated-delta-net) linear attention in every other block with full
attention on the rest. The dense part of the checkpoint (attention projections,
shared expert, router, embeddings, lm_head) is ~2.65 GiB in NVFP4/BF16. On a
4096 MiB card that leaves ~1.3 GiB for everything else: KV cache, MoE prefill
buffers, GDN state, page table, flashinfer workspaces, and CUDA-graph scratch.

Three consumers dominated:

1. **embeddings** — the vocabulary table is 970 MiB in BF16, read once per token
   but resident always.
2. **KV cache** — the hybrid layout prices full-attention layers at
   2 * num_full_layers * H * heads * 2 (K+V) per token in BF16.
3. **MoE prefill buffer** — stock 0.1.2 pinned it at `2 * num_experts` slots
   regardless of what the user asked for.

## 2. The four changes and why they are safe

### 2.1 KV storage dtype (--kv-dtype)

`--kv-dtype fp8_e4m3` stores K/V in FP8 while attention compute stays BF16
(`Context.compute_dtype` is threaded separately; flashinfer receives
`data_type` for KV and `q_data_type` for the query). The pool slab is priced with
the storage dtype (`resolved_kv_dtype`), so `--num-tokens 65536` planning
arithmetic matches what is actually allocated. On the test model: 0.62 GiB for
65,536 tokens instead of ~1.24 GiB BF16. Output quality was subjectively stable
across long math/reasoning sessions; FP8-E4M3 attention K/V is a widely used
trade-off at this precision tier.

### 2.2 CPU embedding (FREETOKEN_CPU_EMBED=1)

After weights load, `embed_tokens.weight` moves to host RAM. The embedding lookup
(`index_select`) runs on CPU per token and only the resulting hidden-state row
crosses PCIe to the GPU. Frees 970 MiB. Cost: one H2D copy of H=2048 BF16 per
token (trivial vs. the MoE decode step) — measured decode stayed 10–14 tok/s.

**Incompatible with CUDA graphs**: graph capture executes the embedding lookup,
and copying from an unpinned CPU tensor during capture raises
`RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned`. Hence `--cuda-graph-max-bs 0` is mandatory with
this env var. (Making the staging buffer pinned to re-enable graphs is future
work; on this GPU the graph gain is small next to CPU-MoE decode anyway.)

### 2.3 CPU-MoE explicit cache size (the bug fix)

Stock 0.1.2 `_adjust_config`:

```python
if is_moe and config.moe_backend == "cpu":
    override("moe_cache_size", 2 * num_experts)   # 256 silently became 512
    override("moe_prefill_overlap", True)         # --disable-moe-prefill-overlap ignored
```

On a 256-expert model the requested 256-slot prefill buffer silently doubled to
512 slots (~34 MiB wasted) and the forced double-buffering re-enabled overlap
that the user explicitly disabled. This branch:

- keeps an explicit `--moe-cache-size` authoritative;
- defaults to `2*E` when overlap is on and `E` when explicitly disabled (the
  single-buffer prefill path is synchronous and fits half the slots);
- logs the resolved size and overlap instead of a stale hardcoded `2*num_experts`.

The same fix is proposed upstream against `main` (where the knob was renamed
`--moe-strategy`); this branch carries the 0.1.2-native version.

### 2.4 VRAM/weight diagnostics + workspace right-sizing

`[VRAM-DIAG]` prints device-free/torch-alloc/reserved/peaks after every init
stage: after_weights, after_moe_cache, after_kv_pool, after_linear_state,
after_page_table, after_full_init. `[WEIGHT-DIAG]` walks `named_parameters` and
prints unique GPU tensors (deduplicated by data_ptr), which is how the 970 MiB
embedding and the exact dense weight total were pinned down.

The flashinfer float workspace floor was derived from the model's TP-local
geometry and SM count instead of the flat 256 MiB, floored at 32 MiB — on the
3050 (20 SMs, 32/8 head geometry) this saves ~224 MiB with no behavioral change.

## 3. The working configuration

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

Flag-by-flag rationale:

| Flag | Why |
|---|---|
| `--kv-dtype fp8_e4m3` | halves KV bytes; 64K pool fits |
| `--moe-backend cpu` | experts stream from host RAM; decode MoE runs on 8 CPU threads |
| `--moe-cache-size 256` | one expert set of prefill slots; explicit is now honored |
| `--disable-moe-prefill-overlap` | single prefill buffer halves GPU slots; prefill is PCIe-bound anyway |
| `--cuda-graph-max-bs 0` | mandatory with CPU-embed (see 2.2) |
| `--cache-type naive` | no radix tree; GDN snapshots + page table are smaller at max_running_req=1 |
| `--max-prefill-length 1024` | GDN prefill transient scales with chunk; 16K chunks OOM below ~64 MiB free |
| `--num-tokens 65536` | the whole pool; covers input+output for long sessions |
| `--max-running-requests 1` | one deep request, not many shallow ones |
| `--expert-load serial` | avoids the parallel reader's whole-shard buffer on a low-RAM host |
| `--moe-cpu-threads 8` | saturates the pinned cores (0..3) without starving the GPU feeder thread |

## 4. Memory ledger (measured)

```
[VRAM-DIAG] after_weights    device_free=931.8 MiB   (2.65 GiB weights + embed on GPU before offload)
[VRAM-DIAG] after_moe_cache  device_free=897.8 MiB   (256-slot prefill cache + CPU executor buffers)
[VRAM-DIAG] after_kv_pool    device_free=817.8 MiB   (K+V = 0.62 GiB @ fp8_e4m3, 65536 tokens)
[VRAM-DIAG] after_linear_state device_free=757.8 MiB (GDN state pool, naive cache)
[VRAM-DIAG] after_page_table device_free=~757.8 MiB   (2 rows x 65536 x int32)
[VRAM-DIAG] after_full_init  device_free=717.8 MiB   (flashinfer workspace, sampler)
steady-state runtime free    ~570 MiB                (decode scratch + fragmentation headroom)
```

With CPU-embed active the 970 MiB table lives in host RAM (page-locked by the
expert-bank loader's serial path), leaving GPU weights at ~1.7 GiB.

## 5. Performance (measured, single request)

- Prefill: ~206–230 tok/s steady per 1024-token chunk; each request's first chunk
  runs at ~5–50 tok/s (first-touch warm-up of the serial expert reader).
- Decode: ~10–14 tok/s across 10K→65K context (expert reads dominate; PCIe +
  8 AVX2 threads on cores 0..3).
- Longest verified session: 65,349 tokens of combined input+output context,
  pool usage 1.00, no OOM, no quality collapse.

## 6. Failure catalog (all reproduced; see also README-4GB.md)

1. CUDA graph capture with CPU-embed → capture-time copy error. Fix:
   `--cuda-graph-max-bs 0`.
2. `--max-prefill-length 16384` → GDN transient OOM (`wy_fast.py new_empty(B,T,H,K)`)
   at ~32 MiB free. Fix: 1024.
3. Prompt longer than `--num-tokens` → request silently dropped (20734-token
   input against a 16384 pool). Fix: size the pool, or split the input.
4. Duplicate `ft serve` processes → port 1919 bind failures, wrong VRAM
   readings, init-time `torch.cuda.Stream()` OOMs. Fix: one server, one
   nvidia-smi logger.
5. `--moe-backend hibryd` (typo) → argparse exit 2. The valid choices: auto,
   fused, offload, cpu, hybrid.
6. `--kv-reserve-tokens` with explicit `--num-tokens`: the pool is sized by
   `--num-tokens`; the reserve only feeds the auto-solver.
7. Stock 0.1.2 CPU-MoE doubling of `--moe-cache-size` (fixed here, see 2.3).

## 7. Journey (what was tried, in order)

1. Stock 0.1.2 OOM at init on the 4-GiB card (fused backend).
2. `--moe-backend offload` + small cache: init passed, decode collapsed — the
   3050's PCIe ceiling starved expert fetches.
3. `--moe-backend cpu`: decode viable at ~10 tok/s; init still tight.
4. `[VRAM-DIAG]` built to find the remaining fat; `[WEIGHT-DIAG]` exposed the
   970 MiB embedding.
5. `FREETOKEN_CPU_EMBED=1`: +970 MiB headroom; CUDA graphs disabled by the
   capture-time copy error → `--cuda-graph-max-bs 0`.
6. `--kv-dtype fp8_e4m3`: KV halved; 64K pool fit with ~720 MiB to spare.
7. Found the `--moe-cache-size` doubling (256→512) via the after_moe_cache
   checkpoint; patched `_adjust_config`; overlap now honored too.
8. Flashinfer workspace floor right-sized 256→32 MiB (geometry-derived bound).
9. 65K-context soak test passed (65,349 tokens, usage 1.00).

## 8. Future work

- Pinned staging buffer for the CPU embedding so CUDA graphs can return
  (`--cuda-graph-max-bs` > 0) — removes per-step launch overhead.
- Port the four changes onto upstream `main` (which renamed `moe_backend` →
  `moe_strategy` and refactored `_init_offload_moe_cache`); only the cache-size
  fix is proposed upstream for now.
- Consider `--cache-type hybrid_radix` once multi-request sharing matters; naive
  keeps GDN state smallest at max_running_req=1.
