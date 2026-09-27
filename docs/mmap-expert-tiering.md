# Plan: mmap-backed expert source with hot/warm/cold tiering

## Problem

The offload MoE path reads the whole packed expert set into **pinned** host RAM at startup
(`moe/host_banks.py` + `moe/expert_banks.py`) and streams it to a GPU LRU slot cache. Because
page-locked memory cannot be reclaimed or swapped, a model whose banks exceed available RAM
cannot start (e.g. Qwen3.8-Flash-Next IQ4_XS: 60.9 GiB of banks on a 62 GiB host).

Goal: keep full expert quality while bounding RAM to the working set, by tiering experts:

```
hot   -> VRAM  (existing LRU slot cache; unchanged)
warm  -> RAM   (pinned subset, or kernel page cache over an mmap)
cold  -> disk  (mapped file, read on demand)
```

This also raises the model-size ceiling: RAM becomes the working set, not the total.

## Design

### 1. Repacked expert store (fixed stride)
Random access must be cheap. Repack each `(layer, role, expert)` into a fixed-stride store:

```
experts/
  index.json          # geometry, per-(layer,role) base offsets, expert stride, source hash
  layer-000.gate_up.bin
  layer-000.down.bin
  ...
```

- `row_bytes` is constant per role, `expert` is the leading dim, so expert `e` of role `r`,
  layer `l` is a byte range `base(l,r) + e * stride(r)` — one `pread`/`mmap` slice.
- The repack tool (`ft experts repack <gguf> --out <dir> [--drop-ple]`) streams the GGUF
  expert tensors verbatim (no dequant), so it is a copy, and it can drop the in-GGUF PLE.
- For each layer, `mmap` the role file read-only in the engine (or one big file per role).

### 2. Expert source abstraction
Introduce `moe/expert_source.py`:

```python
class ExpertSource(Protocol):
    def layer_views(self, layer: int) -> dict[str, Tensor]: ...   # gate_up/down, [E, rows, row_bytes]
    def pread_expert(self, layer: int, role: str, expert: int) -> memoryview: ...
```

Two implementations:
- `PinnedExpertSource` — today's behaviour (anonymous pinned HostBanks). Default when the
  banks fit the pin budget.
- `MmapExpertSource` — file-backed `mmap` of the store. `layer_views` returns mmap views;
  the OS page cache is the warm tier and the SSD the cold tier.

`_gguf_banks` picks the source: pinned if `_bank_bytes <= _pin_budget_bytes()`, else mmap.

### 3. Host-to-device path
`OffloadMoeCache` currently copies full/partial layers from pinned banks with a fused
`cudaMemcpyAsync` batch. For the mmap source:
- Stage through a small **pinned ring** (reuse the io_uring/O_DIRECT helpers already written
  for the PLE disk table in `moe/host_banks.py`) and `cudaMemcpyAsync` into the slot cache.
  Copying straight from pageable mmap also works but is ~2x slower; the ring hides that.
- Keep the existing coalesced batch copy for the pinned source unchanged.

### 4. Policy and prefetch
- **VRAM**: existing `--moe-cache-size` / `--moe-cache-auto` / LRU. Sizing by free VRAM stays.
- **RAM pin subset**: pin the top-`K` experts by usage into pinned buffers, `K` derived from
  the pin budget (`--expert-pin-fraction` / `--expert-pin-budget`). The rest rely on the page
  cache. This is the "auto-reap" behaviour — no experts are removed, only tiered.
- **Usage ranking**: extend the existing `--moe-collect-stats` counters (currently
  miss-rate only) to per-`(layer, expert)` activation counts, and/or add an offline pass
  `ft experts stats --model <gguf> --calib <text>` that writes a usage file consumed by
  `--expert-usage-file`. Absent a usage file, fall back to LRU (the page cache and slot cache
  already approximate it).
- **Prefetch**: `madvise(MADV_WILLNEED)` the next layer's likely experts (from the usage
  file) before the current layer's GEMM, so prefill overlaps I/O with compute.

### 5. Config / flags
- `--expert-source {auto,pinned,mmap}` (default `auto`: pinned if it fits, else mmap).
- `--expert-pin-budget <GiB>` / `--expert-pin-fraction <f>` (default: fill `_pin_budget_bytes`).
- `--expert-usage-file <path>` (from `ft experts stats`).
- `--expert-store <dir>` (default: alongside the checkpoint, or a cache dir).
- `_check_pin_budget` no longer errors for offload; it selects the mmap source instead
  (keeping the error only for `--expert-source pinned`, and for models with no store).

## Files

- `moe/expert_source.py` (new) — source protocol + pinned/mmap implementations.
- `moe/host_banks.py` — file-backed bank variant; reuse the io_uring/O_DIRECT ring helpers.
- `moe/expert_banks.py` — choose the source from the budget; expose it on `ExpertBanks`.
- `moe/gguf_experts.py` — keep the loader; add a store reader/writer used by the repack tool.
- `moe/offload_cache.py` — copy path that reads from a pageable/mmap source via the ring.
- `engine/engine.py` — `_check_pin_budget` fallback; pass the source/usage file through.
- `server/args.py` (+ `engine/config.py`) — the new flags.
- `checkpoint/` or a new `tools/` entry — `ft experts repack` and `ft experts stats`.

## Test strategy

- **Unit**: store round-trip (repack a synthetic GGUF, read every `(layer, role, expert)`,
  bytes equal to the source slice); stride/offset edge cases; usage-rank → pin-set selection.
- **Integration (CPU/one GPU)**: a tiny synthetic MoE GGUF whose banks exceed an artificially
  low `FREETOKEN_PIN_BUDGET_GB` boots with the mmap source and produces identical grouped-GEMM
  output to the pinned source (numerical equivalence).
- **Perf harness**: measure decode tok/s and VRAM-cache hit rate vs pin fraction on the real
  model, to tune the default policy.
- **Regression**: existing `moe`/`layers` suites must stay green (pinned path unchanged).

## Milestones

1. **M1 — pageable/mmap source (correctness).** Repack tool + `MmapExpertSource` + engine
   fallback; the full 512-expert IQ4_XS boots on a 62 GiB host (slow, no tuning). Proves the
   tiering works end to end.
2. **M2 — staged H2D + prefetch.** Pinned ring + `MADV_WILLNEED`; recover most of the
   pinned-source speed for the working set.
3. **M3 — usage/pin policy.** Per-expert stats, usage file, pinned warm subset, auto-reap
   behaviour; tune against tok/s and hit rate.
4. **M4 — polish.** Docs, bench-profile entries, `--expert-source pinned` error path, FTW
   compatibility (store can be written during `ft checkpoint`).

## Risks

- Page-cache reclaim/thrash if the working set approaches RAM; mitigate with the pinned warm
  subset and usage-ranked prefetch.
- H2D from pageable memory is slower than pinned; mitigated by the staging ring.
- Cold prefill reads a lot once (a long prompt can touch ~40 GB); page cache persists across
  requests, and prefetch hides some of it.
- Interaction with `cudaHostRegister` on mapped pages (register the pinned subset only).
- Correctness hinges on exact `(layer, role, expert)` addressing — covered by the round-trip
  test.
