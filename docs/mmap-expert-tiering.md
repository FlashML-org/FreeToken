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
  Preallocate each output file (`posix_fallocate`) and write sequentially so the store lands
  in contiguous extents.
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
- Stage through a small **pinned ring** and `cudaMemcpyAsync` into the slot cache. Copying
  straight from pageable mmap also works but is ~2x slower; the ring hides that.
- **The ring must be filled through the page cache, not around it.** O_DIRECT reads bypass
  the page cache, so filling the ring that way would leave the warm tier unused and make
  `MADV_WILLNEED` prefetch pointless. Default ring fill is a CPU copy from the mmap view into
  the ring slot. The existing io_uring/O_DIRECT helpers may still serve as a cold-path option
  for non-resident pages (check residency with `mincore`) if benchmarking shows they beat a
  faulting copy.
- Call `madvise(MADV_RANDOM)` on the mapped expert files — access within a layer is random,
  and default readahead would pull adjacent experts into the page cache that no routing
  asked for. (`MADV_WILLNEED` on an explicit range in §4 still triggers readahead for exactly
  that range, so the two compose.)
- Keep the existing coalesced batch copy for the pinned source unchanged.

### 4. Policy and prefetch
- **VRAM**: existing `--moe-cache-size` / `--moe-cache-auto` / LRU. Sizing by free VRAM stays.
- **RAM pin subset**: pin the top-`K` experts by usage into pinned buffers, `K` derived from
  the pin budget (`--expert-pin-fraction` / `--expert-pin-budget`). Allocate the budget
  **per layer**, not as one flat global ranking: routing specialization skews by depth, so a
  global top-K starves "boring" layers that still route uniformly (and are hit every token)
  in favor of a few peaky layers. Give each layer a floor (or an even split of the budget)
  before spending any remainder on a global ranking across layers. The rest of each layer's
  experts rely on the page cache. This is the "auto-reap" behaviour — no experts are removed,
  only tiered.
- **Usage ranking**: extend the existing `--moe-collect-stats` counters (currently
  miss-rate only) to per-`(layer, expert)` activation counts, and/or add an offline pass
  `ft experts stats --model <gguf> --calib <text>` that writes a usage file consumed by
  `--expert-usage-file`. Absent a usage file, fall back to LRU (the page cache and slot cache
  already approximate it).
- **Prefetch**: `madvise(MADV_WILLNEED)` the next layer's likely experts (from the usage
  file) before the current layer's GEMM, so prefill overlaps I/O with compute.

### 5. Config / flags
- `--expert-source {auto,pinned,mmap}` (default `auto`: pinned if it fits, else mmap).
- `--expert-pin-budget <GiB>` / `--expert-pin-fraction <f>` (default: fill
  `_pin_budget_bytes() - ring_bytes`; the staging ring is pinned, so carve it out before
  sizing the top-K subset).
- `--expert-usage-file <path>` (from `ft experts stats`).
- `--expert-store <dir>` (default: alongside the checkpoint, or a cache dir).
- `_check_pin_budget` no longer errors for offload; it selects the mmap source instead
  (keeping the error only for `--expert-source pinned`, and for models with no store).

## Files

- `moe/expert_source.py` (new) — source protocol + pinned/mmap implementations.
- `moe/host_banks.py` — file-backed bank variant; reuse the io_uring ring helpers.
- `moe/expert_banks.py` — choose the source from the budget; expose it on `ExpertBanks`.
- `moe/gguf_experts.py` — keep the loader; add a store reader/writer used by the repack tool.
- `moe/offload_cache.py` — copy path that reads from a pageable/mmap source via the ring.
- `engine/engine.py` — `_check_pin_budget` fallback; pass the source/usage file through.
- `server/args.py` (+ `engine/config.py`) — the new flags.
- `checkpoint/` or a new `tools/` entry — `ft experts repack` and `ft experts stats`.

## Test strategy

- **Unit**: store round-trip (repack a synthetic GGUF, read every `(layer, role, expert)`,
  bytes equal to the source slice); stride/offset edge cases; usage-rank → pin-set selection
  (including that the per-layer floor is respected, not just the global ranking).
- **Integration (CPU/one GPU)**: a tiny synthetic MoE GGUF whose banks exceed an artificially
  low `FREETOKEN_PIN_BUDGET_GB` boots with the mmap source and produces identical grouped-GEMM
  output to the pinned source (numerical equivalence).
- **Perf harness**: measure decode tok/s and VRAM-cache hit rate vs pin fraction on the real
  model, to tune the default policy. Break the RAM tier down **per layer** into three
  separate counters — pinned-subset hit rate, page-cache (warm) hit rate, and cold read
  count — rather than one model-wide average, so a layer starved by allocation policy can't
  hide inside a healthy mean.
- **Regression**: existing `moe`/`layers` suites must stay green (pinned path unchanged).

## Milestones

1. **M1 — pageable/mmap source (correctness).** Repack tool + `MmapExpertSource` + engine
   fallback; the full 512-expert IQ4_XS boots on a 62 GiB host (slow, no tuning). Proves the
   tiering works end to end.
2. **M2 — staged H2D + prefetch.** Pinned ring + `MADV_WILLNEED`; recover most of the
   pinned-source speed for the working set. Benchmark ring fill via mmap copy vs O_DIRECT
   in both regimes — cold first read, and repeat reads of the same expert (the mmap copy
   leaves pages resident; an O_DIRECT read never does) — before committing to a default;
   they win in opposite regimes, so a single-axis benchmark picks the wrong default.
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
- Pin quality is bounded by how well the `--calib` corpus matches actual deployment traffic
  (code-heavy vs. chat-heavy calibration ranks experts differently); v1 has no automatic
  detection of calibration/production drift, so a mismatch shows up only as a worse hit rate,
  not an error. Deferred follow-up if per-layer telemetry shows real divergence from
  calibration: online adaptive re-pinning — deliberately out of scope for v1.
