### Environment

- FreeToken 0.1.1 (Windows wheel from `FreeToken-Web@beta`) and cross-checked against `main` (0.1.2)
- Windows 11 + WSL2-free native run, RTX 3080 Ti 12 GB, driver 591.86 (CUDA 13.1), 32 GB RAM
- Model: `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` (36 B MoE, qwen3_5_moe arch, ~17 GB of host bank sources)

### Summary

On a 32 GB Windows machine, serving any MoE model whose total bank sources exceed ~14 GB is currently impossible through the offload backend, because:

1. the NVFP4/FP8 source banks are allocated unpinned (`alloc_layer_banks` → lazy mmaps) and only mapped later via `HostBank.pin()` → `cudaHostRegister`;
2. WDDM enforces a pageable-locking pool of roughly half of system RAM (~13.8 GB measured here), so registering all banks raises `cudaHostRegister failed: out of memory` partway through the pipeline;
3. that failure aborts the whole serve with `RuntimeError: cudaHostRegister failed for X GiB` at `PinPipeline.wait()`, and `_build_copy_plan` then has no usable device pointers;
4. the designed escape hatch for exactly this situation (LOCKED/PAGEABLE residency served by the CPU executor) is scaffolded but not implemented:
   - `HostResidency` docstring: *"their movement paths are not implemented here"* (`moe/host_banks.py`, `HostResidency`)
   - `HostBank.lock()` raises `NotImplementedError`
   - `OffloadMoeCache.set_bank_sources` raises `NotImplementedError` for any non-PINNED layer (`moe/offload_cache.py`)

This matches the on-machine observation in the tweet thread where FreeToken was announced ("windows supports dma, we can pin half of the memory") — half of RAM is simply not enough for 35 B-class models on 32 GB boxes.

### Repro

```
ft serve --model D:\models\Ornith-1.5-35B-A3B-NVFP4 --expert-load parallel
```

→ `RuntimeError: cudaHostRegister failed ... out of memory` once cumulative registrations reach ~13.8 GB. With an instrumented build that catches the per-bank failures, we measured across several runs:

- device_ptr failures: 240 tensors / 16.93 GB of bank sources need mapped pointers
- successful in-place `cudaHostRegister`: ceiling consistently ~13.77–13.82 GB, then OOM
- every unmapped source left in the copy plan poisons execution later (`CUDA error: invalid argument` surfacing asynchronously at KV-pool init)

### Additional smaller issues found along the way

1. **Zero-size registrations**: `host_register` rejects a 0-byte request (`cudaErrorInvalidValue`); small scale banks whose padded size rounds to zero trip this. `alloc_pinned_tensor` already clamps (`nbytes == 0 ? 1 : nbytes`); `host_register` does not.
2. **KV pool sizing ignores `--max-seq-len-override`**: with a 16 K override the pool still requests 146 450 pages (~2.8 GB bf16). `--num-tokens` works as an escape hatch.
3. **`--moe-cpu-layers` does not reduce bank/copy-plan membership** (tested with explicit id lists and counts): all layers' sources still go through `set_bank_sources`/`device_ptr`, so the flag cannot be used to fit under the quota today.

### Proposed direction

Implement the missing LOCKED/PAGEABLE movement paths for Windows:

- `HostBank.lock()` via `VirtualLock` (+ `SetProcessWorkingSetSize` to raise the working-set floor); Linux via `mlock`
- graceful per-bank fallback inside `PinPipeline._run`: on register failure mark the bank LOCKED/PAGEABLE instead of failing the drain
- plumb real `layer_residency` from the loaders into `ExpertBanks` → `set_bank_sources` (accepting LOCKED) and route non-PINNED layers to the CPU executor, reusing the existing `--moe-cpu-layers` routing

We validated the mechanics end-to-end with a runtime harness on the affected machine: in-place registration up to quota succeeds, VirtualLock-based locking of the remaining banks keeps them resident, and the CPU-executor route initializes cleanly (`--moe-backend cpu` completes init with zero copy-plan entries). Happy to contribute the implementation if the approach sounds right.
