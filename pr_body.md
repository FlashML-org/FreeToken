### What

Two small, self-contained fixes found while debugging Windows/WDDM pin-quota failures with large MoE models (full repro and measurements in #55):

1. **`host_register`: clamp zero-byte registrations to 1 byte.** `cudaHostRegister` rejects a 0-byte request with `cudaErrorInvalidValue`. The NVFP4/FP8 offload loaders register per-layer scale banks whose padded size can round to zero on dense layers, which aborts the entire pin pipeline with a cryptic `cudaHostRegister failed for 0.0 GiB`. `alloc_pinned_tensor` in the same file already clamps (`nbytes == 0 ? 1 : nbytes`) — this makes `host_register` consistent with it.

2. **`HostBank.pin`: actionable error on lock-pool exhaustion.** On Windows/WDDM the pageable-locking pool is roughly half of system RAM, so registering ~17 GB of MoE banks on a 32 GB box fails with a bare driver OOM long before system RAM is exhausted. The re-raised error now carries the bank size plus a hint (free RAM / add RAM / `--moe-backend cpu`), instead of just `cudaHostRegister failed for 13.77 GiB`.

### Validation

- Repro machine: RTX 3080 Ti 12 GB / 32 GB RAM / driver 591.86 (CUDA 13.1), serving `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`.
- Before: serve aborts at `PinPipeline.wait()` with `RuntimeError: cudaHostRegister failed for 0.0 GiB`.
- With an equivalent runtime patch (clamp + graceful handling), registration proceeds normally until the WDDM pool ceiling (~13.8 GB measured) — see #55 for the instrumented numbers.
- The clamp itself is exercised by the same path (`alloc_pinned_tensor` has used the identical guard since it shipped).

### Notes

These unblock diagnosing the bigger residency gap tracked in #55 (LOCKED/PAGEABLE movement paths are scaffolded but not implemented). Not included here to keep the diff minimal.
