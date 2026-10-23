# Speeding up prefill with offloaded experts

This guide applies to the `offload` and `hybrid` MoE strategies (see
[models.md](../models.md#moe-strategies)). With `fused` every expert is already on
the GPU and none of this applies.

Prefill speed has two sides: **copy** (moving experts over PCIe) and
**compute** (the forward pass). With overlap on, a chunk takes as long as the
slower of the two, so first find which one is the bottleneck, then work on
that one. Some fixes are flags; others need development work. The
[summary](#summary) at the end separates them.

## How a chunk spends its time

Experts live in host RAM. During prefill each MoE layer copies all of its
experts from host to GPU, for every chunk, however short (a large chunk routes
to nearly every expert anyway). For a chunk of `T` tokens
(`--max-prefill-length`, default 8192):

- **copy** depends on the expert bytes and the PCIe bandwidth
  (`ft bench bw` measures it), not on `T`. On PCIe 5.0, contiguous copies of
  16 MiB or more already reach the full ~55.6 GB/s, and the per-layer expert
  copies are far larger, so copy time per chunk ≈ total expert bytes /
  bandwidth.
- **compute** grows with `T`, at a rate set by the model implementation and
  the quantization method (which kernels run, and how well they use the GPU).

The two run in one of two modes:

- **Serial**: each layer copies its experts, then computes.
  Chunk time = copy + compute.
- **Overlap** (the default): the copy of layer N+1 runs while layer N
  computes. Chunk time ≈ max(copy, compute).

## Step 1: make sure overlap is on

Overlap stages layers in two buffers carved out of the GPU expert cache, each
holding one whole layer of experts. If the cache cannot hold both, the server
logs at startup:

```
WARNING  Disabling MoE prefill overlap: the slot cache cannot hold two whole layers of experts
```

and every prefill runs serial, paying copy + compute on every chunk. Look for
this warning in the startup log.

- Raise `--moe-cache-size` (or `--moe-cache-mem`) until the warning goes away.
  The extra slots come out of the VRAM left for KV and activations.
- Overlap is also off with `--disable-moe-prefill-overlap`, and when some
  layers are OS-locked instead of pinned, on hosts that cap pinned memory (see
  [models.md](../models.md#hosts-that-cap-pinned-memory-wsl)): locked layers
  can only be copied synchronously. A `--moe-cache-size` or `--moe-cache-rate`
  below two layers of experts fails at startup unless overlap is disabled.

Each buffer is sized for a whole layer at the widest expert format of any
layer, while most cache slots may hold a narrower format. A checkpoint that
mixes expert formats (for example unsloth's UD GGUF tiers) therefore needs
more than `2 x experts per layer` slots. Example: GLM-5.3-Flash UD-IQ4_XS has
288 experts per layer; its `ffn_down` experts are IQ4_XS in 39 MoE layers and
Q6_K in 3. The buffers are sized for Q6_K, so overlap needs about 856 slots
instead of 576. At 800 slots it is off.

## Step 2: find the bottleneck

With overlap on, raise `--max-prefill-length` and watch the prefill
`input throughput (token/s)` in the server log:

- It grows about in proportion to the chunk: **copy-bound**. The GPU waits
  for PCIe, and faster kernels will not help.
- It barely moves: **compute-bound**. The copy is already hidden, and a
  larger chunk will not help.

Throughput only hints at the bottleneck; to confirm it, profile a prefill
chunk with Nsight Systems (`nsys profile -t cuda,nvtx -o prefill ft serve ...`,
then send one long request). With overlap (and without
`--moe-prefill-hit-d2d`), each layer's expert copy is a `Memcpy HtoD` on its
own stream, next to the layer's kernels on the compute stream:

- Copy-bound: the HtoD copies run back to back and the compute stream idles
  between layers, waiting for the next layer's copy.
- Compute-bound: each copy finishes before the layer that needs it starts,
  and the copy stream idles.
- No overlap: each layer's copy runs on the compute stream (as a gather
  kernel or a memcpy) before the layer's kernels, never alongside them.

## Copy-bound: cut the PCIe time

1. **Raise the chunk size** (`--max-prefill-length`). The copy is the same
   per chunk, so a larger chunk gives compute more work to hide it behind,
   until compute catches up with the copy. A larger chunk also needs more
   activation memory, see [Activation memory](#activation-memory).
2. **Skip experts already on the GPU** (`--moe-prefill-hit-d2d`, CUDA >= 13).
   Experts already in the cache are copied on the GPU and only the misses
   cross PCIe. The gain grows with the share of experts the cache holds, so it
   matters most with a large cache (high `--moe-cache-rate`). It needs overlap
   and a cache larger than two layers; otherwise it falls back to the full
   copy and logs why once.
3. If the experts fit on the GPU, `--moe-strategy fused` removes the copy
   entirely.

## Compute-bound: speed up the forward pass

At about 8k tokens per chunk the large GEMMs already keep the GPU busy, so a
larger chunk adds little compute throughput. If it still does, the kernels
underuse the GPU at 8k, and that is a kernel problem, not a flag problem.

Compute-bound prefill needs kernel work (profile one chunk with Nsight
Systems first):

- **Quantized GEMMs on native tensor cores.** Check that the large quantized
  GEMMs use the GPU's native low-precision MMA (FP8, FP4) instead of
  dequantizing to bf16 or running a generic fallback.
- **Fused small ops.** Check that norms, activations, routing and similar
  small ops run as fused kernels rather than as chains of separate torch ops,
  each with its own launch and memory round trip.

A kernel speedup lowers compute, so it also raises the chunk size at which
compute hides the copy. After one, the chunk may need to grow again to stay
compute-bound.

## Activation memory

Prefill activations grow with the chunk size, but startup sizing does not
look at the chunk size: weights, expert cache and KV take their share of VRAM
(`--memory-ratio`, or a fixed `--num-tokens`), and prefill activations must
fit in what is left. A chunk that does not fit shows up as allocator warnings
during prefill:

```
expandable_segments: memory mapping failed with OOM
```

and in the worst case as an out-of-memory error.

- With flags: lower `--max-prefill-length`, or lower `--num-tokens` (or
  `--memory-ratio`) to leave more room for activations.
- With development: find what holds the peak. Record one chunk with
  `torch.cuda.memory._record_memory_history()`, save it with
  `torch.cuda.memory._dump_snapshot("prefill.pickle")`, and open the file as a
  flame graph at <https://pytorch.org/memory_viz>. Look for tensors copied
  more often than needed (for example a `.contiguous()` or `torch.cat` of a
  large activation that the next op does not require) and for large
  temporaries that outlive their use.

## Summary

| Situation | With flags | Needs development |
|---|---|---|
| Overlap off (startup warning) | raise `--moe-cache-size` / `--moe-cache-mem` | size the staging buffers for mixed-format checkpoints without the extra slots |
| Copy-bound | raise `--max-prefill-length`; `--moe-prefill-hit-d2d` with a large cache; `fused` if the experts fit | — |
| Compute-bound | — | native tensor-core quantized GEMMs, fused small ops |
| Larger chunk runs out of memory | lower `--max-prefill-length`, `--num-tokens` or `--memory-ratio` | memory snapshot, remove extra activation copies |
