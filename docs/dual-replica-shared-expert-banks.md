# Dual-replica shared expert-bank experiment

This is a field report for the multi-GPU work tracked in
[the 2026 roadmap](https://github.com/FlashML-org/FreeToken/issues/79). It is not a
supported FreeToken configuration or a tensor-parallel implementation.

The experiment ran two independent TP1 servers, one on each GPU. Each server kept
its own scheduler, KV cache, CUDA graphs, and GPU expert cache. The processes read
one immutable set of host expert-bank pages through shared mappings instead of
allocating two physical copies. This gives two independent request lanes while
keeping the host-memory cost of the large expert banks close to one replica.

## Test system

| Component | Configuration |
| --- | --- |
| FreeToken | performance run: `af71ba43206e124f5ff6419b47ee36c6e9981078` plus NVIDIA-checkpoint compatibility changes and an external shared-bank shim; attachment rechecked on `953565667f3141c90d0f0eb469bb2655d2407140` |
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| GPU | 2 x NVIDIA GeForce RTX 5090 D, 32 GB each |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads |
| Host memory | 96 GB installed, about 89 GiB visible to Linux |
| NVIDIA driver | 595.84 |
| Per-replica limit | one running request, 262,144-token maximum sequence length |
| PLE | disk backend |
| MoE | NVFP4 Triton offload backend, automatic GPU expert cache |
| MTP | disabled for the measurements below |

The normalized serve command for each replica was:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
FT_SHARED_ROLE=<owner-or-follower> \
FT_SHARED_ROOT=<private-runtime-directory> \
PYTHONPATH=<shared-bank-shim>:python \
python -m freetoken \
  --model-path <nvidia-Qwen3.8-Flash-Next-NVFP4> \
  --host 127.0.0.1 \
  --port <non-overlapping-port> \
  --ple-backend disk \
  --moe-backend offload \
  --nvfp4-backend triton \
  --expert-load serial \
  --moe-cache-auto \
  --kv-reserve-tokens 295168 \
  --max-seq-len-override 262144 \
  --max-running-requests 1 \
  --cuda-graph-max-bs 2 \
  --memory-ratio 0.90 \
  --page-size 64 \
  --attention-backend qsa_sparse \
  --max-prefill-length 2048 \
  --max-output-tokens 16384 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

The ports must leave room for FreeToken's adjacent internal listener. The first
full-size attempt used adjacent API ports and stopped on an address collision. It
did not fail because of host or GPU memory.

## Shared-bank protocol

The proof of concept used an owner/follower protocol outside the FreeToken source:

1. The owner replaced anonymous `HostBank` allocations with page-aligned,
   file-descriptor-backed `MAP_SHARED` mappings while the regular expert loader
   filled and registered them.
2. After loading, the owner atomically published a versioned manifest containing
   the owner PID, resolved checkpoint path, quantization format, and each tensor's file
   descriptor, mapping size, offset, shape, dtype, byte length, and sample hash.
3. The follower checked the protocol version, resolved checkpoint path, NVFP4
   format, backing-file size, and sample hashes. Opening descriptors through the
   owner's procfs entry also failed closed when the owner was absent. It mapped the
   same pages, registered them with CUDA in its own process, and built an
   `ExpertBanks` view without reading the expert checkpoint again.
4. A supervisor started the owner first, waited for the manifest, attached and
   checked the follower, then exposed both servers through a request router. It
   stopped the follower before the owner so the backing descriptors remained live.

The runtime directory was mode `0700` and shared only between processes of the same
Unix user. A production implementation should avoid trusting a writable JSON
manifest or exposing resolved local paths in it, authenticate the owner and
checkpoint more strongly, validate every
integer before mapping, use descriptor passing or sealed backing files, and define
recovery and cleanup behavior when either process exits unexpectedly.

## Results

The physical expert-bank allocation was 68,136,468,480 bytes (63.457 GiB) across
288 mappings. Both workers mapped the same 288 backing objects. Linux reported
about 31.75 GiB `Pss_Shmem` per worker, which is consistent with one set of physical
pages shared by two processes; summing RSS would incorrectly count those pages
twice. Sample hashes from the beginning, middle, and end of every tensor matched
after load and again after the concurrent long-context run. This was a sampling
check, not a byte-for-byte verification of all weights.

| Measurement | Result |
| --- | ---: |
| Process-group memory before requests | 70.858 GiB |
| Process-group memory after requests | 71.303 GiB |
| Lowest sampled host `MemAvailable` | 13.01 GiB |
| GPU memory after requests | 29,714 / 29,664 MiB |
| Concurrent short-run aggregate, repetition 1 | 60.68 output tokens/s |
| Concurrent short-run aggregate, repetition 2 | 62.02 output tokens/s |
| Per-lane streamed decode estimate | 43.26-47.93 tokens/s |
| Concurrent long inputs | 258,095 input + 38 output tokens per lane |
| Long-input wall time | 319.659 / 319.684 seconds |

Each long prompt contained three markers at different depths; both replicas returned
all markers correctly. The short run used two small Python-generation prompts with
thinking disabled, temperature zero, and a 256-token output limit. Arithmetic and
tool-call shape probes also passed, but generated code was not executed. These
checks establish capacity and transport behavior, not general model quality or
production reliability.

The experiment ran under an 80 GiB memory cgroup with swap disabled. It reached the
cgroup limit and reclaimed file cache, but recorded zero `oom` and `oom_kill`
events. The host had unrelated swap use, so this does not establish a system-wide
zero-swap result.

The controlled performance and long-context measurements used the earlier base
commit in the table plus two local changes that made the NVIDIA mixed-precision
checkpoint load correctly. The shared attachment, 63.457 GiB physical-bank size,
288 mappings, sample checks, two healthy TP1 replicas, and 262,144-token limits were
rechecked on the later upstream commit. No new benchmark numbers are attributed to
that recheck.

## What did not pass

- An appended-turn prefix test reported zero cached tokens and repeated the full
  prefill. The test was stopped; shared expert banks were not established as the
  cause.
- Some child processes did not exit within the 45-second service stop window. The
  service manager timed out and removed the process group. Graceful crash recovery
  and shared-object lifetime remain unresolved.
- The first full-size launch exposed the adjacent-port collision described above.
- Hashing sampled regions detects common attachment and ordering failures, but it is
  not a complete integrity proof.

## Implications for an upstream design

This topology is data-parallel serving with shared immutable CPU weight storage. It
does not make one request use two GPUs and should remain separate from tensor
parallelism. The useful upstream primitive would be an explicit shared host-bank
owner with lifecycle supervision, not implicit sharing through Linux RSS behavior.

An upstream implementation would also need to decide whether sharing belongs below
`ExpertBanks`, in a multi-replica launcher, or in an external serving layer. The
proof of concept suggests the bank layer is sufficient for memory sharing, while
the scheduler and every GPU-resident object can stay replica-local. It also shows
that correctness, prefix reuse, port allocation, and shutdown need independent
acceptance gates; successful attachment alone is not a production-ready result.
