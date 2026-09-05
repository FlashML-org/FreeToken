# DeepSeek-V4-Flash offload feasibility model

This document converts the measured official checkpoint size into a bounded
feasibility calculation for the GMKtec EVO-X2 Strix Halo. It is a planning
artifact only. It does not download weights, start a model, or change the
protected service.

## Inputs

The current numerical payload measurements refer to the official
`DeepSeek-V4-Flash-0731` repository. The paper names that repository as its
official checkpoint, while the current model card reports a 304B label and the
paper reports 284B. The reproduction must therefore pin the repository commit
and record both labels rather than silently treating them as interchangeable.

| Input | Value | Evidence |
|---|---:|---|
| Official `DeepSeek-V4-Flash-0731` safetensors payload | 166,886,535,336 bytes, approximately 155.43 GiB | Read-only `HEAD` measurement of all 48 shards at commit `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Routed expert pool described by paper | Approximately 140 GB | Supplied FreeToken paper |
| System memory available during live check | Approximately 18 GiB | `free -h` on the EVO-X2 |
| ROCm-reported VRAM aperture | 2 GiB | `rocm-smi` on the EVO-X2 |
| Model geometry | 43 layers, 256 routed experts, 1 shared expert, 6 routed experts active | Official `config.json` |

The 18 GiB value is `MemAvailable`, not a guaranteed model allocation. The
operating system, protected service, runtime, KV cache, scheduler, and file
cache all compete for it. The 2 GiB ROCm aperture is reported separately and
must not be added to `MemAvailable` as if it were an independent pool available
for arbitrary model storage.

## Resident-memory deficit

Even an impossible best case that devoted all 18 GiB of currently available
system memory and the full 2 GiB device aperture to weights would provide only
20 GiB of addressable working space. The current official payload would still
exceed that optimistic budget by approximately 135.43 GiB. A realistic runtime budget
is smaller because it must reserve memory for execution and KV state.

The payload-to-observed-availability ratio is approximately:

```text
155.43 GiB / 18 GiB = 8.64x
```

This is a capacity deficit, not a tuning deficit.

## Transfer lower bounds

The paper states that a prefill can move roughly 140 GB of routed expert
weights. The following are ideal lower bounds for moving that volume once. They
exclude filesystem overhead, page faults, conversion, synchronization, and
repeated expert misses.

| Sustained transfer rate | 140 GB lower bound |
|---:|---:|
| 50 GB/s | 2.80 seconds |
| 80 GB/s | 1.75 seconds |
| 100 GB/s | 1.40 seconds |
| 25 GB/s | 5.60 seconds |
| 10 GB/s | 14.00 seconds |
| 1 GB/s | 140 seconds |

Decode is more demanding than this one-time bound because it repeatedly needs
routed expert blocks. If the working set is not resident, each miss incurs
additional transfer and synchronization. The expected token rate therefore
depends on routing locality, cache size, and the actual sustained source and
destination bandwidth, not only on the 13B active-parameter count.

## Decision

The current host cannot hold the official checkpoint in memory while retaining
a usable runtime and KV cache. A swap-backed run could be attempted only as a
separate stress experiment, and its throughput and latency would need to be
reported as offload behavior. It would not establish the paper's interactive
284B result.

The correct next gate is therefore a metadata-only or tiny-slice prototype that
measures the actual layer-transfer path without downloading the full model. A
full checkpoint download is justified only if that prototype demonstrates a
sustained transfer path and a cache policy capable of keeping per-token misses
within an explicitly interactive latency budget.

## Required evidence before a full attempt

1. Exact model conversion and runtime support for the official FP4 plus FP8
   mixed format.
2. A measured layer-transfer bandwidth using a small synthetic tensor with the
   same access pattern, without altering the protected service.
3. A calculated resident budget after reserving OS, runtime, KV, and recovery
   headroom.
4. A predicted per-token miss volume and worst-case transfer latency.
5. A stop condition that prevents uncontrolled swap growth or system thrash.

