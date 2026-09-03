"""Expert-bank load benchmark: baseline vs parallel vs ftw, all through the FRAMEWORK.

Measures the dominant startup cost -- loading the offload expert banks -- via the real
``freetoken`` code paths, with NO bench-local reimplementations of the readers:

  baseline   load_expert_banks(...)                 serial loader (scattered read + pin)
  parallel         load_expert_banks(..., parallel=True)   chunked multi-threaded O_DIRECT read of
                                                     the ORIGINAL checkpoint, no repack
  ftw         load_expert_banks(<ftw_dir>)          read the post-repack banks from the
             unified FTW checkpoint -- i.e. freetoken.checkpoint.load_ftw_banks -- after
             a one-time convert_checkpoint(model -> ftw_dir) (build cost reported apart)

All three return the SAME ``ExpertBanks.sources``; checksums must match baseline. Each mode
runs in its OWN subprocess so its ~bank-sized pinned memory is reclaimed before the next
(two full bank sets at once would OOM).

The FTW is the framework's unified WHOLE-MODEL ftw format (dense weights + experts); this
bench times only the expert-bank portion so baseline / parallel / ftw stay directly comparable.

By default each mode EVICTS the page cache for its files first (fadvise DONTNEED) so every
read measures real disk -- baseline (mmap) would otherwise hit a warm cache and look
artificially fast next to the O_DIRECT parallel/ftw. Pass --no-drop-cache for a warm comparison.

Run:
  PYTHONPATH=python python benchmarks/bench_load_weight_generic.py \
      --model /path/to/DeepSeek-V4-Flash
  ... --modes parallel,ftw            # subset of modes (baseline always runs as the reference)
  ... --keep-ftw            # keep (and reuse) the converted FTW dir across runs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import torch

GiB = float(1 << 30)
ALL_MODES = ("parallel", "ftw")


def _default_ftw_dir(model_path: str) -> str:
    """Scratch dir for the converted FTW. On /var/tmp (big disk, and not swept by /tmp
    cleaners) and MODEL-SPECIFIC so two concurrent benches on different models don't clobber
    each other's in-progress FTW (a fixed shared /tmp path did exactly that). The parent
    picks it once and passes it to every worker via --ftw-dir; override with --ftw-dir."""
    return os.path.join("/var/tmp", "ftw_" + os.path.basename(os.path.normpath(model_path)))


# ---------------- memory sampling + checksum ----------------
def _meminfo_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _status_kb(key: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    return 0


class MemSampler(threading.Thread):
    """Background sampler: peak process RSS + system MemAvailable low-water-mark."""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.min_avail = 1 << 62
        self.max_rss = 0

    def run(self) -> None:
        while not self._stop_evt.wait(self.interval):
            self.min_avail = min(self.min_avail, _meminfo_available_bytes())
            self.max_rss = max(self.max_rss, _status_kb("VmRSS:"))

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=1.0)


def _bank_tensors(v) -> list:
    """Normalize a bank's value to a list of tensors: the per-layer host bank contract
    makes ``ExpertBanks.sources[name]`` a ``list[Tensor]`` (one per layer), but not every
    loader path is on that contract yet -- accept a bare flat Tensor too."""
    return v if isinstance(v, list) else [v]


_HASH_WEIGHT = 1_000_000_007  # keeps position weights (and so the products) inside int64


def _checksum(banks: dict) -> dict:
    """Position-weighted sample hash per bank -- cheap, deterministic, order-sensitive.

    Samples ~1M bytes strided across the bank and folds each one together with its GLOBAL
    byte offset. Weighting by position is what makes a reordering visible: a plain sum is
    permutation-invariant, and reordering experts is precisely what the parallel and FTW
    readers could get wrong. Offsets are global across the per-layer tensors, so the hash
    is independent of how a bank happens to be chunked (one flat [L*E, ...] tensor vs. a
    per-layer list) -- computed without materialising a concatenated copy, which would
    otherwise add a full bank to the peak RSS this bench reports.

    Still a sample, not a proof: at these sizes the stride is tens of KiB, so damage
    confined to less than that can slip through.
    """
    out = {}
    for n in sorted(banks):
        parts = [t.reshape(-1).view(torch.uint8) for t in _bank_tensors(banks[n])]
        total = sum(p.numel() for p in parts)
        step = max(1, total // 1_000_000)
        h, offset = 0, 0
        for p in parts:
            first = (-offset) % step  # first strided sample that lands inside this part
            if first < p.numel():
                sample = p[first::step].to(torch.int64)
                pos = torch.arange(
                    offset + first, offset + p.numel(), step, dtype=torch.int64
                )
                h += int((sample * (pos % _HASH_WEIGHT + 1)).sum().item())
            offset += p.numel()
        out[n] = h
    return out


# ---------------- subprocess workers (framework code only) ----------------
def _model_config(model_path: str):
    from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
    from freetoken.engine.config import EngineConfig

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.gpu_select import bind_assigned_gpu

    dev = bind_assigned_gpu()
    torch.zeros(1, device=dev)  # init CUDA context (pinning / nvfp4 backend pick)
    cfg = EngineConfig(model_path=model_path, tp_info=DistributedInfo(0, 1),
                       dtype=torch.bfloat16, moe_backend="offload")
    return cfg.model_config


def _evict_cache(model_path: str) -> int:
    """Drop the page cache for this dir's weight files (fadvise DONTNEED, no root needed) so
    a read measures real disk, not RAM. Needed for baseline (mmap can hit warm cache);
    parallel/ftw use O_DIRECT and are cold anyway, but evicting also frees the RAM baseline left."""
    n = 0
    for name in os.listdir(model_path):
        if name.endswith((".safetensors", ".ftw", ".gguf")):
            try:
                fd = os.open(os.path.join(model_path, name), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
    return n


def _bench_load(mode: str, model_path: str, *, parallel: bool, workers: int, chunk: int,
                drop_cache: bool = True) -> None:
    """Time load_expert_banks end-to-end (alloc + read + pin) and checksum the sources."""
    from freetoken.moe.expert_banks import load_expert_banks

    mc = _model_config(model_path)
    if drop_cache:
        _evict_cache(model_path)  # cold read: fair vs O_DIRECT modes
    s = MemSampler()
    s.start()
    t = time.perf_counter()
    try:
        banks = load_expert_banks(model_path, mc, device=torch.device("cuda", torch.cuda.current_device()),
                                  dtype=torch.bfloat16, parallel=parallel,
                                  workers=workers, chunk=chunk)
    except NotImplementedError as e:
        s.stop()
        print("@@RESULT@@" + json.dumps({"mode": mode, "skipped": str(e)}))
        return
    load_s = time.perf_counter() - t
    ck = _checksum(banks.sources)
    total = sum(bt.numel() * bt.element_size() for v in banks.sources.values() for bt in _bank_tensors(v))
    s.stop()
    print("@@RESULT@@" + json.dumps({
        "mode": mode, "load_s": round(load_s, 2), "gib": round(total / GiB, 2),
        "gibps": round(total / GiB / load_s, 2) if load_s else 0,
        "peak_rss_gib": round(s.max_rss / GiB, 1), "n_banks": len(banks.sources),
        "quant_format": banks.quant_format, "checksum": ck}))


def worker_baseline(ns):
    _bench_load("baseline", ns.model, parallel=False, workers=ns.workers,
                chunk=ns.chunk_mib << 20, drop_cache=not ns.no_drop_cache)


def worker_parallel(ns):
    _bench_load("parallel", ns.model, parallel=True, workers=ns.workers,
                chunk=ns.chunk_mib << 20, drop_cache=not ns.no_drop_cache)


def worker_ftw(ns):
    # FTW auto-detected by path -> load_expert_banks routes to load_ftw_banks
    _bench_load("ftw", ns.ftw_dir, parallel=False, workers=ns.workers,
                chunk=ns.chunk_mib << 20, drop_cache=not ns.no_drop_cache)


def worker_build(ns):
    """One-time offline convert: HF safetensors -> FTW checkpoint (whole model)."""
    from freetoken.checkpoint.convert import convert_checkpoint

    shutil.rmtree(ns.ftw_dir, ignore_errors=True)
    shard_limit = int(ns.shard_gib * (1 << 30))
    shard_limit -= shard_limit % 4096
    s = MemSampler()
    s.start()
    t = time.perf_counter()
    dev = f"cuda:{torch.cuda.current_device()}" if ns.gpu else None
    idx = convert_checkpoint(ns.model, ns.ftw_dir, moe_backend="offload", shard_limit=shard_limit, device=dev)
    build_s = time.perf_counter() - t
    s.stop()
    print("@@RESULT@@" + json.dumps({
        "mode": "build", "build_s": round(build_s, 2), "gib": round(idx["total_bytes"] / GiB, 2),
        "shards": len(idx["shards"]), "counts": idx["counts"],
        "peak_rss_gib": round(s.max_rss / GiB, 1)}))


_WORKERS = {"baseline": worker_baseline, "parallel": worker_parallel, "ftw": worker_ftw, "build": worker_build}


# ---------------- parent orchestration ----------------
def _spawn(worker: str, ns):
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker", worker, "--model", ns.model,
           "--workers", str(ns.workers), "--chunk-mib", str(ns.chunk_mib),
           "--shard-gib", str(ns.shard_gib), "--ftw-dir", ns.ftw_dir]
    if ns.gpu:
        cmd += ["--gpu", ns.gpu]
    if ns.no_drop_cache:
        cmd.append("--no-drop-cache")
    # Capture ONLY stdout (the @@RESULT@@ line); let stderr inherit the terminal so the
    # framework's tqdm progress bars (and INFO logs), which write to stderr, show live.
    p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    print(f"  [error] worker {worker!r} failed (exit {p.returncode}); see stderr above.")
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--modes", default=",".join(ALL_MODES), help="comma list of: " + ",".join(ALL_MODES))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-mib", type=int, default=8)
    p.add_argument("--shard-gib", type=float, default=8.0, help="FTW shard size cap (ftw build)")
    p.add_argument("--ftw-dir", default="", help="FTW scratch dir (default: /var/tmp/ftw_<model>)")
    p.add_argument("--no-drop-cache", action="store_true",
                   help="don't evict page cache before each read (warm comparison)")
    p.add_argument("--keep-ftw", action="store_true", help="keep + reuse the FTW dir across runs")
    from freetoken.gpu_select import single_gpu_arg

    p.add_argument("--gpu", type=single_gpu_arg, default=None,
                   help="GPU UUID or nvidia-smi index (default: the first visible GPU)")
    p.add_argument("--_worker", default="")
    ns = p.parse_args()

    # The parent picks the FTW dir once (model-specific by default) and passes it to every
    # worker, so concurrent benches on different models never share an FTW path.
    if not ns.ftw_dir:
        ns.ftw_dir = _default_ftw_dir(ns.model)

    from freetoken.gpu_select import assign_gpu

    try:
        assign_gpu(ns.gpu)
    except ValueError as e:
        p.error(str(e))

    if ns._worker:
        from freetoken.gpu_select import bind_assigned_gpu

        bind_assigned_gpu()
        return _WORKERS[ns._worker](ns)

    from freetoken.checkpoint.ftw import is_ftw_checkpoint

    assert os.path.isdir(ns.model), f"model not found: {ns.model}"
    modes = [m.strip() for m in ns.modes.split(",") if m.strip()]
    print(f"model={ns.model}  workers={ns.workers}  chunk={ns.chunk_mib}MiB  modes={modes}")
    if "ftw" in modes:
        print(f"FTW={ns.ftw_dir}")
    print()

    # baseline is the reference every other mode is checked against.
    rp = _spawn("baseline", ns)
    if not rp:
        return
    base_ck = rp["checksum"]
    print(f"[baseline] quant={rp['quant_format']} banks={rp['n_banks']} size={rp['gib']}GiB  "
          f"load={rp['load_s']}s @ {rp['gibps']} GiB/s  peakRSS={rp['peak_rss_gib']}G")
    rows = [("baseline", rp)]

    # build the FTW once (reuse if --keep-ftw and it's already there).
    if "ftw" in modes:
        if ns.keep_ftw and is_ftw_checkpoint(ns.ftw_dir):
            print(f"[FTW ] reusing {ns.ftw_dir} (--keep-ftw)")
        else:
            b = _spawn("build", ns)
            if b:
                print(f"[FTW ] built {b['gib']}GiB / {b['shards']} shard(s) {b['counts']} "
                      f"in {b['build_s']}s  (one-time, offline)")

    for m in modes:
        if m not in ("parallel", "ftw"):
            print(f"  [skip] unknown mode {m!r}")
            continue
        r = _spawn(m, ns)
        rows.append((m, r))
        if r and r.get("skipped"):
            print(f"  [skip] {m}: {r['skipped']}")

    # ---- table ----
    hdr = f"{'mode':<10} {'load_s':>8} {'read GiB/s':>11} {'peakRSS':>8} {'speedup':>8} {'bytes':>6}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    base_tot = rp["load_s"]
    bad = []
    for name, r in rows:
        if r is None or r.get("skipped"):
            print(f"{name:<10} {'-':>8} {'-':>11} {'-':>8} {'-':>8} {'skip':>6}")
            continue
        if name == "baseline":
            print(f"{name:<10} {r['load_s']:>8.2f} {r['gibps']:>11.2f} {r['peak_rss_gib']:>7.1f}G "
                  f"{'1.00x':>8} {'-':>6}")
        else:
            ok = r["checksum"] == base_ck
            if not ok:
                bad.append((name, r["checksum"]))
            spd = f"{base_tot / r['load_s']:.2f}x" if r["load_s"] else "-"
            print(f"{name:<10} {r['load_s']:>8.2f} {r['gibps']:>11.2f} {r['peak_rss_gib']:>7.1f}G "
                  f"{spd:>8} {'ok' if ok else 'BAD':>6}")
    print("=" * len(hdr))
    print(f"banks={rp['gib']}GiB ({rp['n_banks']} x {rp['quant_format']})  "
          f"load_s = end-to-end load_expert_banks; speedup vs baseline.")

    if not ns.keep_ftw:
        shutil.rmtree(ns.ftw_dir, ignore_errors=True)

    # Exit non-zero on a byte mismatch so a scripted run (or CI) cannot read BAD as success.
    if bad:
        for name, ck in bad:
            differing = [b for b in sorted(base_ck) if ck.get(b) != base_ck[b]]
            print(f"[MISMATCH] {name}: bank(s) {differing} differ from baseline")
        raise SystemExit(f"checksum mismatch vs baseline: {', '.join(n for n, _ in bad)}")


if __name__ == "__main__":
    main()
