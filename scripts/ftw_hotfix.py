"""Patch an FTW checkpoint written by an older FreeToken so the current build loads it.

    python scripts/ftw_hotfix.py --ftw <ftw_dir> [--out <new_dir>] \
        [--repo <hf_repo_id> | --source <local_hf_dir>] [--dry-run]

The tool builds the current model on the meta device from the FTW's own config.json, diffs the
FTW index against the tensors that model declares, and repairs the differences:

- renames (DeepSeek-V4: the dense tree moved under ``model.`` and ``.scale`` became ``.weight_scale_inv``)
- fp8 weights the current model declares as bf16 (old runtime fp8 residue) are dequantized, their ``.weight_scale`` dropped
- tensors the model declares but the FTW lacks (``input_scale``) are fetched from the HF repo by byte range
- a Qwen3.8-Flash-Next FTW without the PLE n-gram table gets it written as ``ple-table-*.safetensors``

In place, an FTW that only gains tensors gets one shard appended; when entries are replaced or
dropped the live entries are rewritten into fresh shards so no dead bytes remain. ``--out`` always
writes a fresh, compact FTW dir. While the old shards are kept, the previous index stays as
``freetoken_weight.json.bak`` so the patch can be undone.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import struct
import sys
from typing import Any

import torch

from freetoken.checkpoint.ftw import ALIGN, FORMAT_TAG, INDEX_NAME, _align_up, _dtype_of, _dtype_str, _SHARD_FMT
from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import _decode_target
from freetoken.layers import set_rope_device
from freetoken.layers.quantization.names import NameMap
from freetoken.models import create_model
from freetoken.models.register import get_model_spec
from freetoken.utils.progress import byte_bar, count_bar
from freetoken.utils.torch_utils import torch_dtype

_ST_DTYPES = {
    "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2, "U8": torch.uint8, "I8": torch.int8, "I32": torch.int32, "I64": torch.int64,
}
_CHUNK = 64 << 20
_PLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_FILE_BYTES = 4 << 30
_VERBOSE = False


def log(msg: str) -> None:
    if _VERBOSE:
        print(msg, flush=True)


def bar(total_bytes: int, desc: str):
    """A byte progress bar in normal mode; verbose mode prints per-step lines instead."""
    return None if _VERBOSE else byte_bar(total_bytes, desc)


def tick(b, n: int) -> None:
    if b is not None:
        b.update(n)


def done(b) -> None:
    if b is not None:
        b.close()


# ------------------------------------------------------------------ safetensors slicing
class TensorSource:
    """Single tensors out of an HF repo (byte-range GET) or a local safetensors dir."""

    def __init__(self, repo: str | None, local: str | None):
        assert repo or local, "need --repo or --source to fetch tensors"
        self.repo, self.local = repo, local
        self._headers: dict[str, tuple[dict, int]] = {}
        if local:
            files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(local, "*.safetensors")))
            index = os.path.join(local, "model.safetensors.index.json")
        else:
            from huggingface_hub import HfApi, hf_hub_download

            repo_files = HfApi().list_repo_files(repo)
            files = sorted(f for f in repo_files if f.endswith(".safetensors") and "/" not in f)
            index = hf_hub_download(repo, "model.safetensors.index.json") if "model.safetensors.index.json" in repo_files else None
        if index and os.path.exists(index):
            with open(index) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = {name: shard for shard in files for name in self._header(shard)[0] if name != "__metadata__"}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _range(self, shard: str, start: int, end: int) -> bytes:
        if self.local:
            with open(os.path.join(self.local, shard), "rb") as f:
                f.seek(start)
                return f.read(end - start + 1)
        import requests
        from huggingface_hub import get_token, hf_hub_url

        headers = {"Range": f"bytes={start}-{end}"}
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(hf_hub_url(self.repo, shard), headers=headers, allow_redirects=True, timeout=300)
        if r.status_code != 206:
            raise RuntimeError(f"range read of {shard} failed: HTTP {r.status_code}")
        return r.content

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            n = struct.unpack("<Q", self._range(shard, 0, 7))[0]
            self._headers[shard] = (json.loads(self._range(shard, 8, 8 + n - 1)), 8 + n)
        return self._headers[shard]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        header, base = self._header(shard)
        meta = header[name]
        a, b = meta["data_offsets"]
        raw = bytearray(self._range(shard, base + a, base + b - 1))
        return torch.frombuffer(raw, dtype=_ST_DTYPES[meta["dtype"]]).reshape(meta["shape"])


# ------------------------------------------------------------------ what the current build expects
def expected_tensors(ftw_dir: str, resident_experts: bool):
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_rope_device(torch.device("cpu"))
    kw = dict(model_path=ftw_dir, tp_info=try_get_tp_info(), dtype=torch.bfloat16)
    mc0 = EngineConfig(**kw).model_config
    is_moe = bool(getattr(mc0, "is_moe", False) or getattr(mc0, "moe_enabled", False))
    strategy = "offload" if is_moe and not resident_experts else "fused"
    cfg = EngineConfig(**kw, moe_strategy=strategy)
    object.__setattr__(cfg.model_config, "moe_strategy", strategy)
    object.__setattr__(cfg.model_config, "decode_target", _decode_target(cfg))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg.model_config)
    state = {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}
    arch = cfg.model_config.architectures[0]
    spec = get_model_spec(arch)
    name_map = NameMap(roots=spec.checkpoint_roots, segments=spec.checkpoint_segments, packed=spec.packed_modules_mapping)
    return arch, state, name_map


# ------------------------------------------------------------------ repairs
def dsv4_rename(name: str) -> str:
    if name == "head":
        return "model.head.weight"
    if name.endswith(".scale"):
        name = name[: -len(".scale")] + ".weight_scale_inv"
    return "model." + name


def plan(arch: str, entries: list[dict], expected: dict, name_map: NameMap):
    """Return (renames, dequants, fetches, drops, leftovers) that turn the FTW dense set into ``expected``."""
    dense = {e["name"]: e for e in entries if e["kind"] == "weight"}
    renames: dict[str, str] = {}
    if arch.startswith("DeepseekV4") and not any(n in expected for n in dense):
        for n in dense:
            new = dsv4_rename(n)
            if new not in expected:
                raise SystemExit(f"DeepSeek-V4 rename has no target for {n!r} -> {new!r}")
            renames[n] = new
    names = {renames.get(n, n): e for n, e in dense.items()}

    dequants: list[tuple[str, dict, dict]] = []
    drops: set[str] = set()
    for n, e in names.items():
        exp = expected.get(n)
        scale = names.get(n[: -len(".weight")] + ".weight_scale") if n.endswith(".weight") else None
        if exp and exp[1] == torch.bfloat16 and e["dtype"] == "float8_e4m3fn" and scale is not None and scale["name"] not in expected:
            dequants.append((n, e, scale))
            drops.add(scale["name"])

    fetches: list[tuple[str, list[str]]] = []
    for n in (n for n in expected if n not in names):
        module, _, leaf = n.rpartition(".")
        cands = [f"{m}.{leaf}" for m in name_map.to_checkpoint(module)] if module else [n]
        if n not in cands:
            cands.append(n)
        fetches.append((n, cands))

    leftovers = [n for n in names if n not in expected and n not in drops]
    return renames, dequants, fetches, drops, leftovers


def dequantize_rows(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (weight.float() * scale.float()[:, None]).to(torch.bfloat16)


# ------------------------------------------------------------------ FTW I/O
class ShardWriter:
    """Streams entries into ``freetoken-NNNNN.ftw`` shards, from offset 0 (rewrite) or from the old end (append)."""

    def __init__(self, out_dir: str, shard_limit: int, *, first_shard: int = 0, global_off: int = 0, suffix: str = ""):
        self.out_dir, self.shard_limit, self.suffix = out_dir, shard_limit, suffix
        self.global_off = global_off
        self.first_shard = first_shard
        self.shard_idx = first_shard - 1
        self._f = None
        self._start = self._cur = 0
        self.shards: list[dict] = []
        self.entries: list[dict] = []

    def _path(self, idx: int) -> str:
        return os.path.join(self.out_dir, _SHARD_FMT.format(idx) + self.suffix)

    def _roll(self) -> None:
        if self._f is not None:
            self.shards.append({"file": _SHARD_FMT.format(self.shard_idx), "global_off": self._start, "nbytes": self._cur})
            self._f.close()
        self.shard_idx += 1
        self._start, self._cur = self.global_off, 0
        self._f = open(self._path(self.shard_idx), "wb")

    def _write(self, data) -> None:
        off, n = 0, len(data)
        while off < n:
            if self._f is None or self._cur == self.shard_limit:
                self._roll()
            take = min(n - off, self.shard_limit - self._cur)
            self._f.write(data[off:off + take])
            off += take
            self._cur += take
            self.global_off += take

    def add(self, entry: dict, chunks) -> None:
        nbytes = entry["nbytes"]
        # a tensor that fits a shard never straddles two: roll early like FTWWriter does
        if self._f is None or (nbytes <= self.shard_limit and self._cur + nbytes > self.shard_limit):
            self._roll()
        assert self.global_off % ALIGN == 0
        self.entries.append({**entry, "global_off": self.global_off})
        written = 0
        for c in chunks:
            self._write(c)
            written += len(c)
        assert written == nbytes, (entry["name"], written, nbytes)
        pad = _align_up(self.global_off) - self.global_off
        if pad:
            self._write(bytes(pad))

    def add_tensor(self, name: str, t: torch.Tensor) -> None:
        t = t.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        self.add({"name": name, "kind": "weight", "dtype": _dtype_str(t.dtype), "shape": list(t.shape), "nbytes": len(raw)}, [raw])

    def close(self) -> list[dict]:
        if self._f is not None:
            self.shards.append({"file": _SHARD_FMT.format(self.shard_idx), "global_off": self._start, "nbytes": self._cur})
            self._f.close()
            self._f = None
        return self.shards

    def abort(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        for idx in range(self.first_shard, self.shard_idx + 1):
            try:
                os.remove(self._path(idx))
            except FileNotFoundError:
                pass


def entry_chunks(ftw_dir: str, index: dict, e: dict):
    """The bytes of an existing entry, read from its shards in pieces of at most _CHUNK."""
    pos, remaining = e["global_off"], e["nbytes"]
    for sh in sorted(index["shards"], key=lambda s: s["global_off"]):
        s0, s1 = sh["global_off"], sh["global_off"] + sh["nbytes"]
        if remaining <= 0 or pos >= s1:
            continue
        with open(os.path.join(ftw_dir, sh["file"]), "rb") as f:
            f.seek(pos - s0)
            take = min(s1 - pos, remaining)
            while take > 0:
                buf = f.read(min(take, _CHUNK))
                if not buf:
                    raise ValueError(f"short read in {sh['file']} for {e['name']}")
                yield buf
                take -= len(buf)
                pos += len(buf)
                remaining -= len(buf)
    assert remaining == 0, (e["name"], remaining)


def read_entry(ftw_dir: str, index: dict, e: dict) -> torch.Tensor:
    buf = bytearray(b"".join(entry_chunks(ftw_dir, index, e)))
    return torch.frombuffer(buf, dtype=_dtype_of(e["dtype"])).reshape(e["shape"])


def ple_tensors_present(ftw_dir: str) -> int:
    have = 0
    for path in glob.glob(os.path.join(ftw_dir, "*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            have += sum(_PLE_INFIX in k for k in json.loads(f.read(n)))
    return have


def ple_shards(out_dir: str, source: TensorSource) -> list[str]:
    """Write the PLE n-gram table tensors, and nothing else, into ``ple-table-*.safetensors`` files in ``out_dir``."""
    from safetensors.torch import save_file

    todo = sorted(n for n in source.weight_map if _PLE_INFIX in n)
    if not todo:
        raise SystemExit("ERROR: the source checkpoint has no PLE table tensors")
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0
    b = bar(sum(b1 - a1 for n in todo for a1, b1 in [source._header(source.weight_map[n])[0][n]["data_offsets"]]), "Writing PLE table")

    def flush():
        nonlocal batch, size
        if not batch:
            return
        path = os.path.join(out_dir, f"ple-table-{len(written):05d}.safetensors")
        log(f"  write {os.path.basename(path)}: {len(batch)} tensors, {size / 2**30:.2f} GiB")
        save_file(batch, path)
        written.append(os.path.basename(path))
        batch, size = {}, 0

    for n in todo:
        t = source.get(n)
        batch[n] = t
        size += t.numel() * t.element_size()
        tick(b, t.numel() * t.element_size())
        if size >= _PLE_FILE_BYTES:
            flush()
    flush()
    done(b)
    return written


def copy_side_files(src: str, dst: str) -> None:
    """Everything in the FTW dir except the shards and the index: configs, tokenizer, PLE tables, nested dirs."""
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith((".ftw", ".bak", ".tmp")) or f == INDEX_NAME:
            continue
        s = os.path.join(src, f)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, f), dirs_exist_ok=True)
        else:
            shutil.copy2(s, os.path.join(dst, f))


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ftw", required=True, help="FTW checkpoint dir to repair")
    p.add_argument("--out", help="write a fresh, compact FTW dir here instead of patching in place")
    p.add_argument("--repo", help="HF repo id of the source checkpoint (tensors are read by byte range)")
    p.add_argument("--source", help="local HF safetensors dir of the source checkpoint (instead of --repo)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="print every step")
    ns = p.parse_args(argv)
    global _VERBOSE
    _VERBOSE = ns.verbose

    with open(os.path.join(ns.ftw, INDEX_NAME)) as f:
        index = json.load(f)
    assert index.get("format") == FORMAT_TAG, f"{ns.ftw} is not an FTW checkpoint"
    if ns.out and os.path.abspath(ns.out) == os.path.abspath(ns.ftw):
        print("ERROR: --out must differ from --ftw (omit --out to patch in place)", file=sys.stderr)
        return 2
    resident_experts = any(e["kind"] == "weight" and ".experts." in e["name"] for e in index["tensors"])
    log(f"reading {os.path.join(ns.ftw, INDEX_NAME)}: {len(index['tensors'])} entries, {len(index['shards'])} shards, {index['total_bytes'] / 2**30:.2f} GiB")
    log("building the current model on the meta device from the FTW's config.json" + (" (resident experts)" if resident_experts else ""))
    arch, expected, name_map = expected_tensors(ns.ftw, resident_experts)
    log(f"{arch}: model declares {len(expected)} dense tensors")
    source = TensorSource(ns.repo, ns.source) if (ns.repo or ns.source) else None
    renames, dequants, fetches, drops, leftovers = plan(arch, index["tensors"], expected, name_map)
    need_ple = arch.startswith("Qwen4Exp") and ple_tensors_present(ns.ftw) == 0

    print(f"{arch}: {len(expected)} expected dense tensors, FTW has {sum(e['kind'] == 'weight' for e in index['tensors'])}")
    print(f"  renames {len(renames)}  dequantize {len(dequants)}  fetch {len(fetches)}  drop {len(drops)}  leftover {len(leftovers)}"
          + ("  PLE table: missing" if need_ple else ""))
    for n, cands in fetches[:8]:
        print(f"    fetch {n} <- {cands}")
    if len(fetches) > 8:
        print(f"    ... {len(fetches) - 8} more")
    for old, new in list(renames.items())[:5]:
        log(f"    rename {old} -> {new}")
    if len(renames) > 5:
        log(f"    ... {len(renames) - 5} more renames")
    for n, e, scale_e in dequants:
        log(f"    dequantize {n} (drop {scale_e['name']})")
    for n in leftovers[:8]:
        print(f"    leftover (not declared by the model): {n}")
    if (fetches or need_ple) and source is None:
        print("ERROR: tensors must be fetched but neither --repo nor --source was given", file=sys.stderr)
        return 2
    bad = False
    for n, c in fetches:
        srcs = [x for x in c if source.has(x)]
        if not srcs:
            print(f"ERROR: no source tensor for {n} (tried {c})", file=sys.stderr)
            bad = True
        elif len(srcs) > 1 and not n.endswith(".input_scale"):
            print(f"ERROR: {n} maps to {len(srcs)} source tensors; fusing is not supported here", file=sys.stderr)
            bad = True
    real_leftovers = [n for n in leftovers if not n.startswith(("vision_tower.", "embed_vision."))]
    if real_leftovers:
        print(f"ERROR: the FTW holds {len(real_leftovers)} tensors the current model does not declare (first: {real_leftovers[0]}); this layout is not supported", file=sys.stderr)
        bad = True
    if bad:
        return 2
    if ns.dry_run:
        return 0
    if not (renames or dequants or fetches or drops or need_ple):
        print("nothing to do; the FTW loads as is")
        return 0

    replaced = {n for n, _, _ in dequants} | {n for n, _ in fetches}
    keep: list[tuple[dict, str]] = []
    for e in index["tensors"]:
        name = renames.get(e["name"], e["name"]) if e["kind"] == "weight" else e["name"]
        if e["kind"] == "weight" and (name in drops or name in replaced):
            continue
        keep.append((e, name))
    rewrite = bool(ns.out) or len(keep) < len(index["tensors"])
    work = ns.out or ns.ftw
    log(("--out: writing a fresh FTW" if ns.out else "in place: rewriting shards, no dead bytes" if rewrite else "in place: appending one shard") + f" -> {work}")
    if ns.out:
        copy_side_files(ns.ftw, ns.out)
    if rewrite:
        live = sum(_align_up(e["nbytes"]) for e, _ in keep)
        if shutil.disk_usage(work).free < live + (1 << 30):
            print(f"ERROR: {work} needs about {live / 2**30:.1f} GiB free to rewrite the shards", file=sys.stderr)
            return 2
        w = ShardWriter(work, index["shard_limit"], suffix="" if ns.out else ".tmp")
    else:
        w = ShardWriter(work, index["shard_limit"], first_shard=len(index["shards"]), global_off=index["total_bytes"])
    try:
        if rewrite:
            log(f"rewriting {len(keep)} live entries into new shards under {work}")
            b = bar(sum(e["nbytes"] for e, _ in keep), "Rewriting shards")
            for e, name in keep:
                log(f"  copy {e['name']}" + (f" -> {name}" if name != e["name"] else "") + f"  {e['nbytes']} B")
                w.add({**e, "name": name}, entry_chunks(ns.ftw, index, e))
                tick(b, e["nbytes"])
            done(b)
        b = bar(sum(e["nbytes"] for _, e, _ in dequants), "Dequantizing") if dequants else None
        for n, e, scale_e in dequants:
            log(f"  dequantize {n}: fp8 x {scale_e['name']} -> bf16 {e['shape']}")
            w.add_tensor(n, dequantize_rows(read_entry(ns.ftw, index, e), read_entry(ns.ftw, index, scale_e)))
            tick(b, e["nbytes"])
        done(b)
        for n, cands in (fetches if _VERBOSE or not fetches else count_bar(fetches, "Fetching tensors")):
            srcs = [c for c in cands if source.has(c)]
            log(f"  fetch {n} <- {', '.join(srcs)}")
            vals = [source.get(c) for c in srcs]
            shape, dtype = expected[n]
            if n.endswith(".input_scale"):
                # a fused projection shares one activation scale; the reader takes the max over its parts
                w.add_tensor(n, torch.stack([v.reshape(()).float() for v in vals]).max().reshape(()))
                continue
            v = vals[0]
            if tuple(v.shape) != shape:
                raise SystemExit(f"ERROR: {n}: source shape {tuple(v.shape)} != expected {shape}")
            w.add_tensor(n, v.to(dtype))
    except BaseException:
        w.abort()
        raise
    new_shards = w.close()

    tensors = w.entries if rewrite else [{**e, "name": name} for e, name in keep] + w.entries
    shards = new_shards if rewrite else index["shards"] + new_shards
    new_index = {**index, "tensors": tensors, "total_bytes": w.global_off, "shards": shards,
                 "counts": {**index.get("counts", {}), "weight": sum(e["kind"] == "weight" for e in tensors)},
                 "hotfix": {"from": os.path.abspath(ns.ftw), "renamed": len(renames), "dequantized": len(dequants),
                            "fetched": len(fetches), "dropped": len(drops), "rewritten": rewrite, "source": ns.repo or ns.source}}
    if need_ple:
        log("writing the PLE table into ple-table-*.safetensors")
        got = ple_shards(work, source)
        print(f"  PLE table written: {len(got)} files")
    if rewrite and not ns.out:
        log(f"replacing {len(index['shards'])} old shards with {len(new_shards)} new ones")
        for sh in index["shards"]:
            os.remove(os.path.join(work, sh["file"]))
        for sh in new_shards:
            os.replace(os.path.join(work, sh["file"] + ".tmp"), os.path.join(work, sh["file"]))
    # the old index only allows a rollback while the old shards still exist
    bak = os.path.join(work, INDEX_NAME + ".bak")
    if not ns.out and not rewrite and not os.path.exists(bak):
        shutil.copy2(os.path.join(work, INDEX_NAME), bak)
    tmp = os.path.join(work, INDEX_NAME + ".tmp")
    with open(tmp, "w") as f:
        json.dump(new_index, f)
    os.replace(tmp, os.path.join(work, INDEX_NAME))
    log(f"index written: {os.path.join(work, INDEX_NAME)}")

    final = {e["name"] for e in tensors if e["kind"] == "weight"}
    still_missing = [n for n in expected if n not in final]
    extra = [n for n in final if n not in expected]
    added = len(w.entries) - (len(keep) if rewrite else 0)
    print(f"wrote {work}: {len(tensors)} entries, +{added} new, {'rewritten' if rewrite else 'appended'}, "
          f"{len(shards)} shard(s), total {w.global_off / 2**30:.2f} GiB")
    print(f"  check: missing {len(still_missing)}  extra {len(extra)}")
    for n in (still_missing + extra)[:10]:
        print(f"    {n}")
    return 1 if still_missing else 0


if __name__ == "__main__":
    sys.exit(main())
