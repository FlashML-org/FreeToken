"""Patch an FTW checkpoint written by an older FreeToken so the current build loads it.

    python scripts/ftw_hotfix.py --ftw <ftw_dir> [--out <new_dir>] \
        [--repo <hf_repo_id> [--revision <rev>] | --source <local_hf_dir>] [--dry-run]

The tool builds the current model on the meta device from the FTW's own config.json, diffs the
FTW index against the tensors that model declares, and repairs the differences:

- renames (DeepSeek-V4: the dense tree moved under ``model.`` and ``.scale`` became ``.weight_scale_inv``)
- fp8 weights the current model declares as bf16 (old runtime fp8 residue) are dequantized, their ``.weight_scale`` dropped
- tensors the model declares but the FTW lacks (``input_scale``) are fetched from the HF repo by byte range
- a Qwen3.8-Flash-Next FTW without the PLE n-gram table gets it written as ``ple-table-*.safetensors``

Before anything is written the FTW is checked against the model: shapes, dtypes, byte counts and
shard files must agree, or the layout is reported as unsupported.

In place, new tensors go into an appended shard and the index is swapped atomically; shards left
with unreferenced bytes are then compacted one at a time, each step ending in another atomic index
swap. An interrupted run therefore always leaves a loadable FTW, and a rerun finishes the job.
``--out`` writes a fresh, compact FTW dir instead and never touches the original.
"""

from __future__ import annotations

import argparse
import bisect
import errno
import glob
import json
import math
import os
import re
import shutil
import struct
import sys

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
_FLOAT_DTYPES = {torch.bfloat16, torch.float16, torch.float32}
_CHUNK = 64 << 20
_SHARD_RE = re.compile(r"^freetoken-(\d{5})\.ftw$")
_PLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_FILE_RE = re.compile(r"^ple-table-\d{5}\.safetensors$")
_PLE_FILE_BYTES = 4 << 30
# tensors an old FTW may carry that the text model does not declare; they are kept and never count as errors
_IGNORED_PREFIXES = ("vision_tower.", "embed_vision.")
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


def tensor_bytes(shape, dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


# ------------------------------------------------------------------ safetensors slicing
class TensorSource:
    """Single tensors out of an HF repo (byte-range GET) or a local safetensors dir."""

    def __init__(self, repo: str | None, local: str | None, revision: str | None = None):
        assert repo or local, "need --repo or --source to fetch tensors"
        self.repo, self.local, self.revision = repo, local, revision
        self._headers: dict[str, tuple[dict, int]] = {}
        if local:
            files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(local, "*.safetensors")))
            index = os.path.join(local, "model.safetensors.index.json")
        else:
            from huggingface_hub import HfApi, hf_hub_download

            repo_files = HfApi().list_repo_files(repo, revision=revision)
            files = sorted(f for f in repo_files if f.endswith(".safetensors") and "/" not in f)
            index = (hf_hub_download(repo, "model.safetensors.index.json", revision=revision)
                     if "model.safetensors.index.json" in repo_files else None)
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
        url = hf_hub_url(self.repo, shard, revision=self.revision)
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=300)
        if r.status_code != 206:
            raise RuntimeError(f"range read of {shard} failed: HTTP {r.status_code}")
        return r.content

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            n = struct.unpack("<Q", self._range(shard, 0, 7))[0]
            self._headers[shard] = (json.loads(self._range(shard, 8, 8 + n - 1)), 8 + n)
        return self._headers[shard]

    def meta(self, name: str) -> dict:
        return self._header(self.weight_map[name])[0][name]

    def nbytes(self, name: str) -> int:
        a, b = self.meta(name)["data_offsets"]
        return b - a

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
        if (exp and exp[1] == torch.bfloat16 and e["dtype"] == "float8_e4m3fn" and tuple(e["shape"]) == exp[0]
                and scale is not None and scale["name"] not in expected and list(scale["shape"]) == [e["shape"][0]]):
            dequants.append((n, e, scale))
            drops.add(scale["name"])

    # (name, checkpoint parts): a fused module maps to several checkpoint tensors
    fetches: list[tuple[str, list[str]]] = []
    for n in (n for n in expected if n not in names):
        module, _, leaf = n.rpartition(".")
        parts = [f"{m}.{leaf}" for m in name_map.to_checkpoint(module)] if module else [n]
        fetches.append((n, parts))

    leftovers = [n for n in names if n not in expected and n not in drops]
    return renames, dequants, fetches, drops, leftovers


def resolve_fetches(fetches, source: TensorSource) -> tuple[dict[str, list[str]], list[str]]:
    """Map each missing tensor to the source tensors it is built from; a fused scale needs every part."""
    resolved: dict[str, list[str]] = {}
    errors: list[str] = []
    for n, parts in fetches:
        if source.has(n):
            resolved[n] = [n]
        elif all(source.has(p) for p in parts):
            if len(parts) > 1 and not n.endswith(".input_scale"):
                errors.append(f"{n} maps to {len(parts)} source tensors; fusing is not supported here")
            else:
                resolved[n] = parts
        else:
            errors.append(f"no source tensor for {n}: missing {[p for p in parts if not source.has(p)]}")
    return resolved, errors


def dequantize_rows(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (weight.float() * scale.float()[:, None]).to(torch.bfloat16)


def validate_ftw(ftw_dir: str, index: dict, expected: dict, renames: dict[str, str], dequant_names: set[str]) -> list[str]:
    """Structural checks of the index and shard files, plus shape/dtype agreement with the model."""
    problems: list[str] = []
    pos = 0
    for sh in sorted(index["shards"], key=lambda s: s["global_off"]):
        if sh["global_off"] != pos:
            problems.append(f"shard {sh['file']} starts at {sh['global_off']}, expected {pos}")
        path = os.path.join(ftw_dir, sh["file"])
        if not os.path.exists(path):
            problems.append(f"shard file {sh['file']} is missing")
        elif os.path.getsize(path) != sh["nbytes"]:
            problems.append(f"shard {sh['file']} is {os.path.getsize(path)} B, index says {sh['nbytes']}")
        pos = sh["global_off"] + sh["nbytes"]
    if pos != index["total_bytes"]:
        problems.append(f"shards end at {pos}, index total_bytes is {index['total_bytes']}")
    seen: set[str] = set()
    for e in index["tensors"]:
        if e["name"] in seen:
            problems.append(f"duplicate entry {e['name']}")
        seen.add(e["name"])
        try:
            dt = _dtype_of(e["dtype"])
        except Exception:
            problems.append(f"{e['name']}: unknown dtype {e['dtype']!r}")
            continue
        if tensor_bytes(e["shape"], dt) != e["nbytes"]:
            problems.append(f"{e['name']}: {e['shape']} {e['dtype']} is {tensor_bytes(e['shape'], dt)} B, index says {e['nbytes']}")
        if e["global_off"] < 0 or e["global_off"] % ALIGN:
            problems.append(f"{e['name']}: offset {e['global_off']} is not {ALIGN}-aligned")
        if e["global_off"] + e["nbytes"] > pos:
            problems.append(f"{e['name']}: extends past the last shard")
        if e["kind"] != "weight":
            continue
        name = renames.get(e["name"], e["name"])
        exp = expected.get(name)
        if exp is None:
            continue
        if name in dequant_names:  # shape already checked by plan(); the dtype is what the repair changes
            continue
        if tuple(e["shape"]) != exp[0]:
            problems.append(f"{name}: FTW shape {e['shape']} != model shape {list(exp[0])}")
        # the loader casts between float types; anything else has to match exactly
        if dt != exp[1] and not (dt in _FLOAT_DTYPES and exp[1] in _FLOAT_DTYPES):
            problems.append(f"{name}: FTW dtype {e['dtype']} is not loadable as {exp[1]}")
    blocks = sorted((e["global_off"], e["global_off"] + _align_up(e["nbytes"]), e["name"]) for e in index["tensors"])
    for (_, a1, an), (b0, _, bn) in zip(blocks, blocks[1:]):
        if b0 < a1:
            problems.append(f"{bn} overlaps {an}")
    return problems


# ------------------------------------------------------------------ FTW I/O
class ShardWriter:
    """Streams entries into ``freetoken-NNNNN.ftw`` shards, from offset 0 (new dir) or from the old end (append)."""

    def __init__(self, out_dir: str, shard_limit: int, *, first_shard: int = 0, global_off: int = 0):
        self.out_dir, self.shard_limit = out_dir, shard_limit
        self.global_off = global_off
        self.first_shard = first_shard
        self.shard_idx = first_shard - 1
        self._f = None
        self._start = self._cur = 0
        self.shards: list[dict] = []
        self.entries: list[dict] = []

    def _path(self, idx: int) -> str:
        return os.path.join(self.out_dir, _SHARD_FMT.format(idx))

    def _finish_shard(self) -> None:
        self.shards.append({"file": _SHARD_FMT.format(self.shard_idx), "global_off": self._start, "nbytes": self._cur})
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        self._f = None

    def _roll(self) -> None:
        if self._f is not None:
            self._finish_shard()
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
            self._finish_shard()
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


def fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        # a filesystem may not support fsync on a directory; any other error is a real persistence failure
        if not (os.path.isdir(path) and e.errno in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP)):
            raise
    finally:
        os.close(fd)


def write_index(ftw_dir: str, index: dict) -> None:
    # data of every file the new index points at is synced by its writer; sync the index, then the rename
    tmp = os.path.join(ftw_dir, INDEX_NAME + ".tmp")
    with open(tmp, "w") as f:
        json.dump(index, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(ftw_dir, INDEX_NAME))
    fsync_path(ftw_dir)


def shard_numbers(ftw_dir: str, index: dict) -> tuple[int, list[str]]:
    """The next free shard number, and shard files on disk that the index does not reference."""
    used = {sh["file"] for sh in index["shards"]}
    on_disk = [f for f in os.listdir(ftw_dir) if _SHARD_RE.match(f)]
    nums = [int(_SHARD_RE.match(f).group(1)) for f in set(on_disk) | used if _SHARD_RE.match(f)]
    return (max(nums) + 1 if nums else 0), sorted(f for f in on_disk if f not in used)


# ------------------------------------------------------------------ compaction
def live_ranges(tensors: list[dict], s0: int, s1: int) -> list[tuple[int, int]]:
    """Referenced byte ranges of shard [s0, s1), merged and sorted; every entry owns its ALIGN-padded block."""
    rs = []
    for e in tensors:
        a, b = e["global_off"], e["global_off"] + _align_up(e["nbytes"])
        a, b = max(a, s0), min(b, s1)
        if a < b:
            rs.append((a, b))
    rs.sort()
    merged: list[tuple[int, int]] = []
    for a, b in rs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def dirty_shards(index: dict) -> list[tuple[dict, list[tuple[int, int]], int]]:
    """Shards holding bytes no entry references, with their live ranges and live byte count."""
    out = []
    for sh in sorted(index["shards"], key=lambda s: s["global_off"]):
        s0, s1 = sh["global_off"], sh["global_off"] + sh["nbytes"]
        live = live_ranges(index["tensors"], s0, s1)
        live_bytes = sum(b - a for a, b in live)
        if live_bytes < sh["nbytes"]:
            out.append((sh, live, live_bytes))
    return out


def compact_in_place(work: str, index: dict, next_num: int) -> tuple[dict, int]:
    """Rewrite each dirty shard without its dead bytes; one atomic index swap per shard."""
    dirty = dirty_shards(index)
    b = bar(sum(lb for _, _, lb in dirty), "Compacting shards") if dirty else None
    n = 0
    while dirty:
        sh, live, live_bytes = dirty[0]
        s0, s1 = sh["global_off"], sh["global_off"] + sh["nbytes"]
        delta = sh["nbytes"] - live_bytes
        old_path = os.path.join(work, sh["file"])
        new_file = None
        if live_bytes:
            new_file = _SHARD_FMT.format(next_num)
            next_num += 1
            log(f"  compact {sh['file']} -> {new_file}: keep {live_bytes} B, drop {delta} B")
            with open(old_path, "rb") as src, open(os.path.join(work, new_file), "wb") as dst:
                for a, b1 in live:
                    src.seek(a - s0)
                    left = b1 - a
                    while left > 0:
                        buf = src.read(min(left, _CHUNK))
                        if not buf:
                            raise ValueError(f"short read in {sh['file']}")
                        dst.write(buf)
                        left -= len(buf)
                        tick(b, len(buf))
                dst.flush()
                os.fsync(dst.fileno())
        else:
            log(f"  drop empty shard {sh['file']}")
        starts = [a for a, _ in live]
        prefix = [0]
        for a, b1 in live:
            prefix.append(prefix[-1] + (b1 - a))

        def remap(go: int) -> int:
            i = bisect.bisect_right(starts, go) - 1
            if i < 0 or go >= live[i][1]:
                # only a zero-byte entry can sit in a dead gap: park it at the next live block
                return s0 + prefix[i + 1]
            return s0 + prefix[i] + (go - starts[i])

        tensors = []
        for e in index["tensors"]:
            go = e["global_off"]
            if go >= s1:
                e = {**e, "global_off": go - delta}
            elif go >= s0:
                e = {**e, "global_off": remap(go)}
            tensors.append(e)
        shards = []
        for x in sorted(index["shards"], key=lambda s: s["global_off"]):
            if x["file"] == sh["file"]:
                if new_file:
                    shards.append({"file": new_file, "global_off": s0, "nbytes": live_bytes})
            elif x["global_off"] > s0:
                shards.append({**x, "global_off": x["global_off"] - delta})
            else:
                shards.append(x)
        index = {**index, "tensors": tensors, "shards": shards, "total_bytes": index["total_bytes"] - delta}
        write_index(work, index)
        os.remove(old_path)
        n += 1
        dirty = dirty_shards(index)
    done(b)
    return index, n


# ------------------------------------------------------------------ PLE table
class PleSpec:
    """What the loader demands of the table: split_ngram_parts fp8 shards of [rows, ngram_head_dim] and one scalar scale."""

    def __init__(self, ftw_dir: str):
        with open(os.path.join(ftw_dir, "config.json")) as f:
            cfg = json.load(f)
        t = cfg.get("text_config") or cfg
        self.n_shards = int(t["split_ngram_parts"])
        self.head_dim = int(t["ple_embed_dim"]) // ((int(t["ngram_size"]) - 1) * int(t["heads_per_ngram"]))


def _ple_collect(items, bad: list[str]) -> tuple[dict[int, tuple[tuple, int]], int | None]:
    """Group (name, dtype, shape, nbytes) rows into shard index -> (shape, nbytes) and the scale's element count."""
    shards: dict[int, tuple[tuple, int]] = {}
    scale_numel = None
    for name, dtype, shape, nbytes in items:
        if name.endswith(_PLE_SCALE_SUFFIX):
            scale_numel = math.prod(shape)
            continue
        mt = _PLE_SHARD_RE.search(name)
        if mt is None:
            continue
        idx = int(mt.group(1))
        if dtype != "F8_E4M3":
            bad.append(f"shard {idx} has dtype {dtype}")
        if idx in shards:
            bad.append(f"duplicate shard {idx}")
        shards[idx] = (tuple(shape), nbytes)
    return shards, scale_numel


def classify_ple(spec: PleSpec, shards: dict[int, tuple[tuple, int]], scale_numel: int | None, bad: list[str]) -> tuple[str, str]:
    """('complete' | 'missing' | 'partial', detail) by the loader's rules."""
    if not shards and scale_numel is None and not bad:
        return "missing", ""
    for idx, (shape, nbytes) in sorted(shards.items()):
        if len(shape) != 2 or shape[1] != spec.head_dim:
            bad.append(f"shard {idx} is {list(shape)}, expected [rows, {spec.head_dim}]")
        elif nbytes != shape[0] * shape[1]:
            bad.append(f"shard {idx} holds {nbytes} B for shape {list(shape)}")
    if len({shape for shape, _ in shards.values()}) > 1:
        bad.append("shards differ in shape")
    if scale_numel is not None and scale_numel != 1:
        bad.append(f"weight_scale has {scale_numel} elements")
    if bad:
        return "partial", "; ".join(bad[:3])
    if scale_numel is None or sorted(shards) != list(range(spec.n_shards)):
        return "partial", f"{len(shards)}/{spec.n_shards} shards" + ("" if scale_numel is not None else ", no weight_scale")
    return "complete", ""


def ple_source_status(source: TensorSource, spec: PleSpec) -> tuple[str, str]:
    bad: list[str] = []
    names = [k for k in source.weight_map if _PLE_INFIX in k]
    shards, scale = _ple_collect(((k, source.meta(k)["dtype"], source.meta(k)["shape"], source.nbytes(k)) for k in names), bad)
    return classify_ple(spec, shards, scale, bad)


def ple_table_files(ftw_dir: str) -> list[str]:
    """The files the loader reads the table from: the safetensors index's mapping when there is one, else every *.safetensors."""
    index = os.path.join(ftw_dir, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(glob.glob(os.path.join(ftw_dir, "*.safetensors")))
    with open(index) as f:
        weight_map = json.load(f)["weight_map"]
    return sorted({os.path.join(ftw_dir, sh) for n, sh in weight_map.items() if _PLE_INFIX in n})


def ple_table_status(ftw_dir: str, spec: PleSpec) -> tuple[str, str, list[str]]:
    """The dir's PLE table as the loader would see it: (status, detail, files holding PLE tensors)."""
    bad: list[str] = []
    items: list[tuple] = []
    files: list[str] = []
    for path in ple_table_files(ftw_dir):
        name = os.path.basename(path)
        if not os.path.exists(path):
            bad.append(f"model.safetensors.index.json maps PLE tensors to missing {name}")
            continue
        try:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header, base = json.loads(f.read(n)), 8 + n
        except (struct.error, ValueError):
            bad.append(f"{name} has an unreadable header")
            files.append(name)
            continue
        size = os.path.getsize(path)
        holds = False
        for k, m in header.items():
            if k == "__metadata__" or _PLE_INFIX not in k:
                continue
            holds = True
            a, b = m["data_offsets"]
            if base + b > size:
                bad.append(f"{name} is truncated")
            items.append((k, m["dtype"], m["shape"], b - a))
        if holds:
            files.append(name)
    shards, scale = _ple_collect(items, bad)
    status, detail = classify_ple(spec, shards, scale, bad)
    return status, detail, files


def check_ftw(ftw_dir: str, index: dict, expected: dict, renames: dict[str, str], dequant_names: set[str], ple: PleSpec | None) -> dict:
    """Everything the tool verifies, on the input before planning and on the result after writing."""
    have = {renames.get(e["name"], e["name"]) for e in index["tensors"] if e["kind"] == "weight"}
    return {
        "problems": validate_ftw(ftw_dir, index, expected, renames, dequant_names),
        "missing": [n for n in expected if n not in have],
        "extra": [n for n in have if n not in expected and not n.startswith(_IGNORED_PREFIXES)],
        "dead": dirty_shards(index),
        "ple": ple_table_status(ftw_dir, ple) if ple else ("n/a", "", []),
    }


def ple_shards(out_dir: str, source: TensorSource) -> list[str]:
    """Write the PLE n-gram table tensors, and nothing else, into ``ple-table-*.safetensors`` files in ``out_dir``."""
    from safetensors.torch import save_file

    todo = sorted(n for n in source.weight_map if _PLE_INFIX in n)
    if not todo:
        raise SystemExit("ERROR: the source checkpoint has no PLE table tensors")
    for stale in glob.glob(os.path.join(out_dir, "ple-table-*.safetensors.tmp")):
        os.remove(stale)
    names: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0
    b = bar(sum(source.nbytes(n) for n in todo), "Writing PLE table")

    def flush():
        nonlocal batch, size
        if not batch:
            return
        name = f"ple-table-{len(names):05d}.safetensors"
        log(f"  write {name}: {len(batch)} tensors, {size / 2**30:.2f} GiB")
        save_file(batch, os.path.join(out_dir, name + ".tmp"))
        names.append(name)
        batch, size = {}, 0

    try:
        for n in todo:
            t = source.get(n)
            batch[n] = t
            size += t.numel() * t.element_size()
            tick(b, t.numel() * t.element_size())
            if size >= _PLE_FILE_BYTES:
                flush()
        flush()
    except BaseException:
        for name in names:
            os.remove(os.path.join(out_dir, name + ".tmp"))
        raise
    done(b)
    # every file is complete and synced before any gets its final name, so a scan never sees a half table
    for name in names:
        fsync_path(os.path.join(out_dir, name + ".tmp"))
        os.replace(os.path.join(out_dir, name + ".tmp"), os.path.join(out_dir, name))
    fsync_path(out_dir)
    return names


def side_files(src: str, skip) -> list[str]:
    """Everything in the FTW dir except the shards and the index: configs, tokenizer, PLE tables, nested dirs."""
    return [f for f in os.listdir(src) if not (f.endswith((".ftw", ".bak", ".tmp")) or f == INDEX_NAME or skip(f))]


def copy_side_files(src: str, dst: str, skip) -> None:
    os.makedirs(dst, exist_ok=True)
    for f in side_files(src, skip):
        s = os.path.join(src, f)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, f), dirs_exist_ok=True)
        else:
            shutil.copy2(s, os.path.join(dst, f))


def side_files_bytes(src: str, skip) -> int:
    total = 0
    for f in side_files(src, skip):
        p = os.path.join(src, f)
        total += sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(p) for x in fs) if os.path.isdir(p) else os.path.getsize(p)
    return total


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ftw", required=True, help="FTW checkpoint dir to repair")
    p.add_argument("--out", help="write a fresh, compact FTW dir here (new or empty) instead of patching in place")
    p.add_argument("--repo", help="HF repo id of the source checkpoint (tensors are read by byte range)")
    p.add_argument("--revision", help="HF revision (branch, tag or commit) of --repo")
    p.add_argument("--source", help="local HF safetensors dir of the source checkpoint (instead of --repo)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="print every step")
    ns = p.parse_args(argv)
    global _VERBOSE
    _VERBOSE = ns.verbose

    if not os.path.isfile(os.path.join(ns.ftw, INDEX_NAME)):
        print(f"ERROR: {ns.ftw} has no {INDEX_NAME}; not an FTW checkpoint dir", file=sys.stderr)
        return 2
    with open(os.path.join(ns.ftw, INDEX_NAME)) as f:
        index = json.load(f)
    if index.get("format") != FORMAT_TAG:
        print(f"ERROR: {ns.ftw} is not an FTW checkpoint (format {index.get('format')!r})", file=sys.stderr)
        return 2
    if ns.out:
        out, ftw = os.path.abspath(ns.out), os.path.abspath(ns.ftw)
        if out == ftw or os.path.commonpath([out, ftw]) == ftw:
            print("ERROR: --out must be outside --ftw (omit --out to patch in place)", file=sys.stderr)
            return 2
        if os.path.exists(out) and not os.path.isdir(out):
            print(f"ERROR: --out {ns.out} exists and is not a directory", file=sys.stderr)
            return 2
        if os.path.isdir(out) and os.listdir(out):
            print(f"ERROR: --out {ns.out} exists and is not empty", file=sys.stderr)
            return 2
    if ns.revision and not ns.repo:
        print("  note: --revision only applies to --repo; ignored", file=sys.stderr)
    resident_experts = any(e["kind"] == "weight" and ".experts." in e["name"] for e in index["tensors"])
    log(f"reading {os.path.join(ns.ftw, INDEX_NAME)}: {len(index['tensors'])} entries, {len(index['shards'])} shards, {index['total_bytes'] / 2**30:.2f} GiB")
    log("building the current model on the meta device from the FTW's config.json" + (" (resident experts)" if resident_experts else ""))
    arch, expected, name_map = expected_tensors(ns.ftw, resident_experts)
    log(f"{arch}: model declares {len(expected)} dense tensors")
    source = TensorSource(ns.repo, ns.source, ns.revision) if (ns.repo or ns.source) else None
    renames, dequants, fetches, drops, leftovers = plan(arch, index["tensors"], expected, name_map)
    dequant_names = {n for n, _, _ in dequants}
    ple = PleSpec(ns.ftw) if arch.startswith("Qwen4Exp") else None
    chk = check_ftw(ns.ftw, index, expected, renames, dequant_names, ple)
    problems, dirty = chk["problems"], chk["dead"]
    ple_status, ple_detail, ple_files = chk["ple"]
    need_ple = ple_status in ("missing", "partial")
    next_num, orphans = shard_numbers(ns.ftw, index)

    print(f"{arch}: {len(expected)} expected dense tensors, FTW has {sum(e['kind'] == 'weight' for e in index['tensors'])}")
    print(f"  renames {len(renames)}  dequantize {len(dequants)}  fetch {len(fetches)}  drop {len(drops)}  leftover {len(leftovers)}"
          + (f"  PLE table: {ple_status}" + (f" ({ple_detail})" if ple_detail else "") if need_ple else "")
          + (f"  dead bytes in {len(dirty)} shard(s)" if dirty else ""))
    for n, parts in fetches[:8]:
        print(f"    fetch {n} <- {parts}")
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
    if orphans:
        removing = not ns.out and bool(renames or dequants or fetches or drops or need_ple or dirty)
        print(f"  note: {len(orphans)} shard file(s) not referenced by the index, left by an interrupted run"
              + (", will be removed: " if removing else ", left as is: ") + str(orphans[:3]))

    bad = False
    for msg in problems[:10]:
        print(f"ERROR: {msg}", file=sys.stderr)
        bad = True
    if len(problems) > 10:
        print(f"ERROR: ... {len(problems) - 10} more", file=sys.stderr)
    if (fetches or need_ple) and source is None:
        print("ERROR: tensors must be fetched but neither --repo nor --source was given", file=sys.stderr)
        return 2
    fetch_srcs: dict[str, list[str]] = {}
    if fetches:
        fetch_srcs, errors = resolve_fetches(fetches, source)
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
            bad = True
    # the converter transforms most tensors on the way in (norm offsets, fusion, packing); only the
    # activation scale scalars are stored as the checkpoint has them, so only they can be fetched raw
    unfetchable = [n for n, _ in fetches if not n.endswith(".input_scale")]
    if unfetchable:
        print(f"ERROR: {len(unfetchable)} missing tensor(s) cannot be fetched raw (first: {unfetchable[0]}); reconvert the checkpoint", file=sys.stderr)
        bad = True
    real_leftovers = [n for n in leftovers if not n.startswith(_IGNORED_PREFIXES)]
    if real_leftovers:
        print(f"ERROR: the FTW holds {len(real_leftovers)} tensors the current model does not declare (first: {real_leftovers[0]}); this layout is not supported", file=sys.stderr)
        bad = True
    foreign_ple = [f for f in ple_files if not _PLE_FILE_RE.match(f)]
    if ple_status == "partial" and foreign_ple:
        print(f"ERROR: the PLE table in {ns.ftw} is incomplete ({ple_detail}) and lives in files this tool did not write: {foreign_ple[:3]}; remove them first", file=sys.stderr)
        bad = True
    if need_ple and os.path.exists(os.path.join(ns.ftw, "model.safetensors.index.json")):
        print(f"ERROR: {ns.ftw}/model.safetensors.index.json makes the loader read the PLE table through its mapping, which cannot list the files this tool writes; remove it (an FTW dir does not need it) and rerun", file=sys.stderr)
        bad = True
    if need_ple and source is not None:
        src_status, src_detail = ple_source_status(source, ple)
        if src_status != "complete":
            print(f"ERROR: the PLE table in the source is {src_status}" + (f" ({src_detail})" if src_detail else ""), file=sys.stderr)
            bad = True
    if bad:
        return 2

    replaced = dequant_names | set(fetch_srcs)
    keep: list[tuple[dict, str]] = []
    for e in index["tensors"]:
        name = renames.get(e["name"], e["name"]) if e["kind"] == "weight" else e["name"]
        if e["kind"] == "weight" and (name in drops or name in replaced):
            continue
        keep.append((e, name))
    kept_names = {e["name"] for e, _ in keep}
    skip_ple = (lambda f: bool(_PLE_FILE_RE.match(f))) if need_ple else (lambda f: False)

    # disk: the new entries and the PLE table, plus either one compacted shard next to its original
    # (in place) or the whole compact copy (--out)
    new_bytes = sum(_align_up(tensor_bytes(*expected[n])) for n in replaced)
    table_bytes = sum(source.nbytes(n) for n in source.weight_map if _PLE_INFIX in n) if need_ple else 0
    work = ns.out or ns.ftw
    if ns.out:
        need = sum(_align_up(e["nbytes"]) for e, _ in keep) + new_bytes + table_bytes + side_files_bytes(ns.ftw, skip_ple)
    else:
        after = {**index, "tensors": [e for e in index["tensors"] if e["name"] in kept_names]}
        need = new_bytes + table_bytes + max([lb for _, _, lb in dirty_shards(after)] + [0])
    probe = os.path.abspath(work)
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    print(f"  disk: about {need / 2**30:.1f} GiB needed under {work} ({free / 2**30:.1f} GiB free)")
    if ns.dry_run:
        return 0
    if not (renames or dequants or fetch_srcs or drops or need_ple or dirty or ns.out):
        print("nothing to do; the FTW loads as is")
        return 0
    if free < need + (1 << 30):
        print(f"ERROR: not enough free space under {work}", file=sys.stderr)
        return 2

    def write_new(w: ShardWriter) -> None:
        b = bar(sum(e["nbytes"] for _, e, _ in dequants), "Dequantizing") if dequants else None
        for n, e, scale_e in dequants:
            log(f"  dequantize {n}: fp8 x {scale_e['name']} -> bf16 {e['shape']}")
            w.add_tensor(n, dequantize_rows(read_entry(ns.ftw, index, e), read_entry(ns.ftw, index, scale_e)))
            tick(b, e["nbytes"])
        done(b)
        items = list(fetch_srcs.items())
        for n, srcs in (items if _VERBOSE or not items else count_bar(items, "Fetching tensors")):
            log(f"  fetch {n} <- {', '.join(srcs)}")
            vals = [source.get(c) for c in srcs]
            # a fused projection shares one activation scale; the reader takes the max over its parts
            w.add_tensor(n, torch.stack([v.reshape(()).float() for v in vals]).max().reshape(()))

    hotfix = {"from": os.path.abspath(ns.ftw), "renamed": len(renames), "dequantized": len(dequants),
              "fetched": len(fetch_srcs), "dropped": len(drops), "compacted": 0, "source": ns.repo or ns.source}
    compacted = 0
    if ns.out:
        log(f"--out: writing a fresh FTW -> {work}")
        try:
            copy_side_files(ns.ftw, work, skip_ple)
            w = ShardWriter(work, index["shard_limit"])
            b = bar(sum(e["nbytes"] for e, _ in keep), "Rewriting shards")
            for e, name in keep:
                log(f"  copy {e['name']}" + (f" -> {name}" if name != e["name"] else "") + f"  {e['nbytes']} B")
                w.add({**e, "name": name}, entry_chunks(ns.ftw, index, e))
                tick(b, e["nbytes"])
            done(b)
            write_new(w)
            shards = w.close()
            side = [f for f in index.get("side_files", []) if not skip_ple(f)]
            if need_ple:
                got = ple_shards(work, source)
                side += got
                print(f"  PLE table written: {len(got)} files")
            new_index = {**index, "tensors": w.entries, "shards": shards, "total_bytes": w.global_off,
                         "counts": {**index.get("counts", {}), "weight": sum(e["kind"] == "weight" for e in w.entries)},
                         "side_files": sorted(side), "hotfix": hotfix}
            write_index(work, new_index)
        except BaseException:
            # the dir was new or empty, so nothing of the user's is in it
            shutil.rmtree(work, ignore_errors=True)
            raise
        added = len(w.entries) - len(keep)
    else:
        for f in orphans:
            os.remove(os.path.join(work, f))
        new_entries: list[dict] = []
        new_shards: list[dict] = []
        if dequants or fetch_srcs:
            log(f"in place: appending shard {_SHARD_FMT.format(next_num)} -> {work}")
            w = ShardWriter(work, index["shard_limit"], first_shard=next_num, global_off=index["total_bytes"])
            try:
                write_new(w)
            except BaseException:
                w.abort()
                raise
            new_shards = w.close()
            new_entries = w.entries
            next_num = w.shard_idx + 1
        tensors = [{**e, "name": name} for e, name in keep] + new_entries
        new_index = {**index, "tensors": tensors, "shards": index["shards"] + new_shards,
                     "total_bytes": index["total_bytes"] + sum(sh["nbytes"] for sh in new_shards),
                     "counts": {**index.get("counts", {}), "weight": sum(e["kind"] == "weight" for e in tensors)},
                     "hotfix": hotfix}
        if renames or new_entries or drops:
            # rollback stays possible only while no shard gets replaced: keep the old index then
            bak = os.path.join(work, INDEX_NAME + ".bak")
            if not drops and not dirty and not os.path.exists(bak):
                shutil.copy2(os.path.join(work, INDEX_NAME), bak)
            write_index(work, new_index)
            log(f"index written: {os.path.join(work, INDEX_NAME)}")
        if need_ple:
            for f in ple_files:
                os.remove(os.path.join(work, f))
            got = ple_shards(work, source)
            side = {f for f in new_index.get("side_files", []) if not _PLE_FILE_RE.match(f)} | set(got)
            new_index = {**new_index, "side_files": sorted(side)}
            write_index(work, new_index)
            print(f"  PLE table written: {len(got)} files")
        new_index, compacted = compact_in_place(work, new_index, next_num)
        if compacted:
            new_index = {**new_index, "hotfix": {**hotfix, "compacted": compacted}}
            write_index(work, new_index)
            # the old shards are gone, so an old index can no longer roll anything back
            bak = os.path.join(work, INDEX_NAME + ".bak")
            if os.path.exists(bak):
                os.remove(bak)
        added = len(new_entries)

    chk = check_ftw(work, new_index, expected, {}, set(), ple)
    ple_status, ple_detail, _ = chk["ple"]
    print(f"wrote {work}: {len(new_index['tensors'])} entries, +{added} new, {len(new_index['shards'])} shard(s)"
          + (f", {compacted} compacted" if compacted else "") + f", total {new_index['total_bytes'] / 2**30:.2f} GiB")
    print(f"  check: missing {len(chk['missing'])}  extra {len(chk['extra'])}  problems {len(chk['problems'])}  "
          f"dead bytes in {len(chk['dead'])} shard(s)  PLE table: {ple_status}" + (f" ({ple_detail})" if ple_detail else ""))
    for n in (chk["missing"] + chk["extra"] + chk["problems"])[:10]:
        print(f"    {n}")
    ok = not (chk["missing"] or chk["extra"] or chk["problems"] or chk["dead"]) and ple_status in ("complete", "n/a")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
