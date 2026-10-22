"""Edit primitives for HF safetensors checkpoints: config keys, quantization skip lists, tensor names. Tensor data is never changed.

    from ckpt_edit import Checkpoint
    ck = Checkpoint("/path/to/ckpt")
    ck.set_config("text_config.tie_word_embeddings", False)
    ck.quant_skip_add(["model.language_model.layers.*.mlp.gate"])
    ck.rename(lambda name: name.replace("model.language_model.language_model.", "model.language_model."))
    ck.save(out="/path/to/fixed")      # or ck.save() to edit in place

``save(out=...)`` writes a new directory: changed files are written there, every other file is a symlink to the original.
``save()`` edits in place and first copies each changed JSON file and shard header to ``.ckpt_edit_backup/<time>/``; ``restore()`` puts the latest backup back.
A renamed shard keeps its data bytes: its header is patched in place when the new one fits in the old one's bytes, else the shard is rewritten through a temp file.
A shard that is a symlink (hub cache) or has other hard links is never written through: it becomes a private copy.
"""

from __future__ import annotations

import copy
import errno
import fnmatch
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from typing import Any, Callable, Iterable

INDEX = "model.safetensors.index.json"
CONFIG = "config.json"
SIDECAR = "hf_quant_config.json"
BACKUP_DIR = ".ckpt_edit_backup"
STUB_MARKER = ".stub.json"
# keys that list modules a quantization_config leaves unquantized
SKIP_KEYS = ("ignore", "exclude_modules", "modules_to_not_convert")
_DEFAULT_SKIP_KEY = {"modelopt": "ignore", "compressed-tensors": "ignore", "fp8": "modules_to_not_convert", "mxfp4": "modules_to_not_convert"}


def header_len(path: str) -> int:
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def read_header(path: str) -> tuple[dict, int]:
    """(header, header byte length) of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), n


def encode_header(header: dict, fit: int | None = None) -> bytes:
    """The header JSON padded with spaces to ``fit`` bytes when it fits, else to a multiple of 8."""
    raw = json.dumps(header, separators=(",", ":")).encode()
    size = fit if fit is not None and len(raw) <= fit else (len(raw) + 7) // 8 * 8
    return raw + b" " * (size - len(raw))


def _copy_range(src_fd: int, dst_fd: int, src_off: int, dst_off: int, count: int) -> None:
    fast = hasattr(os, "copy_file_range")
    while count:
        n = 0
        if fast:
            try:
                n = os.copy_file_range(src_fd, dst_fd, count, src_off, dst_off)
            except OSError:
                fast = False
        if n <= 0:
            buf = os.pread(src_fd, min(count, 64 << 20), src_off)
            if not buf:
                raise EOFError("source shard ended early")
            n = os.pwrite(dst_fd, buf, dst_off)
        count, src_off, dst_off = count - n, src_off + n, dst_off + n


def stub_data_end(path: str, n: int) -> int | None:
    """For a fetch_headers.py stub, the end of the block that holds the header; past it the file is a hole. None for anything else.

    A shard counts as a stub only next to the stub marker and when SEEK_DATA finds no data past the header's block."""
    if not os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(path)), STUB_MARKER)):
        return None
    st = os.stat(path)
    end = min(st.st_size, (8 + n + st.st_blksize - 1) // st.st_blksize * st.st_blksize)
    if end >= st.st_size:
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        os.lseek(fd, end, os.SEEK_DATA)
    except OSError as e:
        return end if e.errno == errno.ENXIO else None
    finally:
        os.close(fd)
    return None


def _fsync_dir(path: str) -> None:
    fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_shard(src: str, dst: str, header: dict | bytes, *, private: bool = False) -> str:
    """Write ``header`` (a dict, or the exact header bytes) with the data bytes of ``src`` to ``dst``; returns ``"in place"`` or ``"rewritten"``.

    ``private`` forces a rewrite through a temp file even when ``dst`` is ``src``, so a symlink or a hard link is replaced instead of written through."""
    n_old = header_len(src)
    size = os.path.getsize(src)
    same = os.path.realpath(src) == os.path.realpath(dst)
    data = header if isinstance(header, bytes) else encode_header(header, fit=n_old if same else None)
    if same and not private and len(data) == n_old:
        fd = os.open(dst, os.O_WRONLY)
        try:
            os.pwrite(fd, data, 8)
            os.fsync(fd)
        finally:
            os.close(fd)
        return "in place"
    tmp = dst + ".ckpt_edit.part"
    head = struct.pack("<Q", len(data)) + data
    stub_end = stub_data_end(src, n_old)
    src_fd, dst_fd = os.open(src, os.O_RDONLY), os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.pwrite(dst_fd, head, 0)
        if stub_end is not None:
            # a stub's data is a hole: copy the bytes up to the end of the header's block, keep the rest a hole
            _copy_range(src_fd, dst_fd, 8 + n_old, len(head), stub_end - 8 - n_old)
            os.ftruncate(dst_fd, len(head) + size - 8 - n_old)
        else:
            _copy_range(src_fd, dst_fd, 8 + n_old, len(head), size - 8 - n_old)
        os.fsync(dst_fd)
    except BaseException:
        os.close(dst_fd)
        os.remove(tmp)
        raise
    finally:
        os.close(src_fd)
    os.close(dst_fd)
    os.replace(tmp, dst)
    _fsync_dir(dst)
    return "rewritten"


def _shared(path: str) -> bool:
    """A symlink or a file with other hard links: writing through it would change another copy."""
    return os.path.islink(path) or os.stat(path).st_nlink > 1


def get_path(cfg: dict, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(cfg, dict) or key not in cfg:
            return None
        cfg = cfg[key]
    return cfg


def set_path(cfg: dict, path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for key in parents:
        cfg = cfg.setdefault(key, {})
    cfg[leaf] = value


def entry_names(entry: str, module: str) -> bool:
    """Whether a skip-list entry or target covers ``module``, read loosely across dialects (literal name or parent, glob, ``re:``)."""
    if entry.startswith("re:"):
        try:
            return re.match(entry[3:], module) is not None
        except re.error:
            return False
    names = [module]
    while "." in names[-1]:
        names.append(names[-1].rpartition(".")[0])
    return any(n == entry or fnmatch.fnmatchcase(n, entry) for n in names)


class Checkpoint:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.json: dict[str, dict] = {}
        for name in (CONFIG, SIDECAR, INDEX):
            p = os.path.join(self.path, name)
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    self.json[name] = json.load(f)
        self._orig_json = copy.deepcopy(self.json)
        self.headers: dict[str, dict] = {}
        self.header_len: dict[str, int] = {}
        digest = hashlib.sha256(json.dumps(self.json, sort_keys=True).encode())
        for name in sorted(os.listdir(self.path)):
            if name.endswith(".safetensors"):
                self.headers[name], self.header_len[name] = read_header(os.path.join(self.path, name))
                digest.update(name.encode() + json.dumps(self.headers[name], sort_keys=True).encode())
        # the state the edits are planned against; a plan made on another state is refused
        self.fingerprint = digest.hexdigest()
        self._changed_shards: set[str] = set()
        self.log: list[str] = []

    # --- reading ---

    def tensors(self) -> dict[str, tuple[str, dict]]:
        """name -> (shard, {dtype, shape, data_offsets})."""
        return {k: (shard, v) for shard, h in self.headers.items() for k, v in h.items() if k != "__metadata__"}

    @property
    def config(self) -> dict:
        return self.json.setdefault(CONFIG, {})

    def quant_configs(self) -> list[tuple[str, dict]]:
        """(where, dict) of every quantization config the checkpoint carries: config.json top level, its text_config, and hf_quant_config.json."""
        found = []
        for where, q in ((CONFIG, get_path(self.config, "quantization_config")),
                         (f"{CONFIG}:text_config", get_path(self.config, "text_config.quantization_config")),
                         (SIDECAR, get_path(self.json.get(SIDECAR, {}), "quantization"))):
            if isinstance(q, dict):
                found.append((where, q))
        return found

    def _method(self, where: str, q: dict) -> str:
        return "modelopt" if where == SIDECAR else str(q.get("quant_method") or "").lower()

    def quant_entries(self) -> list[tuple[str, str, list]]:
        """(where, key, list) of every skip list and compressed-tensors targets list."""
        lists = []
        for where, q in self.quant_configs():
            lists += [(where, k, q[k]) for k in SKIP_KEYS if isinstance(q.get(k), list)]
            for gname, group in (q.get("config_groups") or {}).items():
                if isinstance(group, dict) and isinstance(group.get("targets"), list):
                    lists.append((where, f"config_groups.{gname}.targets", group["targets"]))
        return lists

    # --- config edits ---

    def set_config(self, path: str, value: Any) -> None:
        old = get_path(self.config, path)
        set_path(self.config, path, value)
        self.log.append(f"config.json {path}: {json.dumps(old)} -> {json.dumps(value)}")

    def quant_skip_add(self, entries: Iterable[str]) -> None:
        """Add module names or patterns to the skip lists of every quantization config, creating the dialect's list if it has none.

        An empty list next to a non-empty one is left alone (ModelOpt readers take ``ignore`` only when it is non-empty); an fp8 list that is created or empty first gets the ``lm_head`` the dialect skips by default."""
        entries = list(entries)
        for where, q in self.quant_configs():
            method = self._method(where, q)
            keys = [k for k in SKIP_KEYS if isinstance(q.get(k), list)]
            if not keys:
                keys = ["exclude_modules" if where == SIDECAR else _DEFAULT_SKIP_KEY.get(method, "ignore")]
                q[keys[0]] = []
            if any(q[k] for k in keys):
                keys = [k for k in keys if q[k]]
            for key in keys:
                lst = q[key]
                implicit = ["lm_head"] if method == "fp8" and key == "modules_to_not_convert" and not lst else []
                new = list(dict.fromkeys(e for e in implicit + entries if e not in lst))
                lst.extend(new)
                if new:
                    self.log.append(f"{where} {key}: added {new}")

    def quant_skip_remove(self, entries: Iterable[str]) -> None:
        entries = set(entries)
        for where, q in self.quant_configs():
            for key in SKIP_KEYS:
                lst = q.get(key)
                if isinstance(lst, list):
                    gone = [e for e in lst if e in entries]
                    lst[:] = [e for e in lst if e not in entries]
                    if gone:
                        self.log.append(f"{where} {key}: removed {gone}")

    def quant_sub(self, pattern: str, repl: str) -> None:
        """re.sub over the text of every skip-list entry and compressed-tensors target, ``re:`` entries included."""
        self._sub_entries(lambda e: re.sub(pattern, repl, e))

    def quant_rename(self, pattern: str, repl: str) -> None:
        """Apply a tensor rename rule to the skip-list entries and targets that are module names or globs; ``re:`` entries are left alone.

        Each entry is matched as ``entry + "."``, so a rule written for tensor names also renames a container entry."""
        rx = re.compile(pattern)
        self._sub_entries(lambda e: e if e.startswith("re:") else rx.sub(repl, e + ".").removesuffix("."))

    def _sub_entries(self, fn: Callable[[str], str]) -> None:
        for where, key, lst in self.quant_entries():
            new = [fn(e) for e in lst]
            if new != lst:
                self.log.append(f"{where} {key}: {[e for e in lst if e not in new]} -> {[e for e in new if e not in lst]}")
                lst[:] = new

    def stale_quant_entries(self, old_modules: Iterable[str], new_modules: Iterable[str]) -> list[str]:
        """Skip-list entries and targets that named some of ``old_modules`` and name none of ``new_modules``."""
        old, new = list(old_modules), list(new_modules)
        stale = []
        for where, key, lst in self.quant_entries():
            for e in lst:
                if e != "Linear" and any(entry_names(e, m) for m in old) and not any(entry_names(e, m) for m in new):
                    stale.append(f"{where} {key}: {e!r}")
        return stale

    # --- tensor names ---

    def rename(self, fn: Callable[[str], str]) -> dict[str, str]:
        """Rename tensors by ``fn(name)``; returns {old: new} of the names that changed. Refuses a rename that makes two tensors share a name."""
        changes = {}
        for shard, h in self.headers.items():
            for k in h:
                if k != "__metadata__" and fn(k) != k:
                    changes[k] = (shard, fn(k))
        if not changes:
            return {}
        final = [changes[k][1] if k in changes else k for k in self.tensors()]
        if len(final) != len(set(final)):
            dup = sorted({n for n in final if final.count(n) > 1})[:5]
            raise ValueError(f"the rename gives two tensors the same name, e.g. {dup}")
        for old, (shard, new) in changes.items():
            h = self.headers[shard]
            h[new] = h.pop(old)
            self._changed_shards.add(shard)
        index = self.json.get(INDEX)
        if index is not None:
            wm = index["weight_map"]
            index["weight_map"] = {(changes[k][1] if k in changes else k): v for k, v in wm.items()}
        self.log.append(f"renamed {len(changes)} tensors in {len({s for s, _ in changes.values()})} shards")
        return {old: new for old, (_, new) in changes.items()}

    def rename_regex(self, rules: Iterable[tuple[str, str]]) -> dict[str, str]:
        """Apply ``re.sub(pattern, repl, name)`` for each rule in order."""
        compiled = [(re.compile(p), r) for p, r in rules]

        def fn(name: str) -> str:
            for rx, repl in compiled:
                name = rx.sub(repl, name)
            return name

        return self.rename(fn)

    # --- writing ---

    def _changed_json(self) -> list[str]:
        return [n for n, v in self.json.items() if v != self._orig_json.get(n)]

    def pending(self, out: str | None) -> dict:
        """What ``save(out)`` would write: {"json": [file names], "shards": {shard: how}}."""
        shards = {}
        for shard in sorted(self._changed_shards):
            p, n = os.path.join(self.path, shard), self.header_len[shard]
            fits = len(encode_header(self.headers[shard], fit=n)) == n
            if out is None and fits and not _shared(p):
                shards[shard] = "header patched in place"
            elif stub_data_end(p, n) is not None:
                shards[shard] = "rewritten (header-only stub, no data copied)"
            else:
                why = " (a shared symlink / hard link becomes a private copy)" if out is None and _shared(p) else ""
                shards[shard] = f"rewritten, copies {os.path.getsize(p) / 2**30:.1f} GiB{why}"
        return {"json": self._changed_json(), "shards": shards}

    def save(self, out: str | None = None) -> list[str]:
        """Write the edits; returns what was written. ``out=None`` edits in place (with a backup)."""
        changed_json, written = self._changed_json(), []
        if out is None:
            if not changed_json and not self._changed_shards:
                return []
            backup = os.path.join(self.path, BACKUP_DIR, time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 10**9:09d}")
            os.makedirs(backup)
            manifest = {"links": {}, "created": []}
            for name in changed_json:
                p = os.path.join(self.path, name)
                if os.path.lexists(p):
                    if os.path.islink(p):
                        manifest["links"][name] = os.readlink(p)
                    shutil.copyfile(p, os.path.join(backup, name))
                else:
                    manifest["created"].append(name)
            for shard in self._changed_shards:
                p = os.path.join(self.path, shard)
                if os.path.islink(p):
                    manifest["links"][shard] = os.readlink(p)
                with open(p, "rb") as f:
                    (n,) = struct.unpack("<Q", f.read(8))
                    head = f.read(n)
                with open(os.path.join(backup, shard + ".header"), "wb") as f:
                    f.write(head)
            with open(os.path.join(backup, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            for shard in sorted(self._changed_shards):
                p = os.path.join(self.path, shard)
                written.append(f"{shard} ({write_shard(p, p, self.headers[shard], private=_shared(p))})")
            target = self.path
        else:
            target = os.path.realpath(out)
            if target == os.path.realpath(self.path) or target.startswith(os.path.realpath(self.path) + os.sep):
                raise ValueError("--out must be outside the checkpoint directory")
            os.makedirs(target, exist_ok=True)
            for name in sorted(os.listdir(self.path)):
                src, dst = os.path.join(self.path, name), os.path.join(target, name)
                if name == BACKUP_DIR or name in changed_json or name in self._changed_shards or os.path.lexists(dst):
                    continue
                os.symlink(os.path.realpath(src), dst)
            for shard in sorted(self._changed_shards):
                write_shard(os.path.join(self.path, shard), os.path.join(target, shard), self.headers[shard])
                written.append(f"{shard} (rewritten)")
        for name in changed_json:
            dst = os.path.join(target, name)
            tmp = dst + ".ckpt_edit.part"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.json[name], f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            # replaces a symlink without writing through it
            os.replace(tmp, dst)
            written.append(name)
        self._orig_json = copy.deepcopy(self.json)
        self._changed_shards.clear()
        return written


def confirm(plan: dict, lines: list[str], plan_file: str | None) -> bool:
    """Print the plan, then decide whether to apply it.

    With ``plan_file`` (no prompt): a run whose plan is not already in the file writes it there and changes nothing; the next run of the same command applies it only if it computes the same plan from the same checkpoint state.
    Without: ask [y/N], and change nothing when stdin is not a terminal."""
    print("\n".join(lines))
    if plan_file:
        saved = None
        if os.path.exists(plan_file):
            try:
                with open(plan_file) as f:
                    saved = json.load(f)
            except (OSError, ValueError):
                saved = None
            if not isinstance(saved, dict) or "fingerprint" not in saved:
                raise SystemExit(f"--plan {plan_file} exists and is not a plan saved by these scripts; pick another path")
        if saved == plan:
            print(f"\nthis is the plan saved in {plan_file}: applying it")
            return True
        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=1)
        what = "differs from the plan saved there (the command or the checkpoint changed); saved the new one" if saved else "saved"
        print(f"\nplan {what} in {plan_file}. Nothing was changed: check the plan, then run the same command again to apply it.")
        return False
    if not sys.stdin.isatty():
        print("\nstdin is not a terminal, so nothing was changed. Pass --plan <file>: the first run saves the plan, a second identical run applies it.")
        return False
    return input("\napply this plan? [y/N] ").strip().lower() in ("y", "yes")


def plan_lines_for_save(ck: "Checkpoint", out: str | None) -> list[str]:
    todo = ck.pending(out)
    lines = [f"  write: {', '.join(todo['json']) or 'no JSON file'}"]
    lines += [f"  shard {s}: {how}" for s, how in todo["shards"].items()]
    if out is None:
        lines.append(f"  in place: the originals of these files go to {os.path.join(ck.path, BACKUP_DIR)}/ first (--restore puts them back)")
    else:
        lines.append(f"  into {os.path.realpath(out)}: every other file is a symlink to the original")
    return lines


def refuse_stub(path: str) -> None:
    """The fix scripts edit real checkpoints only: a fetch_headers.py stub of a remote repo is for diagnosis."""
    if os.path.isfile(os.path.join(path, STUB_MARKER)):
        raise SystemExit(f"{path} is a header-only stub of an HF repo, which is only diagnosed: download the checkpoint, run diagnose.py on the local directory, and fix that")


def check_out_dir(ck: "Checkpoint", out: str | None) -> None:
    if out is None:
        return
    target, root = os.path.realpath(out), os.path.realpath(ck.path)
    if target == root or target.startswith(root + os.sep):
        raise SystemExit("--out must be outside the checkpoint directory")
    if os.path.isdir(target) and os.listdir(target):
        raise SystemExit(f"--out {target} exists and is not empty; pick a new directory")


def restore(path: str) -> list[str]:
    """Put back the latest in-place backup of ``path``; returns what was restored."""
    root = os.path.join(path, BACKUP_DIR)
    stamps = sorted(os.listdir(root)) if os.path.isdir(root) else []
    if not stamps:
        raise FileNotFoundError(f"no backup under {root}")
    backup, done = os.path.join(root, stamps[-1]), []
    manifest_path = os.path.join(backup, "manifest.json")
    manifest = {"links": {}, "created": []}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    for name in sorted(os.listdir(backup)):
        if name == "manifest.json":
            continue
        target = name.removesuffix(".header")
        p = os.path.join(path, target)
        if target in manifest["links"]:
            # the edit made a private copy of a shared file: point the name back at the original
            tmp = p + ".ckpt_edit.link"
            os.symlink(manifest["links"][target], tmp)
            os.replace(tmp, p)
        elif name.endswith(".safetensors.header"):
            if os.path.islink(p):
                raise SystemExit(f"{p} is a symlink now; the backup was taken of a regular file, restore it by hand")
            with open(os.path.join(backup, name), "rb") as f:
                write_shard(p, p, f.read())
        else:
            shutil.copyfile(os.path.join(backup, name), p + ".ckpt_edit.part")
            os.replace(p + ".ckpt_edit.part", p)
        done.append(target)
    for name in manifest["created"]:
        if os.path.exists(os.path.join(path, name)):
            os.remove(os.path.join(path, name))
            done.append(f"{name} (removed, the edit created it)")
    shutil.rmtree(backup)
    return done
