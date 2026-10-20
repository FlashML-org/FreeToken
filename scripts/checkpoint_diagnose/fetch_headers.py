"""Write header-only stubs of HF safetensors checkpoints: the config files and every shard's header, in sparse files of the real size.

    python scripts/checkpoint_diagnose/fetch_headers.py <out_dir> <repo>[@<revision>] ...

The tensor data stays holes that read as zeros, so a stub takes a few MB of disk and ``diagnose.py`` reads it like the real checkpoint.
Stubs land at ``<out_dir>/<repo>``; a rerun keeps the files that already have the right size.
Python files are never fetched. Gated repos read ``HF_TOKEN``. Prints one JSON line per repo.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import struct
import sys
import threading
import time

import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

STUB_MARKER = ".stub.json"
CONFIG_EXT = (".json", ".jinja")
# large tokenizer files: the load check never reads them
SKIP = {"tokenizer.json", "vocab.json", "merges.txt"}
_FIRST_READ = 1 << 20

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update(build_hf_headers())
    return _local.session


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """The file, or bytes [start, end] of it; retries rate limits and server errors."""
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for attempt in range(6):
        r = _session().get(url, headers=headers, timeout=300)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        if start is not None and r.status_code != 206:
            raise ValueError(f"{url}: asked for a byte range, got status {r.status_code}")
        return r.content
    r.raise_for_status()
    raise RuntimeError(f"{url}: status {r.status_code}")


def _fetch_shard(url: str, dst: str, size: int) -> None:
    if os.path.isfile(dst) and os.path.getsize(dst) == size:
        return
    head = _get(url, 0, min(size, _FIRST_READ) - 1)
    (n,) = struct.unpack("<Q", head[:8])
    if 8 + n > len(head):
        head += _get(url, len(head), 8 + n - 1)
    head = head[: 8 + n]
    json.loads(head[8:])
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.write(head)
        f.truncate(size)
    os.replace(tmp, dst)


def _fetch_file(url: str, dst: str, size: int | None) -> None:
    if os.path.isfile(dst) and size is not None and os.path.getsize(dst) == size:
        return
    data = _get(url)
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)


def fetch_repo(repo: str, out_dir: str, *, workers: int = 8) -> dict:
    """Write the stub of ``repo`` (``name`` or ``name@revision``) under ``out_dir``; returns its record, with ``stub`` the directory."""
    name, _, revision = repo.partition("@")
    info = HfApi().model_info(name, revision=revision or None, files_metadata=True)
    files = [(s.rfilename, s.size) for s in info.siblings or []]
    root = [(f, size) for f, size in files if "/" not in f]
    shards = [(f, size) for f, size in root if f.endswith(".safetensors")]
    configs = [(f, size) for f, size in root if f.endswith(CONFIG_EXT) and f not in SKIP]
    dst = os.path.join(out_dir, name)
    rec = {"repo": name, "revision": info.sha, "stub": dst, "shards": len(shards),
           "gguf": sum(f.endswith(".gguf") for f, _ in files),
           "other_weights": sorted({os.path.splitext(f)[1] for f, _ in files if f.endswith((".bin", ".pt", ".pth", ".npz"))}),
           "files": [f for f, _ in configs + shards]}
    marker = os.path.join(dst, STUB_MARKER)
    if os.path.isfile(marker):
        with open(marker) as f:
            old = json.load(f)
        if old.get("revision") != info.sha:
            # another revision: none of its files can be kept
            for f in old.get("files", []):
                if os.path.isfile(os.path.join(dst, f)):
                    os.remove(os.path.join(dst, f))
            os.remove(marker)
    os.makedirs(dst, exist_ok=True)
    jobs = [(_fetch_file, f, size) for f, size in configs] + [(_fetch_shard, f, size) for f, size in shards]
    with cf.ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(fn, hf_hub_url(name, f, revision=info.sha), os.path.join(dst, f), size) for fn, f, size in jobs]
        for fut in futures:
            fut.result()
    with open(os.path.join(dst, STUB_MARKER), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def is_stub(path: str) -> bool:
    return os.path.isfile(os.path.join(path, STUB_MARKER))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("repos", nargs="+", help="HF repo ids, optionally with @revision")
    args = ap.parse_args()
    failed = 0
    for repo in args.repos:
        try:
            rec = {**fetch_repo(repo, args.out_dir), "ok": True}
        except Exception as e:
            rec = {"repo": repo, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            failed += 1
        print(json.dumps(rec), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
