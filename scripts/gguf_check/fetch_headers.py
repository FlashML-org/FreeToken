"""Write header-only stubs of the GGUF files in HF repos: the KV metadata and tensor table, in a sparse file of the real size.

    python scripts/gguf_check/fetch_headers.py <out_dir> [<repo> ...] [--search <query>]

The weights stay holes that read as zeros, so a stub takes a few MB of disk and ``check.py`` reads it like the real file.
Stubs land at ``<out_dir>/<repo>/<path in repo>``; a stub already there with the right size is kept, so a rerun only fetches what is missing.
Prints one JSON line per file. Gated repos read ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import struct
import sys
import threading

import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

_FIRST_READ, _NEXT_READ = 24 << 20, 16 << 20
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_F32 = 0
# tensors whose values a parse_gguf_config reads, so their bytes go into the stub
VALUE_TENSORS = ("rope_freqs.weight",)

_local = threading.local()


def _get_range(url: str, start: int, n: int) -> requests.Response:
    """Bytes [start, start + n) of ``url``, or fewer at the end of the file; refuses a reply that is not exactly that range."""
    r = _session().get(url, headers={"Range": f"bytes={start}-{start + n - 1}"}, timeout=300)
    r.raise_for_status()
    got = r.headers.get("Content-Range", "")
    if r.status_code != 206 or not got.startswith(f"bytes {start}-"):
        raise ValueError(f"{url}: asked for bytes {start}-{start + n - 1}, got status {r.status_code} {got!r}")
    end = int(got.split("-", 1)[1].split("/", 1)[0])
    if len(r.content) != end - start + 1:
        raise ValueError(f"{url}: short read at byte {start}")
    return r


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update(build_hf_headers())
    return _local.session


class _Remote:
    """Reads a remote file front to back through range requests, keeping every byte read so far."""

    def __init__(self, url: str):
        self.url, self.buf, self.pos, self.size = url, bytearray(), 0, None

    def _need(self, n: int) -> None:
        while len(self.buf) < self.pos + n:
            start = len(self.buf)
            if self.size is not None and start >= self.size:
                raise EOFError(f"{self.url}: header runs past the end of the file")
            r = _get_range(self.url, start, max(_FIRST_READ if start == 0 else _NEXT_READ, self.pos + n - start))
            if self.size is None:
                self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            self.buf += r.content

    def read(self, n: int) -> bytes:
        self._need(n)
        self.pos += n
        return bytes(self.buf[self.pos - n : self.pos])

    def unpack(self, fmt: str) -> tuple:
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))

    def string(self) -> str:
        (n,) = self.unpack("Q")
        return self.read(n).decode("utf-8", "replace")

    def skip_value(self, kind: int) -> None:
        if kind in _SCALAR:
            self.read(struct.calcsize(_SCALAR[kind]))
        elif kind == _STRING:
            self.string()
        elif kind == _ARRAY:
            elem, n = self.unpack("I")[0], self.unpack("Q")[0]
            if elem in _SCALAR:
                self.read(struct.calcsize(_SCALAR[elem]) * n)
            else:
                for _ in range(n):
                    self.skip_value(elem)
        else:
            raise ValueError(f"{self.url}: unknown GGUF value type {kind}")


def fetch(repo: str, path: str, size: int, out_dir: str) -> dict:
    dst = os.path.join(out_dir, repo, path)
    rec = {"repo": repo, "path": path, "stub": dst, "size": size, "value_tensors": list(VALUE_TENSORS)}
    # the sidecar marks a finished stub and which tensors it holds the values of
    sidecar = dst + ".json"
    if os.path.isfile(dst) and os.path.getsize(dst) == size and os.path.isfile(sidecar):
        with open(sidecar) as f:
            if json.load(f).get("value_tensors") == rec["value_tensors"]:
                return {**rec, "ok": True, "cached": True}
    url = hf_hub_url(repo, path)
    try:
        f = _Remote(url)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _, n_tensors, n_kv = f.unpack("IQQ")
        align = 32
        for _ in range(n_kv):
            key, kind = f.string(), f.unpack("I")[0]
            if key == "general.alignment" and kind in (4, 5):
                align = f.unpack("I")[0]
            else:
                f.skip_value(kind)
        tensors = {}
        for _ in range(n_tensors):
            name = f.string()
            (nd,) = f.unpack("I")
            dims = f.unpack(f"{nd}Q")
            kind, offset = f.unpack("IQ")
            tensors[name] = (dims, kind, offset)
        header_end = f.pos
        data_start = (header_end + align - 1) // align * align
        tmp = dst + ".part"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(tmp, "wb") as out:
            out.write(f.buf[:header_end])
            out.truncate(f.size)
            for name in VALUE_TENSORS:
                if name not in tensors:
                    continue
                dims, kind, offset = tensors[name]
                if kind != _F32 or len(dims) != 1:
                    raise ValueError(f"{name} is type {kind} with dims {dims}, expected 1-D F32")
                start, n = data_start + offset, dims[0] * 4
                r = _get_range(url, start, n)
                if len(r.content) != n:
                    raise ValueError(f"{name} runs past the end of the file")
                out.seek(start)
                out.write(r.content)
        os.replace(tmp, dst)
        rec.update(header_bytes=header_end, tensors=n_tensors)
        with open(sidecar, "w") as f:
            json.dump(rec, f)
        return {**rec, "ok": True}
    except Exception as e:
        return {**rec, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def list_gguf(api: HfApi, repo: str) -> list[tuple[str, int]]:
    """(path, size) of every model GGUF in the repo; vision projectors and imatrix files are left out."""
    files = []
    for entry in api.list_repo_tree(repo, recursive=True):
        path = entry.path
        if path.endswith(".gguf") and not any(s in path.lower() for s in ("mmproj", "imatrix")):
            files.append((path, entry.size))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--search", action="append", default=[], help="also take every HF GGUF repo whose name matches this query (repeatable)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    api = HfApi()
    repos = list(args.repos)
    for query in args.search:
        repos += [m.id for m in api.list_models(search=query, filter="gguf") if m.id not in repos]
    jobs = []
    for repo in repos:
        try:
            jobs += [(repo, path, size) for path, size in list_gguf(api, repo)]
        except Exception as e:
            print(json.dumps({"repo": repo, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
    print(f"{len(jobs)} GGUF files in {len(repos)} repos", file=sys.stderr, flush=True)
    failed = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for rec in ex.map(lambda job: fetch(*job, args.out_dir), jobs):
            failed += not rec["ok"]
            print(json.dumps(rec), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
