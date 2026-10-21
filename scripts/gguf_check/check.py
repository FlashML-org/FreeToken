"""Check that FreeToken loads GGUF files, on the CPU and without reading their weights.

    python scripts/gguf_check/check.py --kernels kernels.json --out results.jsonl <gguf file or dir> ...
    python scripts/gguf_check/check.py --report results.jsonl

Per file, through the code the server runs:

- config: the family's ``parse_gguf_config`` accepts the file (unsupported files must fail here with a clear error)
- dry load: the model builds on the meta device, and the family's weight reader yields exactly its state dict (names, shapes, packed vs float); the expert banks build for every MoE layer with the shapes the model needs
- kernels: every quantized ggml type a GGUF linear, embedding or expert bank uses has the ops it needs in the ``probe_kernels.py`` table
- tokenizer: it builds, the chat template renders, and the file names an EOS token

The weight reader runs on meta tensors, so no weight bytes are read or copied. A reader that needs real values (a numpy dequant) gets a second run on the real bytes, once enough RAM is free.
The files can be real or ``fetch_headers.py`` stubs. Passing does not prove the weights are right or that the model fits in memory.
Results append to ``--out``; a rerun skips the files already there, except crashed ones (and failed ones with ``--retry-failed``).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import functools
import glob
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import struct
import sys
import time
import traceback
import warnings
from collections import Counter

MOE_TOKENS = ("mmid1", "mmid8", "mmid16", "mmid512")
# ops each layer kind runs over the token counts the server sends; fused-glu experts also need glu<T> up to the type's mmvq_max_tokens
REQUIRED_OPS = {
    "linear": ("mm1", "mm8", "mm16", "mm512"),
    "embedding": ("get_rows",),
    "experts gate_up": MOE_TOKENS,
    "experts down": MOE_TOKENS,
}
GLU_TOKENS = (1, 8)
_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_UINT32 = 4


class _Field:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class _Tensor:
    __slots__ = ("name", "shape", "tensor_type", "data_offset", "n_bytes")


class FastReader:
    """The parts of gguf-py's ``GGUFReader`` FreeToken reads, parsed in pure Python, with the same checks.

    gguf-py builds numpy views for every field and takes seconds and about 0.7 GB per large file, which a scan over many files cannot afford."""

    def __init__(self, path: str):
        import numpy as np

        from freetoken.third_party.gguf import GGML_QUANT_SIZES, GGMLQuantizationType

        self.data = data = np.memmap(path, dtype=np.uint8, mode="r")
        pos = 0

        def unpack(fmt):
            nonlocal pos
            vals = struct.unpack_from("<" + fmt, data, pos)
            pos += struct.calcsize("<" + fmt)
            return vals

        def string():
            nonlocal pos
            (n,) = unpack("Q")
            pos += n
            return data[pos - n : pos].tobytes().decode("utf-8")

        def value(kind):
            if kind in _SCALAR:
                return unpack(_SCALAR[kind])[0]
            if kind == _STRING:
                return string()
            if kind != _ARRAY:
                raise ValueError(f"{path}: unknown GGUF value type {kind}")
            elem, n = unpack("I")[0], unpack("Q")[0]
            if elem in _SCALAR:
                return list(unpack(f"{n}{_SCALAR[elem]}"))
            return [value(elem) for _ in range(n)]

        if data[:4].tobytes() != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        pos = 4
        _, n_tensors, n_kv = unpack("IQQ")
        self.fields = {}
        align = 32
        for _ in range(n_kv):
            key = string()
            if key in self.fields:
                raise KeyError(f"{path}: duplicate GGUF key {key}")
            kind = unpack("I")[0]
            self.fields[key] = _Field(value(kind))
            if key == "general.alignment":
                align = self.fields[key].value
                if kind != _UINT32 or align <= 0 or align & (align - 1):
                    raise ValueError(f"{path}: bad general.alignment {align}")
        infos = []
        for _ in range(n_tensors):
            name = string()
            (nd,) = unpack("I")
            dims = unpack(f"{nd}Q")
            kind, offset = unpack("IQ")
            infos.append((name, dims, kind, offset))
        start = (pos + align - 1) // align * align
        self.tensors, names = [], set()
        for name, dims, kind, offset in infos:
            if name in names:
                raise ValueError(f"{path}: duplicate tensor {name}")
            names.add(name)
            block, size = GGML_QUANT_SIZES[GGMLQuantizationType(kind)]
            t = _Tensor()
            t.name, t.shape, t.tensor_type, t.data_offset = name, list(dims), kind, start + offset
            t.n_bytes = _prod(dims) // block * size
            if t.data_offset + t.n_bytes > len(data):
                raise ValueError(f"{path}: tensor {name} runs past the end of the file")
            self.tensors.append(t)


def _prod(dims) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def _clear_caches() -> None:
    """Drop the caches that hold a file's metadata, config or rope tables."""
    import freetoken.models.gguf.reader as reader
    from freetoken.layers import rotary
    from freetoken.utils import hf

    for fn in (reader._reader, reader.load_gguf_metadata, reader.gguf_split_paths, reader.gguf_tensor_index,
               rotary.get_rope, rotary._cos_sin_table, hf._load_hf_config):
        fn.cache_clear()


_KERNELS: dict | None = None
_TOKENIZERS: dict = {}
_IN_FLIGHT = None
_MIN_FREE_GB = 32.0


def _init_worker(kernels: dict | None, in_flight, min_free_gb: float) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import torch

    import freetoken.models.gguf.reader as reader
    from freetoken.distributed import set_tp_info
    from freetoken.layers import rotary

    reader._reader = functools.cache(FastReader)
    set_tp_info(rank=0, size=1)
    rotary.set_rope_device(torch.device("cpu"))
    global _KERNELS, _IN_FLIGHT, _MIN_FREE_GB
    _KERNELS, _IN_FLIGHT, _MIN_FREE_GB = kernels, in_flight, min_free_gb


def _build_model(path: str):
    """The engine's config and the model it builds for ``path`` (experts offloaded), on the meta device."""
    import torch

    from freetoken.distributed import try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    config = EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy="offload",
                          mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)))
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return config, create_model(config.model_config)


def _gguf_layers(model) -> tuple[list, bool]:
    """(kind, ggml types) of every GGUF linear and embedding, and whether any GGUF MoE layer runs the fused-glu kernel."""
    from freetoken.layers.base import BaseOP
    from freetoken.layers.quantization import QuantKind
    from freetoken.layers.quantization.moe.gguf import fused_glu
    from freetoken.layers.quantization.registry import LayerKind
    from freetoken.layers.quantization.scheme import gguf_types

    kinds = {LayerKind.LINEAR: "linear", LayerKind.EMBEDDING: "embedding"}
    seen, out, glu = set(), [], False

    def walk(op):
        nonlocal glu
        if id(op) in seen:
            return
        seen.add(id(op))
        method = getattr(op, "quant_method", None)
        if method is not None and method.kind is QuantKind.GGUF:
            if method.layer_kind in kinds:
                out.append((kinds[method.layer_kind], gguf_types(method.scheme)))
            elif method.layer_kind is LayerKind.MOE:
                glu = glu or fused_glu(op) is not None
        for value in vars(op).values():
            for item in value if isinstance(value, (list, tuple)) else (value,):
                if isinstance(item, BaseOP):
                    walk(item)

    walk(model)
    return out, glu


def _check_kernels(layers, tied: bool, glu: bool, expert_tags, problems: list[str]) -> dict:
    from freetoken.layers.quantization.ggml_types import is_quantized, type_name
    from freetoken.third_party.gguf import GGMLQuantizationType

    used: dict[str, Counter] = {}
    if tied:
        layers = layers + [("linear", types) for kind, types in layers if kind == "embedding"]
    for kind, types in layers:
        for t in types:
            used.setdefault(kind, Counter())[type_name(t)] += 1
    for _, gate_up, down in expert_tags:
        used.setdefault("experts gate_up", Counter())[gate_up] += 1
        used.setdefault("experts down", Counter())[down] += 1
    if _KERNELS is not None:
        for kind, names in used.items():
            for name in names:
                if not is_quantized(int(GGMLQuantizationType[name])):
                    continue
                probe = _KERNELS.get(name, {})
                need = list(REQUIRED_OPS[kind])
                if kind == "experts gate_up" and glu:
                    need += [f"glu{t}" for t in GLU_TOKENS if t <= probe.get("mmvq_max_tokens", 0)]
                missing = [op for op in need if probe.get(op) != "ok"]
                if missing:
                    problems.append(f"{kind} {name}: no {', '.join(missing)} kernel")
    return {k: dict(v) for k, v in used.items()}


@functools.cache
def _meta_readers():
    """The GGUF tensor readers, and stand-ins that give matrices as meta tensors."""
    import torch

    from freetoken.models.gguf import tensors
    from freetoken.models.gguf.tensors import _FLOAT_DTYPES

    real_float = tensors.float_tensor

    def float_tensor(info):
        # 1-D tensors (norms, biases, rope_freqs) are small, and config parses read their values
        if len(info.ne) == 1:
            return real_float(info)
        return torch.empty(tuple(reversed(info.ne)), dtype=_FLOAT_DTYPES[info.ggml_type], device="meta")

    def packed_rows(info):
        rows = _prod(info.ne[1:])
        return torch.empty(rows, info.nbytes // rows, dtype=torch.uint8, device="meta")

    fakes = {"float_tensor": float_tensor, "packed_rows": packed_rows}
    real = {name: getattr(tensors, name) for name in fakes}
    return real, fakes


@functools.cache
def _meta_wins():
    """A torch function mode that moves CPU operands to meta when an op mixes the two, so a reader can combine meta weights with the small tensors it builds."""
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._pytree import tree_flatten, tree_unflatten

    class MetaWins(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            flat, spec = tree_flatten((args, kwargs or {}))
            if any(isinstance(x, torch.Tensor) and x.is_meta for x in flat):
                flat = [x.to("meta") if isinstance(x, torch.Tensor) and not x.is_meta else x for x in flat]
            args, kwargs = tree_unflatten(flat, spec)
            return func(*args, **kwargs)

    return MetaWins


def _swap_readers(use_fakes: bool) -> None:
    """Point every loaded binding of the readers, including the family modules' imported names, at the fakes or back."""
    real, fakes = _meta_readers()
    src, dst = (real, fakes) if use_fakes else (fakes, real)
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("freetoken.models"):
            for name, fn in src.items():
                if getattr(module, name, None) is fn:
                    setattr(module, name, dst[name])


def _wait_for_ram(gb: float, timeout: float = 600.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open("/proc/meminfo") as f:
            avail = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:")) / 2**20
        if avail >= gb:
            return True
        time.sleep(2)
    return False


def _dry_load(path: str, config, model, *, meta: bool) -> list[str]:
    import torch

    from freetoken.models.register import _load_attr

    spec = config.model_spec
    expected = model.state_dict()
    problems, seen = [], set()
    _swap_readers(meta)
    try:
        with _meta_wins()() if meta else contextlib.nullcontext():
            for key, t in _load_attr(spec.module, spec.iter_weights)(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True):
                ref = expected.get(key)
                if ref is None:
                    problems.append(f"unexpected tensor {key}")
                    continue
                if key in seen:
                    problems.append(f"tensor {key} yielded twice")
                seen.add(key)
                if tuple(t.shape) != tuple(ref.shape) or (t.dtype == torch.uint8) != (ref.dtype == torch.uint8):
                    problems.append(f"{key}: got {tuple(t.shape)} {t.dtype}, model has {tuple(ref.shape)} {ref.dtype}")
    finally:
        _swap_readers(False)
    missing = sorted(set(expected) - seen)
    if missing:
        problems.append(f"{len(missing)} tensors never loaded, e.g. {missing[:3]}")
    return problems


def _expert_banks(path: str, cfg, problems: list[str]) -> list:
    """Build the expert banks on the meta device and check their shapes; returns each MoE layer's (format, gate_up type, down type)."""
    import torch

    if not cfg.num_experts:
        return []
    from freetoken.layers.quantization.ggml_types import float_dtype, row_bytes
    from freetoken.models.gguf.experts import load_gguf_expert_banks
    from freetoken.third_party.gguf import GGMLQuantizationType

    banks = load_gguf_expert_banks(path, cfg.num_moe_layers, first_layer=cfg.first_k_dense_replace, dummy=True, device=torch.device("meta"))
    e, inter, hidden = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    for layer, (_, gate_up, down) in enumerate(banks.layer_tags):
        for role, name, rows, k in (("gate_up", gate_up, 2 * inter, hidden), ("down", down, hidden, inter)):
            t = int(GGMLQuantizationType[name])
            want = (e, rows, k if float_dtype(t) is not None else row_bytes(t, k))
            got = tuple(banks.sources[role][layer].shape)
            if got != want:
                problems.append(f"MoE layer {layer} {role} bank is {got}, the model needs {want}")
    return banks.layer_tags


def _tokenizer_key(meta: dict) -> str:
    blob = json.dumps(sorted((k, v) for k, v in meta.items() if k.startswith("tokenizer.")))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _check_tokenizer(path: str) -> dict:
    from freetoken.models.gguf.tokenizer import gguf_eos_token_ids, load_gguf_tokenizer

    problems = []
    tok = load_gguf_tokenizer(path)
    eos = sorted(gguf_eos_token_ids(path))
    if not eos:
        problems.append("tokenizer: no EOS token")
    tail = None
    if tok.chat_template is None:
        problems.append("tokenizer: no chat template")
    else:
        try:
            tail = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)[-60:]
        except Exception as e:
            problems.append(f"chat template: {type(e).__name__}: {str(e)[:150]}")
    return {"problems": problems, "eos": eos, "template_tail": tail}


def check(path: str) -> dict:
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
    from freetoken.models.gguf.reader import gguf_architecture, load_gguf_metadata

    if _IN_FLIGHT is not None:
        _IN_FLIGHT[os.getpid()] = path
    t0 = time.time()
    out = {"path": path, "problems": []}
    problems = out["problems"]
    try:
        out["arch"] = gguf_architecture(path)
        if out["arch"] not in GGUF_ARCH_TO_REGISTRY:
            out["status"] = "unsupported"
            return out
        config, model = _build_model(path)
        cfg = config.model_config
        out["config"] = {"layers": cfg.num_layers, "hidden": cfg.hidden_size, "experts": cfg.num_experts, "vocab": cfg.vocab_size, "tied": cfg.tie_word_embeddings}
        out["reads"] = "meta"
        try:
            load_problems = _dry_load(path, config, model, meta=True)
        except Exception:
            out["reads"] = "real"
            if not _wait_for_ram(_MIN_FREE_GB):
                raise MemoryError(f"the weight reader needs real bytes and free RAM stayed under {_MIN_FREE_GB} GB") from None
            load_problems = _dry_load(path, config, model, meta=False)
        problems.extend(load_problems)
        tags = _expert_banks(path, cfg, problems)
        layers, glu = _gguf_layers(model)
        out["types"] = _check_kernels(layers, cfg.tie_word_embeddings, glu, tags, problems)
        meta = load_gguf_metadata(path)
        key = _tokenizer_key(meta)
        if key not in _TOKENIZERS:
            try:
                _TOKENIZERS[key] = _check_tokenizer(path)
            except Exception as e:
                _TOKENIZERS[key] = {"problems": [f"tokenizer: {type(e).__name__}: {str(e)[:200]}"]}
        tok = _TOKENIZERS[key]
        problems.extend(tok["problems"])
        out["tokenizer"] = {"hash": key, "pre": meta.get("tokenizer.ggml.pre"), "eos": tok.get("eos"), "template_tail": tok.get("template_tail")}
    except Exception as e:
        problems.append(f"{type(e).__name__}: {str(e)[:300]}")
        out["traceback"] = traceback.format_exc()[-1500:]
    finally:
        _clear_caches()
        out["seconds"] = round(time.time() - t0, 1)
        if _IN_FLIGHT is not None:
            _IN_FLIGHT.pop(os.getpid(), None)
    out.setdefault("status", "fail" if problems else "pass")
    return out


def gguf_files(paths: list[str]) -> list[str]:
    """Every model GGUF under ``paths``, once; a split model is checked through its first file."""
    files = []
    for p in paths:
        found = sorted(glob.glob(os.path.join(p, "**", "*.gguf"), recursive=True)) if os.path.isdir(p) else [p]
        for f in found:
            m = _SPLIT.search(f)
            if os.path.isfile(f) and (m is None or m.group(1) == "00001") and not any(s in os.path.basename(f).lower() for s in ("mmproj", "imatrix")):
                files.append(os.path.abspath(f))
    return list(dict.fromkeys(files))


def read_results(path: str) -> list[dict]:
    """The latest record per file; a line cut short by a killed run is dropped."""
    latest = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[rec["path"]] = rec
    return list(latest.values())


def _problem_kind(problem: str) -> str:
    """A problem with tensor names, layer numbers and sizes masked, to group the same failure across files."""
    problem = re.sub(r"blk\.\d+", "blk.N", problem)
    problem = re.sub(r"model\.layers\.\d+", "model.layers.N", problem)
    problem = re.sub(r"/\S+\.gguf", "<file>", problem)
    return re.sub(r"\b\d+\b", "N", problem)[:160]


def report(results: list[dict]) -> None:
    supported = [r for r in results if r["status"] != "unsupported"]
    by_arch: dict[str, list[int]] = {}
    for r in supported:
        stats = by_arch.setdefault(r.get("arch") or "?", [0, 0])
        stats[0] += r["status"] == "pass"
        stats[1] += 1
    print(f"{sum(r['status'] == 'pass' for r in supported)}/{len(supported)} files of served architectures pass")
    for arch, (ok, n) in sorted(by_arch.items()):
        print(f"  {arch:16s} {ok}/{n}")
    unsupported = Counter(r.get("arch") for r in results if r["status"] == "unsupported")
    if unsupported:
        print("not checked, architecture not served: " + ", ".join(f"{a} {n}" for a, n in unsupported.most_common()))
    real = sum(r.get("reads") == "real" for r in supported)
    if real:
        print(f"{real} files needed real weight reads")
    kinds = Counter(_problem_kind(r["problems"][0]) for r in supported if r["status"] != "pass")
    if kinds:
        print("failures, by first problem:")
        for kind, n in kinds.most_common():
            print(f"  {n:5d}  {kind}")


class _Pool:
    """Worker processes that check files, and know which files were running when a worker died."""

    def __init__(self, kernels: dict | None, min_free_gb: float):
        self.ctx = mp.get_context("spawn")
        self.manager = self.ctx.Manager()
        self.in_flight = self.manager.dict()
        self.args = (kernels, self.in_flight, min_free_gb)

    def run(self, files: list[str], workers: int):
        """Yields (path, record), or (path, None) for a file that was running when a worker died."""
        pending = list(files)
        while pending:
            self.in_flight.clear()
            # no max_tasks_per_child: on Python 3.12 the pool hangs once every worker has retired
            with cf.ProcessPoolExecutor(min(workers, len(pending)), mp_context=self.ctx, initializer=_init_worker, initargs=self.args) as pool:
                futures = {pool.submit(check, f): f for f in pending}
                pending, broken = [], False
                for fut in cf.as_completed(futures):
                    try:
                        yield futures[fut], fut.result()
                    except cf.process.BrokenProcessPool:
                        broken = True
                        pending.append(futures[fut])
            if broken:
                # a worker that died before taking a file leaves none in flight: blame every file left
                running = set(self.in_flight.values()) & set(pending) or set(pending)
                for path in running:
                    yield path, None
                pending = [p for p in pending if p not in running]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="GGUF files or directories to search")
    ap.add_argument("--out", help="results JSONL, appended to")
    ap.add_argument("--kernels", help="kernel table from probe_kernels.py; without it the kernel check is skipped")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=32.0, help="free RAM a worker waits for before reading real weight bytes")
    ap.add_argument("--retry-failed", action="store_true", help="check again the files that failed in --out")
    ap.add_argument("--report", metavar="JSONL", help="only print the summary of an earlier run")
    args = ap.parse_args()

    if args.report:
        report(read_results(args.report))
        return 0
    if not args.out or not args.paths:
        ap.error("give --out and at least one path")
    kernels = None
    if args.kernels:
        with open(args.kernels) as f:
            kernels = json.load(f)["types"]
    else:
        print("no --kernels table: the kernel check is skipped", file=sys.stderr)
    keep = {"pass", "unsupported"} if args.retry_failed else {"pass", "unsupported", "fail"}
    done = {r["path"] for r in read_results(args.out) if r["status"] in keep}
    files = [f for f in gguf_files(args.paths) if f not in done]
    print(f"{len(files)} files to check, {len(done)} already in {args.out}", file=sys.stderr, flush=True)

    pool = _Pool(kernels, args.min_free_gb)
    with open(args.out, "a") as out:

        def write(rec):
            mark = {"pass": "ok  ", "unsupported": "skip"}.get(rec["status"], "FAIL")
            print(f"{mark} {rec['path']}" + (f"  {rec['problems'][0][:120]}" if rec["problems"] else ""), file=sys.stderr, flush=True)
            out.write(json.dumps(rec) + "\n")
            out.flush()

        crashed = []
        for path, rec in pool.run(files, args.workers):
            if rec is None:
                crashed.append(path)
            else:
                write(rec)
        # each file that was running when a worker died runs alone, to find the one that kills it
        for path in crashed:
            for _, rec in pool.run([path], 1):
                write(rec or {"path": path, "status": "crash", "problems": ["worker process died (out of memory or a crash)"]})
    results = read_results(args.out)
    report(results)
    return 0 if all(r["status"] in ("pass", "unsupported") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
