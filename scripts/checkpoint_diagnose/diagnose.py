"""Diagnose why FreeToken does not load an HF safetensors checkpoint, without reading its weights.

    python scripts/checkpoint_diagnose/diagnose.py <ckpt dir or HF repo[@revision]> ... [--out report.jsonl] [--text-model-only]

A repo id is fetched as a header-only stub (``fetch_headers.py``) into ~/.cache/freetoken/checkpoint-stubs first. GGUF and FTW checkpoints are pointed elsewhere.
Per checkpoint, through the code the server runs, on the CPU and the meta device:

- config: the architecture is registered and the quantization_config parses
- build: the model builds with a quant method for every layer
- dense load: the family's weight reader yields exactly the model's state dict (names and shapes)
- experts: every MoE layer's expert banks fill, packed by the layer's expert method

The readers get meta tensors with the header's shapes and dtypes; a reader that needs values gets a second run on the real bytes (zeros in a stub).
On a failure it also checks the tensor names against transformers' own model and its checkpoint conversion rules, and lints the files for the known non-standard forms.
The failure is then put in one category (see README.md), with the fixes the fix scripts can apply; ``--out`` keeps them for ``--from-report``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import functools
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STUB_DIR = os.path.expanduser("~/.cache/freetoken/checkpoint-stubs")

CATEGORIES = {
    "arch": "architecture not served",
    "quant_method": "quantization method not supported",
    "quant_model": "quantization method supported, but not for this model or layer",
    "loader": "model supported, the loader does not read this checkpoint",
    "packing": "the checkpoint packs tensors in a form the loader does not read",
    "checkpoint": "the checkpoint itself is non-standard",
}
FIX_SCRIPTS = {"config": "fix_config.py", "rename": "rename_tensors.py"}
# tensors stored next to a quantized weight; they fold into the module's .weight when the names are compared with transformers
QUANT_COMPANIONS = (
    "weight_scale", "weight_scale_2", "weight_scale_inv", "weight_global_scale", "input_scale", "input_global_scale",
    "weight_zero_point", "k_scale", "v_scale", "q_scale", "prob_scale", "output_scale", "weight_shape",
)
_QUANT_DTYPES = ("U8", "I8", "F8_E4M3", "F8_E5M2", "I32", "F4")
_DIGITS = re.compile(r"\.\d+(?=\.|$)")
# known non-standard prefixes: (pattern, replacement, what is wrong, checkpoint root the family expects)
PREFIX_RULES = (
    (r"^model\.language_model\.(?:language_model\.)+", "model.language_model.", "the text tower is nested more than once under language_model", "model.language_model"),
    (r"^model\.language_model\.visual\.", "model.visual.", "the vision tower is stored under the text tower", "model.visual"),
    (r"^language_model\.model\.", "model.language_model.", "the text tower uses the older language_model.model. layout", "model.language_model"),
    (r"^language_model\.lm_head\.", "lm_head.", "lm_head uses the older language_model.lm_head layout", "model.language_model"),
)
# fp8 KV calibration scales under the projection's name; vLLM reads them as attn.k_scale, which FreeToken drops
KV_SCALE_RULE = (r"\.([kv])_proj\.output_scale$", r".\1_scale")
# the families whose weight reader drops .k_scale / .v_scale tensors; for the others the rename only moves the unexpected key
KV_SCALE_DROPPERS = ("freetoken.models.qwen3_5_moe",)
_PER_EXPERT = re.compile(r"^(?P<pre>.*\.layers\.)(?P<layer>\d+)(?P<mid>\..*\.experts\.)(?P<e>\d+)\.(?P<proj>[^.]+)\.")
# compressed-tensors targets written with vLLM's internal module names
TARGET_SWAPS = ((r"language_model\[\.\]model\[\.\]", "model[.]language_model[.]"), (r"language_model\\\.model\\\.", r"model\\.language_model\\."))
# per-module tensors a quant method's reader produces next to the weight
_SCALE_ROLES = (".weight_scale", ".weight_scale_inv", ".weight_scale_2", ".weight_global", ".input_scale")


def pattern_of(name: str) -> str:
    return _DIGITS.sub(".N", name)


def group_names(names, limit: int = 4) -> str:
    """Names grouped by their pattern (layer and expert numbers as N), most common first."""
    c = Counter(pattern_of(n) for n in names)
    parts = [f"{p} x{n}" if n > 1 else p for p, n in c.most_common(limit)]
    return ", ".join(parts) + (f", ... ({len(c) - limit} more patterns)" if len(c) > limit else "")


def shard_files(path: str) -> list[str]:
    """Root *.safetensors names, consolidated.safetensors only when it is the only one; subdirectory layouts are reached through safetensors_weight_map below."""
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    return [f for f in files if f != "consolidated.safetensors"] or files


def read_tensor_table(path: str) -> dict[str, dict]:
    """name -> {dtype, shape, shard} from every shard header."""
    table = {}
    for f in shard_files(path):
        for k, v in _header(os.path.join(path, f)).items():
            if k != "__metadata__":
                table[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": f}
    return table


_ITEM_BITS = {"F64": 64, "I64": 64, "U64": 64, "F32": 32, "I32": 32, "U32": 32, "F16": 16, "BF16": 16, "I16": 16, "U16": 16,
              "I8": 8, "U8": 8, "BOOL": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "F4": 4}


def check_files(path: str) -> list[str]:
    """What safetensors and the loaders would refuse in the files themselves: data offsets that leave gaps or do not cover the file, a tensor whose bytes do not match its shape, an index that disagrees with the shard headers."""
    problems, names = [], {}
    for f in shard_files(path):
        p = os.path.join(path, f)
        with open(p, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
        header = _header(p)
        spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in header.items() if k != "__metadata__")
        pos = 0
        for start, end, k in spans:
            if start != pos:
                problems.append(f"{f}: tensor {k} starts at byte {start} of the data, expected {pos} (gap or overlap)")
                break
            bits = _ITEM_BITS.get(header[k]["dtype"])
            numel = 1
            for d in header[k]["shape"]:
                numel *= d
            if bits is not None and end - start != numel * bits // 8:
                problems.append(f"{f}: tensor {k} has {end - start} bytes, its shape {header[k]['shape']} {header[k]['dtype']} needs {numel * bits // 8}")
                break
            pos = end
        size = os.path.getsize(p) - 8 - n
        if not problems and pos != size:
            problems.append(f"{f}: the tensors cover {pos} data bytes but the file has {size} (an incomplete download?)")
        names.update({k: f for k in header if k != "__metadata__"})
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as fh:
            wm = json.load(fh).get("weight_map", {})
        read = set(shard_files(path))
        # entries for shards the loaders skip (a subdir, an unread consolidated.safetensors) say nothing about the read ones
        wm = {k: v for k, v in wm.items() if "/" not in v and (v in read or v != "consolidated.safetensors")}
        absent = [k for k in wm if k not in names]
        unlisted = [k for k in names if k not in wm]
        moved = [k for k in wm if k in names and wm[k] != names[k]]
        if absent:
            problems.append(f"the index lists {len(absent)} tensors no shard holds: {group_names(absent)}")
        if unlisted:
            problems.append(f"the shards hold {len(unlisted)} tensors the index does not list: {group_names(unlisted)}")
        if moved:
            problems.append(f"the index puts {len(moved)} tensors in another shard than the one that holds them: {group_names(moved)}")
    return problems


def split_module(name: str) -> tuple[str, str]:
    """(module, leaf) of a tensor name: model.layers.0.mlp.gate.weight_scale -> (model.layers.0.mlp.gate, weight_scale)."""
    module, _, leaf = name.rpartition(".")
    return module, leaf


def stored_modules(table: dict) -> dict[str, tuple[bool, str]]:
    """module -> (stored quantized, weight dtype) for every module with a 2-D weight or a packed weight."""
    out = {}
    by_module: dict[str, dict[str, dict]] = {}
    for name, meta in table.items():
        module, leaf = split_module(name)
        by_module.setdefault(module, {})[leaf] = meta
    for module, leaves in by_module.items():
        w = leaves.get("weight_packed") or leaves.get("weight")
        if w is None or (len(w["shape"]) != 2 and "weight_packed" not in leaves):
            continue
        quantized = "weight_packed" in leaves or w["dtype"] in _QUANT_DTYPES
        out[module] = (quantized, w["dtype"])
    return out


# --- format routing (main process, no torch) ---


def route(path: str) -> dict | None:
    """A record for a checkpoint this script does not check (GGUF, FTW, no safetensors), else None."""
    files = os.listdir(path)
    if "freetoken_weight.json" in files:
        return {"path": path, "status": "skip", "format": "ftw",
                "hint": "an FTW checkpoint converted by `ft checkpoint`: diagnose its source HF checkpoint; docs/ftw-hotfix.md covers FTW repairs"}
    stub = _stub_record(path)
    gguf = any(f.endswith(".gguf") for f in files) or bool(stub and stub.get("gguf"))
    shards = [f for f in files if f.endswith(".safetensors")]
    if gguf and not shards:
        return {"path": path, "status": "skip", "format": "gguf",
                "hint": "a GGUF checkpoint: check it with scripts/gguf_check/ (README.md there); the GGUF section of this README sorts its failures"}
    code = _missing_config_code(path) if stub else []
    if code:
        return {"path": path, "status": "skip", "format": "remote code",
                "hint": f"the repo ships its own config code ({', '.join(code)}), which a header-only stub does not fetch and FreeToken runs with trust_remote_code; download the repo and diagnose the local directory"}
    if not shards:
        other = sorted({os.path.splitext(f)[1] for f in files if f.endswith((".bin", ".pt", ".pth", ".npz"))} | set((stub or {}).get("other_weights") or ()))
        return {"path": path, "status": "skip", "format": "other",
                "hint": f"no root *.safetensors shard ({', '.join(other) or 'no weights found'}); FreeToken reads safetensors and GGUF only"}
    return None


def _missing_config_code(path: str) -> list[str]:
    """The .py files the config's auto_map names for AutoConfig that are not in ``path``."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return []
    files = []
    for c in (cfg, cfg.get("text_config") or {}):
        ref = (c.get("auto_map") or {}).get("AutoConfig") if isinstance(c, dict) else None
        if isinstance(ref, str) and "--" not in ref:
            files.append(ref.rsplit(".", 1)[0] + ".py")
    return [f for f in dict.fromkeys(files) if not os.path.isfile(os.path.join(path, f))]


def _stub_record(path: str) -> dict | None:
    p = os.path.join(path, ".stub.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


# --- meta-device readers (worker) ---


@functools.cache
def _header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def _meta_tensor(meta: dict):
    import torch
    from freetoken.models.weight import _ST_DTYPE

    return torch.empty(meta["shape"], dtype=_ST_DTYPE[meta["dtype"]], device="meta")


class _MetaSlice:
    def __init__(self, meta: dict):
        self.meta = meta

    def get_shape(self):
        return list(self.meta["shape"])

    def get_dtype(self):
        return self.meta["dtype"]

    def __getitem__(self, idx):
        return _meta_tensor(self.meta)[idx]


class _MetaSafeOpen:
    """safetensors.safe_open whose tensors are meta tensors with the header's shape and dtype."""

    def __init__(self, filename, framework="pt", device="cpu"):
        self._h = _header(os.fspath(filename))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return [k for k in self._h if k != "__metadata__"]

    def offset_keys(self):
        return self.keys()

    def metadata(self):
        return self._h.get("__metadata__")

    def get_tensor(self, name):
        return _meta_tensor(self._h[name])

    def get_slice(self, name):
        return _MetaSlice(self._h[name])


def _meta_iter_expert_tensors_parallel(model_path, is_expert, **_kwargs):
    from freetoken.models.loader import safetensors_weight_map
    from freetoken.utils.hf import download_hf_weight

    folder = download_hf_weight(model_path)
    for name, shard in safetensors_weight_map(folder).items():
        if is_expert(name):
            yield name, _meta_tensor(_header(os.path.join(folder, shard))[name])


@functools.cache
def _meta_wins():
    """A torch function mode that moves CPU operands to meta when an op mixes the two."""
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


@contextlib.contextmanager
def meta_reads(enabled: bool):
    """Point safetensors.safe_open and every binding of iter_expert_tensors_parallel at the meta readers."""
    if not enabled:
        yield
        return
    import safetensors

    import freetoken.models.weight as weight

    real_open, real_parallel = safetensors.safe_open, weight.iter_expert_tensors_parallel
    swapped = []
    safetensors.safe_open = _MetaSafeOpen
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("freetoken.") and getattr(module, "iter_expert_tensors_parallel", None) is real_parallel:
            module.iter_expert_tensors_parallel = _meta_iter_expert_tensors_parallel
            swapped.append(module)
    try:
        with _meta_wins()():
            yield
    finally:
        safetensors.safe_open = real_open
        for module in swapped:
            module.iter_expert_tensors_parallel = real_parallel


def _needs_values(exc: BaseException) -> bool:
    return "meta" in f"{type(exc).__name__}: {exc}".lower()


# --- load stages (worker) ---


def _init_worker() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # the readers' progress bars; errors reach the main process in the record
    sys.stderr = open(os.devnull, "w")
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import torch

    from freetoken.distributed import set_tp_info
    from freetoken.layers import rotary

    set_tp_info(rank=0, size=1)
    rotary.set_rope_device(torch.device("cpu"))


def _engine_config(path: str, text_only: bool):
    import torch

    from freetoken.distributed import try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig

    mm = MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS) if text_only else frozenset())
    return EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy="offload", mm=mm)


def _build_model(config):
    import torch

    from freetoken.engine.engine import _decode_target
    from freetoken.layers.quantization import QuantBackend, set_quant_backend
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    set_quant_backend(QuantBackend.parse(config.quant_backend))
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return create_model(config.model_config)


class _Recorder(dict):
    """The state dict model.load_state_dict consumes: it records the keys a layer asks for and does not find, and the shapes that differ, instead of stopping at the first.

    The layers' own load_state_dict decide what they take (a tied lm_head pops lm_head.weight itself); what is left at the end is unexpected."""

    def __init__(self, expected: dict):
        super().__init__()
        self.expected, self.missing, self.shapes = expected, [], []

    def pop(self, key, *default):
        import torch

        if key not in self:
            if default:
                return default[0]
            self.missing.append(key)
            return torch.empty_like(self.expected[key], device="meta")
        t = super().pop(key)
        ref = self.expected.get(key)
        if ref is None:
            return t
        if tuple(t.shape) != tuple(ref.shape):
            self.shapes.append(f"{key}: checkpoint {tuple(t.shape)}, model {tuple(ref.shape)}")
            return torch.empty_like(ref, device="meta")
        return t.to(dtype=ref.dtype)


_LOSSY = {("float8_e4m3fn", "bfloat16"), ("float8_e5m2", "bfloat16"), ("bfloat16", "float8_e4m3fn"), ("float32", "float8_e4m3fn"), ("bfloat16", "uint8")}


def dense_load(path: str, config, model, *, meta: bool) -> tuple[list[str], list[str], dict]:
    """(problems, warnings, facts) of the engine's dense weight load: what load_weight yields against the model's state dict."""
    import torch

    from freetoken.models import load_weight

    expected = model.state_dict()
    is_moe = bool(getattr(config.model_config, "is_moe", False))
    loaded, dups, casts = _Recorder(expected), [], Counter()
    with meta_reads(meta):
        for key, t in load_weight(path, torch.device("cpu"), include_moe_experts=not is_moe, include_vision=bool(config.active_encoders)):
            ref = expected.get(key)
            # as _materialize_loaded_weight_state_dict: drop an input_scale the layer declares none for, cast to the parameter dtype
            if ref is None and key.endswith(".input_scale"):
                continue
            if key in loaded:
                dups.append(key)
            if ref is not None and (str(t.dtype).removeprefix("torch."), str(ref.dtype).removeprefix("torch.")) in _LOSSY:
                casts[f"{t.dtype} -> {ref.dtype}"] += 1
            # the check keeps no tensor bytes
            dict.__setitem__(loaded, key, t if t.is_meta else torch.empty_like(t, device="meta"))
    try:
        model.load_state_dict(loaded)
    except RuntimeError as e:
        if not str(e).startswith("Unexpected keys"):
            raise
    unexpected, missing, shapes = sorted(loaded), sorted(loaded.missing), loaded.shapes
    problems = []
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected tensors: {group_names(unexpected)}")
    if missing:
        problems.append(f"{len(missing)} model tensors never loaded: {group_names(missing)}")
    if shapes:
        problems.append(f"{len(shapes)} shape mismatches, e.g. {shapes[0]}")
    warns = [f"{n} tensors cast {k} at load (lossy)" for k, n in casts.items()]
    if dups:
        warns.append(f"{len(dups)} tensors are read twice and the engine keeps the last: {group_names(dups)}")
    facts = {"missing_only_scales": bool(missing) and not unexpected and all(m.endswith(_SCALE_ROLES) for m in missing)}
    return problems, warns, facts


def expert_load(path: str, config, model, *, meta: bool) -> list[str]:
    """Problems of filling every MoE layer's expert banks, as build_expert_banks fills them; nothing is kept."""
    import torch

    from freetoken.engine.engine import offload_layer_methods
    from freetoken.moe.expert_banks import method_groups
    from freetoken.models.weight import experts_scattered
    from freetoken.moe.expert_pieces import iter_expert_pieces

    methods = offload_layer_methods(model)
    if methods is None:
        return []
    mc = config.model_config
    num_layers, num_experts = mc.num_moe_layers, mc.num_experts
    written = torch.zeros(num_layers, num_experts, dtype=torch.int32)
    groups = method_groups(methods)
    device = torch.device("meta" if meta else "cpu")
    # the reader load_expert_banks picks: the parallel one (names from the index) for experts stored as many small tensors
    parallel = experts_scattered(path)

    def fill(method, layers, par: bool) -> str | None:
        keep = set(layers)
        layout = {role: spec for role, spec in method.layout().items() if not spec.resident}
        pieces = iter_expert_pieces(path, mc, method.kind, layers=layers if len(groups) > 1 else None, parallel=par)
        for layer, e0, e1, piece in pieces:
            if not (0 <= layer < num_layers and 0 <= e0 < e1 <= num_experts):
                return f"expert piece out of range: layer {layer}, experts {e0}:{e1} of {num_layers} x {num_experts}"
            if layer not in keep:
                continue
            if written[layer, e0:e1].any():
                return f"expert rows read more than once: layer {layer}, experts {e0}:{e1}"
            written[layer, e0:e1] = 1
            out = {role: torch.empty((e1 - e0, *spec.shape), dtype=spec.dtype, device=device) for role, spec in layout.items()}
            method.pack(piece, out)
        return None

    with meta_reads(meta):
        for method, layers in groups:
            try:
                problem = fill(method, layers, parallel)
            except NotImplementedError:
                # load_expert_banks rebuilds serially when a quant has no parallel reader
                if not parallel:
                    raise
                written[layers] = 0
                problem = fill(method, layers, False)
            if problem:
                return [problem]
    missing = (written == 0).nonzero().tolist()
    if missing:
        return [f"expert banks were not filled: {len(missing)} (layer, expert) rows missing (first {missing[:4]})"]
    return []


# --- transformers cross-check (worker) ---


def hf_check(path: str, table: dict) -> dict:
    """Whether transformers' own model takes these tensor names, through its checkpoint conversion rules.

    Quantization companions fold into their module's ``.weight``. Returns the names it does not map, the model keys nothing maps to, and the names a WeightConverter repacks."""
    import torch
    import transformers
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    transformers.utils.logging.disable_progress_bar()
    cfg = transformers.AutoConfig.from_pretrained(path)
    arch = (getattr(cfg, "architectures", None) or [None])[0]
    cls = getattr(transformers, arch or "", None)
    if cls is None:
        return {"skipped": f"transformers has no class {arch}"}
    with torch.device("meta"):
        model = cls._from_config(cfg)
    meta_sd = model.state_dict()
    transforms = get_model_conversion_mapping(model)
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    ignore_unexpected = [re.compile(p) for p in (getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])]
    ignore_missing = [re.compile(p) for p in (getattr(model, "_keys_to_ignore_on_load_missing", None) or [])]
    names = set()
    for name in table:
        module, leaf = split_module(name)
        if leaf in QUANT_COMPANIONS:
            continue
        names.add(module + ".weight" if leaf == "weight_packed" else name)
    mapped, converted = {}, set()
    for name in names:
        new, conv = rename_source_key(name, renamings, converters, model.base_model_prefix, meta_sd)
        mapped[name] = new
        if conv is not None:
            converted.add(name)
    unmapped = sorted(n for n, k in mapped.items() if k not in meta_sd and not any(rx.search(k) for rx in ignore_unexpected))
    tied = bool(getattr(cfg.get_text_config(), "tie_word_embeddings", False))
    targets = set(mapped.values())
    missing = sorted(k for k in meta_sd if k not in targets and not any(rx.search(k) for rx in ignore_missing)
                     and not (tied and k.endswith("lm_head.weight")))
    renamed = sorted(n for n, k in mapped.items() if k != n and n not in converted and k in meta_sd)
    return {"arch": arch, "unmapped": unmapped, "missing": missing, "converted": sorted(converted), "renamed": renamed}


# --- lints (worker) ---


def _skip_entry(dialect: str, modules: list[str], all_modules) -> list[str]:
    """Skip-list entries in the dialect's syntax that cover exactly ``modules`` of ``all_modules``.

    A pattern (layer numbers as a wildcard) stands for its modules only when every module of ``all_modules`` with that pattern is in ``modules``; the others are listed by name."""
    chosen = set(modules)
    by_pattern: dict[str, list[str]] = {}
    for m in all_modules:
        by_pattern.setdefault(pattern_of(m), []).append(m)
    entries = []
    for p in sorted({pattern_of(m) for m in chosen}):
        whole = all(m in chosen for m in by_pattern.get(p, ())) and "N" in p.split(".")
        if whole and dialect == "modelopt":
            entries.append(p.replace(".N", ".*"))
        elif whole and dialect == "compressed-tensors":
            entries.append("re:^" + re.escape(p).replace(r"\.N", r"\.\d+") + "$")
        else:
            entries += sorted(m for m in by_pattern.get(p, [p]) if m in chosen)
    return entries


def _entry_matches(dialect: str, entry: str, name: str) -> bool:
    from freetoken.layers.quantization.names import ct_set, name_set, substr_set

    if dialect == "compressed-tensors":
        return ct_set((entry,), class_names=False)(name)
    if dialect == "fp8":
        return name_set((entry,))(name) or substr_set((entry,))(name)
    return name_set((entry,))(name)


def _raw_skip_entries(path: str) -> list[str]:
    from ckpt_edit import SKIP_KEYS, Checkpoint

    entries = []
    for _, q in Checkpoint(path).quant_configs():
        for key in SKIP_KEYS:
            entries += [e for e in q.get(key) or () if isinstance(e, str)]
    return list(dict.fromkeys(entries))


def lint_quant(path: str, table: dict, spec, hf_config) -> list[dict]:
    """Modules whose stored form contradicts what the quantization_config declares for them, with the skip-list edits that make the two agree."""
    from freetoken.models.register import checkpoint_quant_config

    quant = checkpoint_quant_config(path, hf_config, spec)
    dialect = getattr(quant, "dialect", "none")
    if dialect in ("none", "gguf"):
        return []
    declared_q_stored_f, declared_f_stored_q, exempt_dense = [], [], []
    stored = stored_modules(table)
    for module, (quantized, _) in stored.items():
        # embedding tables (and PLE n-gram tables) are read dense or by their own method whatever the config says
        if any("embed" in seg for seg in module.split(".")) or module.endswith("conv1d"):
            continue
        declared = quant.scheme_for_name(module) is not None
        if quant.unquantized(module):
            if declared and not quantized:
                exempt_dense.append(module)
        elif declared and not quantized:
            declared_q_stored_f.append(module)
        elif quantized and not declared:
            declared_f_stored_q.append(module)
    findings = []
    if exempt_dense:
        findings.append({
            "id": "quant_unlisted_dense_router", "level": "config", "category": "loader",
            "message": f"{len(exempt_dense)} modules FreeToken serves in bf16 anyway are stored bf16 and not listed as skipped: {group_names(exempt_dense)}; listing them works around a reader that wants the stored form to match the config",
            "fix": {"op": "skip_add", "entries": _skip_entry(dialect, exempt_dense, stored), "auto": True},
        })
    targets = _lint_ct_targets(path, table, dialect)
    if declared_q_stored_f:
        findings.append({
            "id": "quant_declared_but_dense", "level": "config", "category": "checkpoint",
            "message": f"{len(declared_q_stored_f)} modules are stored unquantized but the quantization_config does not list them as skipped: {group_names(declared_q_stored_f)}",
            "fix": {"op": "skip_add", "entries": _skip_entry(dialect, declared_q_stored_f, stored), "auto": True},
        })
    # targets that name no module explain the quantized modules nothing covers; recheck after fixing them
    if declared_f_stored_q and not targets:
        entries = _raw_skip_entries(path)
        remove, add = [], []
        for e in entries:
            hits = [m for m in declared_f_stored_q if _entry_matches(dialect, e, m)]
            if not hits:
                continue
            remove.append(e)
            # an entry naming a container also covers its dense children: list those alone instead
            dense = [m for m, (q, _) in stored.items() if not q and _entry_matches(dialect, e, m)]
            add += _skip_entry(dialect, dense, stored) if dense else []
        msg = f"{len(declared_f_stored_q)} modules are stored quantized but the quantization_config skips them or no target covers them: {group_names(declared_f_stored_q)}"
        finding = {"id": "quant_skipped_but_quantized", "level": "config", "category": "checkpoint", "message": msg}
        if remove:
            finding["message"] += f"; skip entries that cover them: {remove[:6]}"
            finding["fixes"] = [{"op": "skip_remove", "entries": remove, "auto": True}] + ([{"op": "skip_add", "entries": add, "auto": True}] if add else [])
        findings.append(finding)
    return findings + targets


def _lint_ct_targets(path: str, table: dict, dialect: str) -> list[dict]:
    if dialect != "compressed-tensors":
        return []
    from ckpt_edit import Checkpoint

    modules = list(stored_modules(table))
    dead: dict[tuple[str, str], list[str]] = {}
    for _, q in Checkpoint(path).quant_configs():
        for group in (q.get("config_groups") or {}).values():
            for target in group.get("targets") or ():
                if target == "Linear" or any(_entry_matches(dialect, target, m) for m in modules):
                    continue
                for pat, repl in TARGET_SWAPS:
                    swapped = re.sub(pat, repl, target)
                    if swapped != target and any(_entry_matches(dialect, swapped, m) for m in modules):
                        dead.setdefault((pat, repl), []).append(target)
                        break
    return [{
        "id": "ct_target_vllm_names", "level": "config", "category": "checkpoint",
        "message": f"{len(ts)} compressed-tensors targets name modules as vLLM does (language_model.model.) and match no tensor, e.g. {ts[0]!r}",
        "fix": {"op": "quant_sub", "pattern": pat, "repl": repl, "auto": True},
    } for (pat, repl), ts in dead.items()]


def rename_rules(findings: list[dict]) -> list[tuple[str, str]]:
    return [tuple(r) for f in findings if f["level"] == "rename" for r in f["fix"]["rules"]]


def renamed_table(table: dict, rules) -> dict:
    """The tensor table as it reads after the rename rules."""
    out = {}
    for name, meta in table.items():
        for pat, repl in rules:
            name = re.sub(pat, repl, name)
        out[name] = meta
    return out


def lint_kv_scales(table: dict, spec) -> list[dict]:
    hits = [n for n in table if re.search(KV_SCALE_RULE[0], n)]
    if not hits or spec.module not in KV_SCALE_DROPPERS:
        return []
    return [{
        "id": "kv_output_scale", "level": "rename", "category": "loader",
        "message": f"{len(hits)} fp8 KV-cache scales stored as <k|v>_proj.output_scale ({group_names(hits, 2)}); vLLM reads them as k_scale / v_scale, FreeToken drops k_scale / v_scale but not output_scale",
        "fix": {"op": "rename", "rules": [list(KV_SCALE_RULE)], "auto": True},
    }]


def lint_prefixes(table: dict, spec) -> list[dict]:
    roots = {ckpt for _, ckpt in spec.checkpoint_roots}
    names = list(table)
    findings = []
    for pat, repl, why, root in PREFIX_RULES:
        if root not in roots:
            continue
        rx = re.compile(pat)
        hits = [n for n in names if rx.search(n)]
        if not hits:
            continue
        renamed = {rx.sub(repl, n) for n in hits}
        if renamed & (set(names) - set(hits)):
            continue
        findings.append({
            "id": "prefix", "level": "rename", "category": "checkpoint",
            "message": f"{why}: {len(hits)} tensors {rx.search(hits[0]).group(0)} -> {repl}",
            "fix": {"op": "rename", "rules": [[pat, repl]], "quant_too": True, "auto": True},
        })
    return findings


def lint_experts(table: dict, num_experts: int | None) -> list[dict]:
    """Per-expert tensors (experts.E.<proj>) where some layer lacks some expert's projection: a checkpoint that lost tensors, not a packing form."""
    if not num_experts:
        return []
    seen: dict[tuple[str, str, str], set[int]] = {}
    for name in table:
        m = _PER_EXPERT.match(name)
        if m:
            seen.setdefault((m["pre"] + m["layer"], m["mid"], m["proj"]), set()).add(int(m["e"]))
    gaps = [(f"{pre}{mid}N.{proj}", sorted(set(range(num_experts)) - es)) for (pre, mid, proj), es in seen.items() if len(es) < num_experts]
    if not gaps:
        return []
    shown = "; ".join(f"{where} lacks experts {missing[:6]}" for where, missing in gaps[:3])
    return [{"id": "expert_tensors_incomplete", "level": "none", "category": "checkpoint",
             "message": f"{len(gaps)} (layer, projection) groups of per-expert tensors are incomplete: {shown}; the checkpoint is missing tensors (download it again, or ask its uploader)"}]


def lint_vision(path: str, table: dict, spec, text_only: bool) -> list[dict]:
    from freetoken.models.config import VISION_KEY_PREFIXES

    if text_only or not spec.encoders:
        return []
    # checkpoint names put the tower under a root (model.visual., model.vision_tower.)
    segments = tuple("." + p for p in VISION_KEY_PREFIXES)
    if not any(("." + n).count(s) for n in table for s in segments):
        return [{"id": "no_vision_tower", "level": "runtime", "category": "checkpoint",
                 "message": "the architecture has a vision tower but the checkpoint stores none: a text-only checkpoint under a multimodal architecture; serve it with --text-model-only (and diagnose it with --text-model-only)"}]
    if spec.mm_processor and not any(os.path.isfile(os.path.join(path, f)) for f in ("preprocessor_config.json", "processor_config.json")):
        return [{"id": "no_processor_config", "level": "file", "category": "checkpoint",
                 "message": "the checkpoint has a vision tower but no preprocessor_config.json, so image requests fail with a 400; copy that file from the base model the checkpoint was tuned from"}]
    return []


def lint_misc(table: dict, hf_config) -> list[dict]:
    from freetoken.layers.quantization.configs import quantization_config_of

    out = []
    dtypes = Counter(m["dtype"] for m in table.values())
    if quantization_config_of(hf_config) is None and (dtypes["F8_E4M3"] or dtypes["F8_E5M2"]):
        out.append({"id": "fp8_without_config", "level": "none", "category": "checkpoint",
                    "message": f"{dtypes['F8_E4M3'] + dtypes['F8_E5M2']} fp8 tensors but no quantization_config says how to read them (scales, block size)"})
    if any(n.endswith((".scales", ".biases")) for n in table) and any(m["dtype"] == "U32" for m in table.values()):
        out.append({"id": "mlx", "level": "none", "category": "quant_method", "message": "MLX-quantized weights (.scales / .biases, U32 packing); FreeToken does not read MLX"})
    return out


def expert_layout(table: dict) -> str | None:
    per_expert = any(re.search(r"\.experts\.\d+\.(gate|up|down|w[123])", n) for n in table)
    stacked = any(re.search(r"\.experts\.(gate_up_proj|down_proj|w13_weight|w2_weight)$", n) for n in table)
    if per_expert and stacked:
        return "mixed: some layers per expert, some stacked"
    return "per expert (experts.N.<proj>)" if per_expert else "stacked (experts.gate_up_proj / down_proj)" if stacked else None


# --- classification ---

_QUANT_METHOD = re.compile(r"quantization method .* is not supported|quant_algo .* is not supported|compressed-tensors .*scheme .* is not supported|MIXED_PRECISION without|fp8 checkpoint with weight_block_size|fp8 block size .* is not supported|target class .* is not supported")
_QUANT_MODEL = re.compile(r"no quant method for|provides no (expert reader|nvfp4_expert_spec|iter_weights_parallel)|needs in/out sizes divisible|is not supported for|cannot use the .* kernel")


def classify(rec: dict) -> str:
    """The category of a failed record (README.md, "Safetensors: six kinds of failure"), first match wins."""
    first = (rec["problems"] or [""])[0]
    findings = rec["findings"]
    hf = rec.get("transformers") or {}
    if rec["stage"] == "files":
        return "checkpoint"
    if rec["stage"] == "config" and "architecture" in first.lower() and "not supported" in first:
        return "arch"
    if any(f["category"] == "quant_method" for f in findings) or _QUANT_METHOD.search(first):
        return "quant_method"
    # a broken prefix misnames every module under it, so whatever fails next is its symptom
    if any(f["id"] == "prefix" and f["category"] == "checkpoint" for f in findings):
        return "checkpoint"
    if _QUANT_MODEL.search(first) or rec.get("missing_only_scales"):
        return "quant_model"
    if hf.get("unmapped") or hf.get("missing"):
        return "checkpoint"
    if any(f["category"] == "checkpoint" for f in findings):
        return "checkpoint"
    if rec["stage"] == "experts" and hf.get("converted"):
        return "packing"
    return "loader"


def _fix_list(findings: list[dict]) -> list[dict]:
    fixes = []
    for f in findings:
        for fix in f.get("fixes") or ([f["fix"]] if "fix" in f else []):
            fixes.append({"level": f["level"], "finding": f["id"], **fix})
    return fixes


def run_lints(path: str, table: dict, spec, hf_config, text_only: bool, warns: list[str], num_experts: int | None = None) -> list[dict]:
    """Every lint; the quantization lints see the names as they read after the rename fixes, which a rename applies first."""
    def safe(fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            warns.append(f"lint {fn.__name__} failed: {type(e).__name__}: {str(e)[:200]}")
            return []

    if spec is None:
        return safe(lint_misc, table, hf_config) if hf_config is not None else []
    findings = safe(lint_prefixes, table, spec) + safe(lint_kv_scales, table, spec)
    after = renamed_table(table, rename_rules(findings))
    findings += safe(lint_experts, after, num_experts) + safe(lint_vision, path, after, spec, text_only) + safe(lint_misc, after, hf_config)
    return findings + safe(lint_quant, path, after, spec, hf_config)


class _Stop(Exception):
    """A stage found problems of its own; the later stages would only repeat them."""


def check(path: str, text_only: bool) -> dict:
    rec = {"path": path, "stage": "config", "status": "fail", "problems": [], "warnings": [], "findings": []}
    stub = _stub_record(path)
    if stub:
        rec["repo"], rec["revision"] = stub["repo"], stub["revision"]
    t0 = time.time()
    hf_config = spec = table = None
    try:
        from freetoken.models.register import get_model_spec
        from freetoken.utils import cached_load_hf_config

        rec["stage"] = "files"
        table = read_tensor_table(path)
        file_problems = check_files(path)
        if file_problems:
            rec["problems"] += file_problems
            raise _Stop
        rec["stage"] = "config"
        hf_config = cached_load_hf_config(path)
        rec["arch"] = (getattr(hf_config, "architectures", None) or [None])[0]
        spec = get_model_spec(rec["arch"])
        rec["stage"] = "quant"
        config = _engine_config(path, text_only)
        mc = config.model_config
        rec["dialect"] = getattr(getattr(mc, "quant", None), "dialect", None)
        rec["moe"] = {"layers": mc.num_moe_layers, "experts": mc.num_experts, "layout": expert_layout(table)} if getattr(mc, "is_moe", False) else None
        rec["encoders"] = [e.kind for e in config.active_encoders]
        rec["stage"] = "build"
        model = _build_model(config)
        for stage, fn in (("dense", dense_load), ("experts", expert_load)):
            rec["stage"] = stage
            try:
                out = fn(path, config, model, meta=True)
            except Exception as e:
                if not _needs_values(e):
                    raise
                rec.setdefault("real_reads", []).append(stage)
                out = fn(path, config, model, meta=False)
            problems, warns, facts = out if stage == "dense" else (out, [], {})
            rec["warnings"] += warns
            rec.update(facts)
            if problems:
                rec["problems"] += problems
                break
        else:
            rec["stage"], rec["status"] = "done", "pass"
    except _Stop:
        pass
    except Exception as e:
        rec["problems"].append(f"{type(e).__name__}: {str(e)[:600]}")
        rec["traceback"] = traceback.format_exc()[-2500:]
    if table is not None:
        rec["findings"] = run_lints(path, table, spec, hf_config, text_only, rec["warnings"], (rec.get("moe") or {}).get("experts"))
    if rec["status"] == "fail" and table is not None and rec["stage"] in ("build", "dense", "experts"):
        try:
            rec["transformers"] = hf_check(path, table)
        except Exception as e:
            rec["transformers"] = {"skipped": f"{type(e).__name__}: {str(e)[:200]}"}
        hf = rec["transformers"]
        if "skipped" not in hf and not hf["unmapped"] and not hf["missing"]:
            # transformers reads these names, so a rename is a workaround for the loader rather than a repair
            for f in rec["findings"]:
                if f["id"] == "prefix":
                    f["category"] = "loader"
    if rec["status"] == "fail":
        rec["category"] = classify(rec)
    # nothing to repair in the files of a checkpoint whose format or architecture is not served
    fixes = [] if rec.get("category") in ("arch", "quant_method") else _fix_list(rec["findings"])
    # a remote repo is only diagnosed: fixes are made on a local download, diagnosed again
    rec["fixes"] = [] if stub else fixes
    rec["fixable_locally"] = bool(stub and fixes)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# --- main process ---


def resolve(target: str, stub_dir: str) -> tuple[str | None, dict | None]:
    """(local path, None), or (None, error record) when a repo cannot be fetched."""
    if os.path.isdir(target):
        return os.path.abspath(target), None
    from fetch_headers import fetch_repo

    try:
        return fetch_repo(target, stub_dir)["stub"], None
    except Exception as e:
        return None, {"path": target, "status": "skip", "format": "unknown", "hint": f"not a local directory, and fetching it from HF failed: {type(e).__name__}: {str(e)[:300]}"}


def _short(rec: dict) -> str:
    return rec.get("repo") or rec["path"]


def print_record(rec: dict, report_path: str | None) -> None:
    print(f"\n== {_short(rec)}" + (" (header-only stub)" if rec.get("repo") else ""))
    if rec["status"] == "skip":
        print(f"   {rec['hint']}")
        return
    bits = [rec.get("arch") or "?", f"quant {rec.get('dialect') or 'none'}"]
    if rec.get("moe"):
        bits.append(f"MoE {rec['moe']['layers']} layers x {rec['moe']['experts']} experts, {rec['moe']['layout'] or 'experts not found by name'}")
    print("   " + ", ".join(bits))
    if rec["status"] == "pass":
        print("   PASS: config, build, dense load" + (", expert banks" if rec.get("moe") else ""))
    else:
        cat = rec["category"]
        print(f"   FAIL at {rec['stage']}: [{cat}] {CATEGORIES[cat]}")
        for p in rec["problems"]:
            print(f"     - {p}")
    hf = rec.get("transformers") or {}
    if hf.get("skipped"):
        print(f"   transformers check skipped: {hf['skipped']}")
    elif hf:
        if hf["unmapped"] or hf["missing"]:
            parts = []
            if hf["unmapped"]:
                parts.append(f"{len(hf['unmapped'])} checkpoint names it cannot map ({group_names(hf['unmapped'], 3)})")
            if hf["missing"]:
                parts.append(f"{len(hf['missing'])} model keys nothing fills ({group_names(hf['missing'], 3)})")
            print(f"   transformers {hf['arch']} would not load it either: " + "; ".join(parts))
        else:
            extra = []
            if hf["renamed"]:
                extra.append(f"renames {len(hf['renamed'])} ({group_names(hf['renamed'], 2)})")
            if hf["converted"]:
                extra.append(f"repacks {len(hf['converted'])} ({group_names(hf['converted'], 2)})")
            print("   transformers loads these names" + (": " + "; ".join(extra) if extra else ""))
    for w in rec["warnings"]:
        print(f"   warning: {w}")
    for f in rec["findings"]:
        print(f"   {'note' if rec['status'] == 'pass' else 'finding'} [{f['level']}, {f['category']}]: {f['message']}")
    if rec.get("fixable_locally") and rec["status"] == "fail":
        print("   fixable: download the checkpoint and run diagnose.py on the local directory to get the fix commands (a remote repo is only diagnosed)")
    fixes = rec.get("fixes") or []
    if not fixes or rec["status"] != "fail":
        return
    target, report_arg = rec["path"], report_path or "<report.jsonl>"
    # one level per round: a rename changes the names the config findings are about, so diagnose again after it
    level = "rename" if any(f["level"] == "rename" and f.get("auto") for f in fixes) else "config"
    if any(f["level"] == level and f.get("auto") for f in fixes):
        if not report_path:
            print("   fixes: rerun with --out <report.jsonl> to get the fix command")
            return
        print(f"   next step ({level} fix; it prints a plan and asks before writing):")
        print(f"     python scripts/checkpoint_diagnose/{FIX_SCRIPTS[level]} {target} --out <new dir> --from-report {report_arg}")
        print("     then diagnose <new dir> again")


def summary(records: list[dict]) -> None:
    status = Counter(r["status"] for r in records)
    print(f"\n{status['pass']} pass, {status['fail']} fail, {status['skip']} not checked")
    cats = Counter(r["category"] for r in records if r["status"] == "fail")
    for cat in CATEGORIES:
        if cats[cat]:
            print(f"  {cats[cat]:4d}  [{cat}] {CATEGORIES[cat]}")
            for r in records:
                if r["status"] == "fail" and r["category"] == cat:
                    fixable = [lv for lv in FIX_SCRIPTS if any(f["level"] == lv and f.get("auto") for f in r.get("fixes", []))]
                    note = f"  (fixes: {', '.join(fixable)})" if fixable else "  (fixable on a local download)" if r.get("fixable_locally") else ""
                    print(f"          {_short(r)}{note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="checkpoint directories or HF repo ids")
    ap.add_argument("--out", metavar="JSONL", help="append the records here; the fix scripts read them with --from-report")
    ap.add_argument("--text-model-only", action="store_true", help="check as `ft serve --text-model-only` loads: no vision tower")
    args = ap.parse_args()

    records = []
    out = open(args.out, "a") if args.out else None
    ctx = mp.get_context("spawn")
    pool = None
    for target in args.targets:
        path, rec = resolve(target, STUB_DIR)
        rec = rec or route(path)
        if rec is None:
            # one worker process, reused; a crash takes only the checkpoint it was checking
            pool = pool or cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_init_worker)
            try:
                rec = pool.submit(check, path, args.text_model_only).result()
            except cf.process.BrokenProcessPool:
                pool = None
                rec = {"path": path, "status": "fail", "stage": "?", "category": "loader", "problems": ["the check process died (out of host memory, or a crash)"],
                       "warnings": [], "findings": [], "fixes": []}
        records.append(rec)
        print_record(rec, args.out)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
    if pool:
        pool.shutdown()
    if out:
        out.close()
    if len(records) > 1:
        summary(records)
    return 0 if all(r["status"] != "fail" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
