"""Qwen3.8-Flash-Next checkpoint reader (bf16, block-fp8, ModelOpt and llm-compressor exports).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseReader``.
* :func:`load_ple_table` -- the n-gram table (47.7 GiB fp8 or 95.4 GiB bf16), 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named under the checkpoint's dialect, for the offload cache's expert reader.

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``); ``model.visual.*`` is kept only when the model built the tower.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import stat
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts: per-expert, un-fused. Matched against the RAW weight_map key in nvfp4_banks.
# The ``model.language_model.`` anchor excludes the MTP head's stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2"}

# leaves the model builds as quant-aware Linear layers: their tensors are read under the QuantConfig
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "index_qk_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "input_mix_weight_down", "block_inject_weight", "input_mix_weight_up",
    "key_proj", "value_proj", "lm_head",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.layers\.(?P<layer>\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_RE = re.compile(r"\.layers\.(?P<layer>\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$")
_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# The per-layer HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM pads the merged rows to a multiple of 16 for cuBLAS (hyperconnection.py pad_size).
# The top-level hyper_connection_mixer has no injection and never fuses.
_PAD_TO = {"input_mix_weight_down_block_inject": 16}
_HC_WITH_INJECT = (".attn_hyper_connection", ".mlp_hyper_connection")


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(f"scale has {flat.numel()} elements for a weight with {rows} output rows (shape {tuple(scale.shape)}); expected 1 or {rows}")
    return flat.contiguous()


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in.

    The part table is the family's packed_modules_mapping. The QuantConfig picks the GDN in_proj layout and, through its dialect's storage names, which checkpoint tensor fills which role.
    """

    def __init__(self, quant: QuantConfig | None, packed: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in packed if fused != "experts"}  # experts: bank reader
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores}, scheme)
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        if fused in _PAD_TO and not parent.endswith(_HC_WITH_INJECT):
            return module, 0, 1
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        scheme = self.scheme(module)
        roles = {"weight": "weight"} if scheme is None else {n: r for r, e in self.quant.storage(scheme).items() for n in e.names}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(f"{name}: the checkpoint's quant config declares {module} {scheme or 'unquantized'}, stored as {sorted(roles)}")
        if role == "weight":
            self._check_weight(name, module, scheme, tensor)
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, scheme))
        part = parts.setdefault(idx, {})
        if role in part:
            raise ValueError(f"{name}: {module} already has its {role} from another checkpoint tensor")
        part[role] = tensor
        expected[idx] = set(roles.values())
        if scheme is not None and set(part) == expected[idx]:
            parts[idx] = self._finish(module, scheme, part)
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], scheme)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, _scheme) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}")
        return lines

    @staticmethod
    def _check_weight(name: str, module: str, scheme: QuantScheme | None, tensor: torch.Tensor) -> None:
        if scheme is None:
            if tensor.dtype in _QUANT_DTYPES:
                raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
            return
        elem = _ELEM_DTYPES.get(scheme.weight.elem)
        if elem is None:
            raise ValueError(f"{name}: qwen4_exp has no dense reader for {scheme}")
        if tensor.dtype is not elem:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}")
        rows, cols = tensor.shape[0], tensor.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        if block_rows > 1 and rows % block_rows or block_cols > 1 and cols % block_cols:
            raise ValueError(f"{name}: {rows}x{cols} is not a multiple of the {block_rows}x{block_cols} scale block of {module}")

    def _finish(self, module: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one complete part against ``scheme`` and put its scales in the layer's form."""
        storage = self.quant.storage(scheme)
        part = {role: 1.0 / t.to(torch.float32) if storage[role].reciprocal else t for role, t in part.items()}
        weight = part["weight"]
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            try:
                out[scale_role] = _per_row_scale(part[scale_role], rows)
            except ValueError as exc:
                raise ValueError(f"{module}.{scale_role}: {exc}") from None
        else:
            want = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != want:
                raise ValueError(f"{module}: {scale_role} is {tuple(part[scale_role].shape)}, expected {want} for {scheme}")
            if scheme.weight.scale == "e4m3" and part[scale_role].dtype is not torch.float8_e4m3fn:
                raise ValueError(f"{module}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{module}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out

    @staticmethod
    def _emit(target: str, parts: list[dict[str, torch.Tensor]], scheme: QuantScheme | None) -> list[tuple[str, torch.Tensor]]:
        pad_to = _PAD_TO.get(target.rpartition(".")[2], 0)
        if pad_to and scheme is not None:
            raise ValueError(f"{target}: qwen4_exp pads this fusion to {pad_to} rows and reads it bf16 only, the checkpoint's quant config declares {scheme}")
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so one range covers them: max is exact then and safe if they drift
                out.append((f"{target}.{role}", torch.stack(tensors).max()))
                continue
            pad = (-sum(t.shape[0] for t in tensors)) % pad_to if pad_to else 0
            if pad:
                tensors.append(torch.zeros(pad, *tensors[0].shape[1:], dtype=tensors[0].dtype, device=tensors[0].device))
            out.append((f"{target}.{role}", tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)))
        return out


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the model's state dict minus the routed experts.
    Each Linear is read under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, per-channel / per-tensor fp8, block-fp8, MXFP8 and NVFP4 projections in any mix land as the model's state dict.
    Fusions, per role: attention q|k|v -> ``qkv_proj``; GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, or ``in_proj_qkvz`` + ``in_proj_ba`` when qkv|z is quantized; shared-expert gate|up -> ``gate_up_proj``; each per-layer HC's ``input_mix_weight_down`` | ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.
    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the routed experts are NVFP4 and always come from the offload cache's expert reader.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    reader = _DenseReader(get_quant_config(), spec.packed_modules_mapping)
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = None if name.startswith(VISION_KEY_PREFIXES) else reader.add(name, tensor)
                if emitted is None:
                    yield name, tensor
                else:
                    yield from emitted

    if reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor scale (1 for a bf16 table)."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` view of the bank in the stored dtype."""
        return self.bank.tensor


_PLE_ST_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16}
_PLE_SCALE_ST_DTYPE = "BF16"


@dataclass(frozen=True)
class PleShardPart:
    """One ``shard_<i>`` tensor of the table: where its ``rows_per_part x cols`` bytes sit on disk."""

    index: int
    name: str
    path: str
    file_offset: int
    nbytes: int


@dataclass(frozen=True)
class PleShardIdentity:
    """What preflight saw of one shard file: its inode identity, size, timestamps and the SHA-256 of the header bytes it parsed.

    Best-effort evidence that the file a payload read opens is the file the header came from; it does not authenticate the checkpoint."""

    path: str
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int
    header_base: int
    header_sha256: str

    @classmethod
    def of(cls, path: str, st: os.stat_result, header_base: int, header_sha256: str) -> "PleShardIdentity":
        return cls(path, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, header_base, header_sha256)


@dataclass(frozen=True)
class PleTableLayout:
    """The table as the shard headers describe it, after the aggregate schema check: parts ``0..N-1`` of equal ``[rows_per_part, cols]`` blocks of one dtype (F8_E4M3 or BF16), one BF16 scale (1 for a bf16 table that stores none), one layer; ``files`` is what preflight saw of each shard file."""

    parts: tuple[PleShardPart, ...]
    scale: torch.Tensor  # bf16 scalar
    rows_per_part: int
    cols: int
    layer: int
    files: tuple[PleShardIdentity, ...]
    dtype: torch.dtype = torch.float8_e4m3fn

    @property
    def total_rows(self) -> int:
        return len(self.parts) * self.rows_per_part

    @property
    def total_bytes(self) -> int:
        return self.total_rows * self.cols * self.dtype.itemsize


_PLE_HEADER_MAX_BYTES = 64 << 20  # a real shard header is ~150 KB; a length past this is refused unread


def _canonical_shard_path(path: str) -> str:
    """Resolve a discovered shard path once (the HF cache is symlinks into ``blobs/``); every later open is O_NOFOLLOW on the target."""
    try:
        return os.path.realpath(path, strict=True)
    except OSError as exc:
        raise ValueError(f"cannot resolve PLE shard {path}: {exc.strerror}") from exc


def _open_ple_shard(path: str) -> int:
    """Open a canonical shard path without following a symlink that has since appeared there; only a regular file is accepted."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)  # NONBLOCK: a FIFO must not park us
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"PLE shard {path} is a symlink; refusing to follow it") from exc
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"PLE shard {path} is not a regular file")
    return fd


def _pread_exact(fd: int, nbytes: int, offset: int, path: str) -> bytes:
    chunks = []
    while nbytes:
        chunk = os.pread(fd, min(nbytes, 1 << 20), offset)
        if not chunk:
            raise ValueError(f"PLE shard {path} ended after {offset} bytes; a further {nbytes} were declared")
        chunks.append(chunk)
        nbytes -= len(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _header_bytes(fd: int, path: str, size: int) -> bytes:
    """The raw header of an open shard, bounded: a header length past the budget or past the file is refused with the number, not read."""
    prefix = os.pread(fd, 8, 0)
    if len(prefix) != 8:
        raise ValueError(f"PLE shard {path} is {size} bytes; not a safetensors file")
    n = struct.unpack("<Q", prefix)[0]
    if n > _PLE_HEADER_MAX_BYTES:
        raise ValueError(
            f"PLE shard {path} declares a {n}-byte safetensors header; the budget is {_PLE_HEADER_MAX_BYTES} bytes"
        )
    if 8 + n > size:
        raise ValueError(
            f"PLE shard {path} declares a {n}-byte safetensors header that runs past the end of the {size}-byte file"
        )
    return _pread_exact(fd, n, 8, path)


def _safetensors_header(path: str) -> tuple[dict, PleShardIdentity]:
    """The shard's JSON header plus what was seen of the file, through a nofollow open and a bounded read."""
    fd = _open_ple_shard(path)
    try:
        st = os.fstat(fd)
        raw = _header_bytes(fd, path, st.st_size)
        try:
            header = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
            raise ValueError(f"PLE shard {path} safetensors header is not UTF-8 JSON: {exc}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"PLE shard {path} safetensors header is a JSON {type(header).__name__}, expected an object")
        return header, PleShardIdentity.of(path, st, 8 + len(raw), hashlib.sha256(raw).hexdigest())
    finally:
        os.close(fd)


_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def _check_same_file(identity: PleShardIdentity, st: os.stat_result, when: str) -> None:
    for field in _IDENTITY_FIELDS:
        seen, now = getattr(identity, field), getattr(st, field)
        if seen != now:
            raise ValueError(f"PLE shard {identity.path} changed {when}: {field} {seen} -> {now}")


def revalidate_ple_shard(identity: PleShardIdentity) -> None:
    """Immediately before a payload read: the canonical path must still open nofollow to the same regular file (device, inode, size, mtime, ctime) and its header bytes must still hash to what preflight parsed. Refuses with the field or digest that moved."""
    fd = _open_ple_shard(identity.path)
    try:
        _check_same_file(identity, os.fstat(fd), "after preflight")
        raw = _header_bytes(fd, identity.path, identity.st_size)
        if 8 + len(raw) != identity.header_base:
            raise ValueError(
                f"PLE shard {identity.path} header changed after preflight: length {identity.header_base - 8} -> {len(raw)}"
            )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != identity.header_sha256:
            raise ValueError(
                f"PLE shard {identity.path} header changed after preflight: sha256 {identity.header_sha256} -> {digest}"
            )
    finally:
        os.close(fd)


def _read_bytes(path: str, offset: int, nbytes: int) -> bytes:
    fd = _open_ple_shard(path)
    try:
        return _pread_exact(fd, nbytes, offset, path)
    finally:
        os.close(fd)


def _data_offsets(where: str, meta: dict, payload_bytes: int) -> tuple[int, int]:
    """``meta["data_offsets"]`` as ``(begin, end)``, both inside the file's payload."""
    offsets = meta.get("data_offsets")
    ok = isinstance(offsets, list) and len(offsets) == 2 and all(type(o) is int for o in offsets)
    if not ok or not 0 <= offsets[0] <= offsets[1] <= payload_bytes:
        raise ValueError(f"PLE tensor {where} data_offsets {offsets} run past the {payload_bytes}-byte payload")
    return offsets[0], offsets[1]


def ple_table_layout(folder: str) -> PleTableLayout:
    """Parse every shard header holding a piece of the table and check that they describe one consistent table.

    Refuses, naming the tensor and file: a part that is not F8_E4M3 or BF16, or not the first part's dtype, not a positive 2-D shape, or a different shape from the first part; a byte extent that is not ``rows x cols x itemsize``; an extent outside its file; a scale that is not exactly one finite positive BF16 scalar (an fp8 table must store one; a bf16 table may omit it and is unscaled); tensors from two layers; duplicate or non-contiguous part indices. No table bytes are read.
    """
    parts: dict[int, PleShardPart] = {}
    scale: tuple[str, bytes] | None = None  # (where, raw bf16)
    layer: tuple[int, str] | None = None  # (layer id, the tensor it was taken from)
    rows = cols = 0
    dtype: torch.dtype | None = None  # the first part's dtype, which every other part must share
    first = ""  # the part whose shape every other part must match
    files: list[PleShardIdentity] = []
    for path in _ple_table_files(folder):
        header, identity = _safetensors_header(path)
        files.append(identity)
        base = identity.header_base
        payload_bytes = identity.st_size - base
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            part = _PLE_SHARD_RE.search(key)
            scale_match = None if part is not None else _PLE_SCALE_RE.search(key)
            if part is None and scale_match is None:
                continue
            where = f"{key} in {os.path.basename(path)}"
            key_layer = int((part or scale_match).group("layer"))
            if layer is None:
                layer = (key_layer, where)
            elif key_layer != layer[0]:
                raise ValueError(f"PLE tensor {where} is for layer {key_layer}; {layer[1]} is for layer {layer[0]}")
            begin, end = _data_offsets(where, meta, payload_bytes)
            if scale_match is not None:
                if meta.get("dtype") != _PLE_SCALE_ST_DTYPE or meta.get("shape") not in ([], [1]) or end - begin != 2:
                    raise ValueError(
                        f"PLE weight_scale {where} must be one BF16 scalar, got dtype {meta.get('dtype')}, "
                        f"shape {meta.get('shape')}, {end - begin} bytes"
                    )
                if scale is not None:
                    raise ValueError(f"PLE table has two weight_scale tensors: {scale[0]} and {where}")
                scale = (where, _read_bytes(path, base + begin, 2))
                continue
            shard_dtype = _PLE_ST_DTYPES.get(meta.get("dtype"))
            if shard_dtype is None or dtype not in (None, shard_dtype):
                raise ValueError(f"PLE shard {where} has dtype {meta.get('dtype')}, expected one of {sorted(_PLE_ST_DTYPES)} for every shard")
            dtype = shard_dtype
            shape = meta.get("shape")
            if not (isinstance(shape, list) and len(shape) == 2 and all(type(d) is int and d > 0 for d in shape)):
                raise ValueError(f"PLE shard {where} has shape {shape}, expected a positive 2-D [rows, cols]")
            if first and shape != [rows, cols]:
                raise ValueError(f"PLE shard {where} is {shape}, expected {[rows, cols]} like {first}")
            rows, cols = shape
            first = first or where
            itemsize = dtype.itemsize
            if end - begin != rows * cols * itemsize:
                raise ValueError(
                    f"PLE shard {where} spans {end - begin} bytes; shape {shape} x {itemsize} byte{'s' if itemsize != 1 else ''} "
                    f"needs {rows * cols * itemsize}"
                )
            index = int(part.group("shard"))
            if index in parts:
                seen = parts[index]
                raise ValueError(f"duplicate PLE shard {index}: {seen.name} in {os.path.basename(seen.path)} and {where}")
            parts[index] = PleShardPart(index, key, path, base + begin, end - begin)
    if not parts or sorted(parts) != list(range(len(parts))):
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(parts)[:8]}")
    if scale is None:
        if dtype is torch.float8_e4m3fn:
            raise ValueError("PLE table has no weight_scale")
        value = torch.tensor(1.0, dtype=torch.bfloat16)  # a bf16 table without one is unscaled
    else:
        value = torch.frombuffer(bytearray(scale[1]), dtype=torch.bfloat16).clone().reshape(())
        if not bool(torch.isfinite(value)) or float(value) <= 0:
            raise ValueError(f"PLE weight_scale {scale[0]} must be finite and positive, got {float(value)}")
    return PleTableLayout(tuple(parts[i] for i in range(len(parts))), value, rows, cols, layer[0], tuple(files), dtype)


def expected_ple_rows(qwen4_args, *, ple_index: int = 0) -> int:
    """The row count the checkpoint table must have for this config.

    HF sizes the table at init as the sum of the per-head n-gram vocabulary primes (the same
    :func:`derive_ngram_hash_constants` the dummy-weight path uses), padded up to
    ``make_ngram_vocab_size_divisible_by``; Qwen3.8-Flash-Next: 16 primes after 19,999,999 -> 320,001,536.
    """
    from .ple import derive_ngram_hash_constants

    heads = (int(qwen4_args.ngram_size) - 1) * int(qwen4_args.heads_per_ngram)
    _, sizes, _ = derive_ngram_hash_constants(  # vocab_size only shapes the multipliers, unused here
        vocab_size=1, ngram_size=int(qwen4_args.ngram_size), num_ngram_heads=heads,
        ngram_vocab_size_base=int(qwen4_args.ngram_vocab_size_base), ple_layer_index=ple_index,
    )
    divisible = int(qwen4_args.make_ngram_vocab_size_divisible_by)
    return -(-sum(sizes) // divisible) * divisible


def check_ple_geometry(layout: PleTableLayout, qwen4_args) -> None:
    """The layout must be exactly the table this config addresses: part count, row width, layer, and the padded total row count (equal, not merely covering)."""
    parts = int(qwen4_args.split_ngram_parts)
    if len(layout.parts) != parts:
        raise ValueError(f"PLE table needs shards 0..{parts - 1}, found {len(layout.parts)}")
    if layout.cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {layout.cols} wide, config says {qwen4_args.ngram_head_dim}")
    layer_ids = tuple(getattr(qwen4_args, "ple_layer_ids", None) or ())
    if layer_ids and layout.layer not in layer_ids:
        raise ValueError(f"PLE table tensors are for layer {layout.layer}, config ple_layer_ids is {list(layer_ids)}")
    expected = expected_ple_rows(qwen4_args, ple_index=layer_ids.index(layout.layer) if layer_ids else 0)
    if layout.total_rows != expected:
        raise ValueError(
            f"PLE table has {layout.total_rows} rows ({len(layout.parts)} x {layout.rows_per_part}); "
            f"config geometry requires {expected}"
        )


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one; canonical paths (see :func:`_canonical_shard_path`)."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        files = iter_weight_files(folder)
    else:
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        files = [os.path.join(folder, shard) for shard in {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}]
    return sorted(_canonical_shard_path(path) for path in files)


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the PLE n-gram table tensors, and only those, into ``ple-table-*.safetensors`` next to an FTW checkpoint.

    The table is served from safetensors files in the checkpoint dir (see load_ple_table), not from FTW entries."""
    from safetensors.torch import save_file

    folder = download_hf_weight(model_path)
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush():
        nonlocal batch, size
        if batch:
            name = f"ple-table-{len(written):05d}.safetensors"
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(folder):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_FILE_BYTES:
                    flush()
    flush()
    return written


_PLE_HEADROOM_PCT = 12  # of effective memory kept free of the pinned bank: Python heaps, reader bounces, page-alignment slack


def effective_memory_available(meminfo: str = "/proc/meminfo", cgroup: str = "/sys/fs/cgroup") -> int | None:
    """Host bytes a startup allocation may take: the tighter of ``MemAvailable`` and the cgroup v2 headroom (``memory.max - memory.current``, a known zero when over); ``None`` when neither is readable."""
    bounds = []
    try:
        with open(meminfo, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    bounds.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(cgroup, "memory.max"), encoding="utf-8") as fh:
            limit = fh.read().strip()
        if limit != "max":
            with open(os.path.join(cgroup, "memory.current"), encoding="utf-8") as fh:
                bounds.append(max(0, int(limit) - int(fh.read().strip())))
    except (OSError, ValueError):
        pass
    return min(bounds) if bounds else None


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB (fp8) and must not also sit in the page cache while the bank holds the same bytes.
    An fp8 table needs the checkpoint's scalar ``weight_scale``; a bf16 table without one is unscaled.

    Schema, geometry and the effective-memory admission (12% headroom) all finish before the bank
    exists; any failure after that (a read, a shard that changed, the pin) discards the bank and
    publishes nothing.
    """
    folder = download_hf_weight(model_path)
    layout = ple_table_layout(folder)
    check_ple_geometry(layout, qwen4_args)
    available = effective_memory_available()
    if available is not None and layout.total_bytes > available * (100 - _PLE_HEADROOM_PCT) // 100:
        raise ValueError(
            f"PLE table needs {layout.total_bytes} bytes of host memory for the pinned bank; effective memory "
            f"available is {available} bytes, {available * (100 - _PLE_HEADROOM_PCT) // 100} after the "
            f"{_PLE_HEADROOM_PCT}% headroom"
        )

    bank = HostBank((layout.total_rows, layout.cols), layout.dtype)
    part_bytes = layout.rows_per_part * layout.cols * layout.dtype.itemsize
    identity = {f.path: f for f in layout.files}
    bar = byte_bar(layout.total_bytes, "Loading PLE table")
    buf: memoryview | None = None
    try:
        buf = bank.memoryview()
        verified: set[str] = set()
        for part in layout.parts:
            if part.path not in verified:  # the file is what preflight parsed, right before its first read
                revalidate_ple_shard(identity[part.path])
                verified.add(part.path)
            read_range_into(buf, part.path, file_offset=part.file_offset, nbytes=part.nbytes,
                            dest_offset=part.index * part_bytes, workers=workers, chunk=chunk)
            bar.update(part.nbytes)
        for shard in layout.files:  # ... and still is once its bytes are in the bank
            _check_same_file(shard, os.stat(shard.path), "while reading")
        buf.release()
        buf = None
        bar.close()
        if pin and torch.cuda.is_available():
            bank.pin()
    except BaseException:
        with contextlib.suppress(Exception):  # never mask the failure with its own cleanup
            if buf is not None:
                buf.release()
            bar.close()
            bank.discard()
        raise
    return PleTable(bank=bank, weight_scale=layout.scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.8-Flash-Next NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


__all__ = [
    "nvfp4_expert_spec",
    "PleShardIdentity",
    "PleShardPart",
    "PleTable",
    "PleTableLayout",
    "check_ple_geometry",
    "effective_memory_available",
    "expected_ple_rows",
    "iter_weights",
    "load_ple_table",
    "ple_table_layout",
    "revalidate_ple_shard",
]
