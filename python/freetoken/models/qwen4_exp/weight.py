"""Qwen3.8-Flash-Next checkpoint reader (the NVFP4 and the official block-fp8 releases).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseFuser``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named, for the offload cache's expert reader.

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
from freetoken.layers.quantization import get_quant_config
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

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
_KIND_SUFFIXES = (".weight_scale_inv", ".weight")
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn}


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    return rename_vl_prefix(raw_name)


def _split_kind(name: str) -> tuple[str, str]:
    """``name`` -> ``(module, kind)``; kind is "" for tensors that are neither a weight nor a block scale."""
    for suffix in _KIND_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


class _DenseFuser:
    """Concatenates checkpoint projection parts into the model's merged buffers, per kind (weight / block scale).

    The part table is the family's packed_modules_mapping. The QuantConfig picks the GDN in_proj layout and validates each part against the scheme the model built its buffer from.
    """

    def __init__(self, quant, packed: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in packed if fused != "experts"}  # experts: bank reader
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        self.buf: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def scheme(self, module: str):
        return None if self.quant is None else self.quant.scheme_for(module)

    def _target(self, parent: str, leaf: str) -> tuple[str, int] | None:
        candidates = self.by_part.get(leaf)
        if not candidates:
            return None
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
            if not candidates:
                raise ValueError(f"{parent}.{leaf}: no merged projection for the {'split' if split else 'fused'} GDN layout")
        fused, idx = candidates[0]
        if fused in _PAD_TO and not parent.endswith(_HC_WITH_INJECT):
            return None
        return f"{parent}.{fused}", idx

    def check(self, module: str, name: str, tensor: torch.Tensor) -> None:
        """``tensor`` (checkpoint key ``name``) must match the scheme the model built ``module`` from."""
        scheme = self.scheme(module)
        if name.endswith(".weight_scale_inv"):
            if scheme is None or not scheme.has("weight_scale_inv"):
                raise ValueError(f"{name}: {module} has no block scale in the checkpoint's quant config ({scheme})")
            return
        is_fp8 = tensor.dtype in _FP8_DTYPES
        if scheme is None:
            if is_fp8:
                raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
            return
        expected = _ELEM_DTYPES.get(scheme.weight.elem)
        if expected is not None and tensor.dtype is not expected:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}")
        rows, cols = (scheme.weight.group or (1, 1))
        if rows > 1 and tensor.shape[0] % rows or cols > 1 and tensor.shape[1] % cols:
            raise ValueError(f"{name}: {tuple(tensor.shape)} is not a multiple of the {rows}x{cols} scale block of {module}")

    def check_unfused(self, name: str, tensor: torch.Tensor) -> None:
        module, kind = _split_kind(name)
        if kind == ".weight_scale_inv" or (kind == ".weight" and tensor.dtype in _FP8_DTYPES):
            self.check(module, name, tensor)

    def fuse(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Buffer a part; return the merged ``[(name, tensor)]`` once its kind is complete, ``[]`` while incomplete, ``None`` if ``name`` is not a part."""
        module, kind = _split_kind(name)
        if not kind:
            return None
        parent, _, leaf = module.rpartition(".")
        hit = self._target(parent, leaf)
        if hit is None:
            return None
        fused, idx = hit
        self.check(fused, name, tensor)
        slots = self.buf.setdefault((fused, kind), {})
        slots[idx] = tensor
        parts = self.groups[fused.rpartition(".")[2]]
        if len(slots) < len(parts):
            return []
        del self.buf[(fused, kind)]
        rows = [slots[i] for i in range(len(parts))]
        pad_to = _PAD_TO.get(fused.rpartition(".")[2], 0) if kind == ".weight" else 0
        pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
        if pad:
            rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
        return [(fused + kind, torch.cat(rows, dim=0))]


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
    A dense projection is bf16 or 128x128 block-fp8 (``.weight`` e4m3 + ``.weight_scale_inv``) as the checkpoint's QuantConfig says: the official releases skip everything but the routed experts, the community NVFP4-FP8 requants quantize the attention / GDN projections.
    Fusions, per kind: attention q|k|v -> ``qkv_proj``; GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, or ``in_proj_qkvz`` + bf16 ``in_proj_ba`` when qkv|z is quantized; shared-expert gate|up -> ``gate_up_proj``; each per-layer HC's ``input_mix_weight_down`` | ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.
    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the routed experts are NVFP4 and always come from the offload cache's expert reader.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    fuser = _DenseFuser(get_quant_config(), spec.packed_modules_mapping)
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
                fused = fuser.fuse(name, tensor)
                if fused is None:
                    fuser.check_unfused(name, tensor)
                    yield name, tensor
                else:
                    yield from fused

    assert not fuser.buf, f"Incomplete projection fusions: {sorted(k[0] + k[1] for k in fuser.buf)}"


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
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"
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
    """The table as the shard headers describe it, after the aggregate schema check: parts ``0..N-1`` of equal ``[rows_per_part, cols]`` F8_E4M3 blocks, one BF16 scale, one layer; ``files`` is what preflight saw of each shard file."""

    parts: tuple[PleShardPart, ...]
    scale: torch.Tensor  # bf16 scalar
    rows_per_part: int
    cols: int
    layer: int
    files: tuple[PleShardIdentity, ...]

    @property
    def total_rows(self) -> int:
        return len(self.parts) * self.rows_per_part

    @property
    def total_bytes(self) -> int:
        return self.total_rows * self.cols  # one byte per F8_E4M3 element


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

    Refuses, naming the tensor and file: a part that is not F8_E4M3, not a positive 2-D shape, or a different shape from the first part; a byte extent that is not ``rows x cols x 1``; an extent outside its file; a scale that is not exactly one finite positive BF16 scalar; tensors from two layers; duplicate or non-contiguous part indices. No table bytes are read.
    """
    parts: dict[int, PleShardPart] = {}
    scale: tuple[str, bytes] | None = None  # (where, raw bf16)
    layer: tuple[int, str] | None = None  # (layer id, the tensor it was taken from)
    rows = cols = 0
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
            if meta.get("dtype") != _PLE_ST_DTYPE:
                raise ValueError(f"PLE shard {where} has dtype {meta.get('dtype')}, expected {_PLE_ST_DTYPE}")
            shape = meta.get("shape")
            if not (isinstance(shape, list) and len(shape) == 2 and all(type(d) is int and d > 0 for d in shape)):
                raise ValueError(f"PLE shard {where} has shape {shape}, expected a positive 2-D [rows, cols]")
            if first and shape != [rows, cols]:
                raise ValueError(f"PLE shard {where} is {shape}, expected {[rows, cols]} like {first}")
            rows, cols = shape
            first = first or where
            if end - begin != rows * cols:
                raise ValueError(f"PLE shard {where} spans {end - begin} bytes; shape {shape} x 1 byte needs {rows * cols}")
            index = int(part.group("shard"))
            if index in parts:
                seen = parts[index]
                raise ValueError(f"duplicate PLE shard {index}: {seen.name} in {os.path.basename(seen.path)} and {where}")
            parts[index] = PleShardPart(index, key, path, base + begin, end - begin)
    if not parts or sorted(parts) != list(range(len(parts))):
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(parts)[:8]}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")
    value = torch.frombuffer(bytearray(scale[1]), dtype=torch.bfloat16).clone().reshape(())
    if not bool(torch.isfinite(value)) or float(value) <= 0:
        raise ValueError(f"PLE weight_scale {scale[0]} must be finite and positive, got {float(value)}")
    return PleTableLayout(tuple(parts[i] for i in range(len(parts))), value, rows, cols, layer[0], tuple(files))


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
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.

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

    bank = HostBank((layout.total_rows, layout.cols), torch.float8_e4m3fn)
    part_bytes = layout.rows_per_part * layout.cols
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


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


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
