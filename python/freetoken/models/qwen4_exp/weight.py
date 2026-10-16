"""Qwen3.8-Flash-Next checkpoint reader (the NVFP4 and the official block-fp8 releases).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseFuser``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named, for the offload cache's expert reader.

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``); ``model.visual.*`` is kept only when the model built the tower.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import drop_page_cache, iter_weight_files, safetensors_weight_map
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    decode_csf_scales,
)
from freetoken.layers.quantization import QuantKind, get_quant_config
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
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale|weight_scale_2|input_scale|weight_scale\.nvfp4_csf_fixed|weight_scale\.nvfp4_csf_exceptions)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
    # per-expert input scales are runtime quantities, not stored by the banks
    ignored_kinds=frozenset({"input_scale"}),
)
# FP4-Local stores the routed-expert block scales as lossless CSF slab pairs instead of the
# plain E4M3 matrix; decode_csf_scales expands each pair back into the matrix the banks read.
_CSF_KINDS = ("weight_scale.nvfp4_csf_fixed", "weight_scale.nvfp4_csf_exceptions")
_NVFP4_LOCAL_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="Qwen3.8-Flash-Next-FP4-Local CSF NVFP4 experts",
    scale_pair=_CSF_KINDS,
    scale_decoder=decode_csf_scales,
    ignored_kinds=frozenset({"input_scale"}),
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
# Scale tensor names the modelopt dialect stores per scheme role; nvidia's MIXED_PRECISION
# exports name them after the role, while the model's linears keep the role's FreeToken name
# (MXFP8 block scales arrive as weight_scale_inv, NVFP4 globals as weight_global).
_MODEL_SCALE_ROLE = (
    (QuantKind.MXFP8, ".weight_scale", "weight_scale_inv"),
    (QuantKind.NVFP4, ".weight_scale", "weight_scale"),
    (QuantKind.NVFP4, ".weight_scale_2", "weight_global"),
)

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
# FP4-Local tables pack the rows as e2m1 (two per byte), keep a per-16 e4m3 scale row per shard
# and put the global scale in ``weight_scale_2``.
_PLE_SHARD_SCALE_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight_scale$"
)
_PLE_GLOBAL_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale_2"
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
        return _rename_scaled(raw_name)
    return rename_vl_prefix(raw_name)


def _rename_scaled(raw_name: str) -> str | None:
    """Keep a checkpoint scale when the scheme that stores it names it after the model's role.

    nvidia's MIXED_PRECISION exports call the MXFP8 block scale ``weight_scale`` and the NVFP4
    per-tensor global ``weight_scale_2``; the model's linears keep the role names ``weight_scale_inv``
    / ``weight_global``, so rename along the module prefix. Routed experts read their scales
    through the bank readers, so a scheme there keeps nothing.
    """
    quant = get_quant_config()
    if quant is None or _EXPERT_RE.search(raw_name):
        return None
    module, suffix = _split_scale_suffix(raw_name)
    if suffix is None:
        return None
    scheme = quant.scheme_for_name(module)
    if scheme is None:
        return None
    for kind, csuf, role in _MODEL_SCALE_ROLE:
        if scheme.kind is kind and suffix == csuf:
            return rename_vl_prefix(module) + "." + role
    return None


def _split_scale_suffix(raw_name: str) -> tuple[str, str | None]:
    for suffix in _SCALE_SUFFIXES:
        if raw_name.endswith(suffix):
            return raw_name[: -len(suffix)], suffix
    return raw_name, None


class _GlobalExpander:
    """Expand a scalar NVFP4 ``weight_global`` to the model's ``[out]`` fp16 buffer.

    nvidia's W4A16 exports store the per-tensor global as a scalar fp32, while the linear's
    buffer holds one entry per output row (load_state_dict asserts the exact shape). The global
    and its ``.weight`` arrive in either order, so the expansion is flushed once both are seen.
    """

    def __init__(self) -> None:
        self._scalars: dict[str, torch.Tensor] = {}
        self._rows: dict[str, int] = {}

    def on_weight(self, module: str, rows: int) -> tuple[str, torch.Tensor] | None:
        self._rows[module] = rows
        scalar = self._scalars.pop(module, None)
        return None if scalar is None else self._expand(module, scalar, rows)

    def on_global(self, name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
        """A ready pair, or None for a scalar still waiting for its weight."""
        if tensor.numel() != 1:
            return name, tensor
        module = name[: -len(".weight_global")]
        if module in self._rows:
            return self._expand(module, tensor, self._rows[module])
        self._scalars[module] = tensor
        return None

    def check_complete(self) -> None:
        assert not self._scalars, f"scalar weight_global without a weight: {sorted(self._scalars)}"

    @staticmethod
    def _expand(module: str, scalar: torch.Tensor, rows: int) -> tuple[str, torch.Tensor]:
        return module + ".weight_global", scalar.reshape(-1).to(torch.float16).expand(rows).contiguous()


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
    A dense projection is bf16, 128x128 block-fp8, or MXFP8 per-32 (``.weight`` e4m3 + ``.weight_scale``) as the checkpoint's QuantConfig says; the vision MLP is W4A16 NVFP4 in the FP4-Local release.
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
    globals_ = _GlobalExpander()
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
                if name.endswith(".weight_global"):
                    pair = globals_.on_global(name, tensor)
                    if pair is None:
                        continue
                    yield pair
                    continue
                fused = fuser.fuse(name, tensor)
                if fused is None:
                    fuser.check_unfused(name, tensor)
                    yield name, tensor
                else:
                    yield from fused
                if name.endswith(".weight"):
                    pair = globals_.on_weight(name.rpartition(".")[0], tensor.shape[0])
                    if pair is not None:
                        yield pair

    globals_.check_complete()
    assert not fuser.buf, f"Incomplete projection fusions: {sorted(k[0] + k[1] for k in fuser.buf)}"


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    globals_ = _GlobalExpander()
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None or not name.startswith(VISION_KEY_PREFIXES):
                    continue
                tensor = f.get_tensor(raw_name)
                if name.endswith(".weight_global"):
                    pair = globals_.on_global(name, tensor)
                    if pair is None:
                        continue
                    yield pair
                    continue
                yield name, tensor
                if name.endswith(".weight"):
                    pair = globals_.on_weight(name.rpartition(".")[0], tensor.shape[0])
                    if pair is not None:
                        yield pair
    globals_.check_complete()


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor scale.

    FP8 tables hold e4m3 values and a scalar ``weight_scale``; NVFP4 tables hold packed e2m1
    values in ``bank`` and the per-16 e4m3 scale rows in ``scale_bank``, with ``weight_scale``
    reading the checkpoint's global ``weight_scale_2``.
    """

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16 or fp32)
    scale_bank: HostBank | None = None  # [total_rows, ngram_head_dim // 16] e4m3, NVFP4 tables

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim // (2 if packed else 1)]`` view of the value bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


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


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    encoding = getattr(qwen4_args, "ple_embedding_dtype", "fp8")
    if encoding not in ("fp8", "nvfp4"):
        raise ValueError(f"unsupported ple_embedding_dtype {encoding!r}")
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale_parts: dict[int, tuple[str, int, int]] = {}
    scale: torch.Tensor | None = None
    rows = cols = 0
    st_dtype = _PLE_ST_DTYPE if encoding == "fp8" else "U8"
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_GLOBAL_SCALE_SUFFIX if encoding == "nvfp4" else _PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is not None:
                if meta["dtype"] != st_dtype:
                    raise ValueError(
                        f"PLE table shard {key} has dtype {meta['dtype']}, "
                        f"expected {st_dtype} for ple_embedding_dtype={encoding}"
                    )
                shape = meta["shape"]
                if rows and tuple(shape) != (rows, cols):
                    raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
                rows, cols = shape
                begin, end = meta["data_offsets"]
                parts[int(match.group("shard"))] = (path, base + begin, end - begin)
                continue
            if encoding == "nvfp4":
                match = _PLE_SHARD_SCALE_RE.search(key)
                if match is not None:
                    if meta["dtype"] != _PLE_ST_DTYPE:
                        raise ValueError(f"PLE scale shard {key} has dtype {meta['dtype']}, expected {_PLE_ST_DTYPE}")
                    shape = meta["shape"]
                    if rows and tuple(shape) != (rows, cols // 8):
                        raise ValueError(f"PLE scale shard {key} is {shape}, expected {[rows, cols // 8]}")
                    begin, end = meta["data_offsets"]
                    scale_parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    head_dim = qwen4_args.ngram_head_dim
    if encoding == "nvfp4":
        if cols != head_dim // 2:
            raise ValueError(
                f"PLE table row is {cols} packed bytes, config says {head_dim // 2} for a {head_dim}-dim head"
            )
        if sorted(scale_parts) != list(range(expected)):
            raise ValueError(
                f"PLE table needs scale shards 0..{expected - 1}, found {len(scale_parts)}: {sorted(scale_parts)[:8]}"
            )
    elif cols != head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {head_dim}")
    if scale is None:
        raise ValueError(f"PLE table has no {'weight_scale_2' if encoding == 'nvfp4' else 'weight_scale'}")

    bank = HostBank((expected * rows, cols), torch.uint8 if encoding == "nvfp4" else torch.float8_e4m3fn)
    shard_bytes = rows * cols
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    scale_bank = None
    if encoding == "nvfp4":
        scale_cols = cols // 8
        scale_bank = HostBank((expected * rows, scale_cols), torch.float8_e4m3fn)
        scale_shard_bytes = rows * scale_cols
        bar = byte_bar(expected * scale_shard_bytes, "Loading PLE table scales")
        try:
            sbuf = scale_bank.memoryview()
            for shard in range(expected):
                path, offset, nbytes = scale_parts[shard]
                assert nbytes == scale_shard_bytes, f"PLE scale shard {shard} is {nbytes} B, expected {scale_shard_bytes}"
                read_range_into(sbuf, path, file_offset=offset, nbytes=nbytes,
                                dest_offset=shard * scale_shard_bytes, workers=workers, chunk=chunk)
                bar.update(nbytes)
        finally:
            bar.close()
        if pin and torch.cuda.is_available():
            scale_bank.pin()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale, scale_bank=scale_bank)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config):
    """Pick the expert source spec for the checkpoint's scale encoding."""
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    if any(".nvfp4_csf_fixed" in name for name in weight_map):
        return _NVFP4_LOCAL_SOURCE_SPEC
    return _NVFP4_SOURCE_SPEC


__all__ = [
    "nvfp4_expert_spec",
    "PleTable",
    "iter_weights",
    "load_ple_table",
]
