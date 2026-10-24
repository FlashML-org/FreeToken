"""Qwen3.8-Flash-Next checkpoint reader (bf16, block-fp8, ModelOpt and llm-compressor exports).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseReader``.
* :func:`load_ple_table` -- the n-gram table (47.7 GiB fp8 or 95.4 GiB bf16), 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named under the checkpoint's dialect, for the offload cache's expert reader.

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

# FP4-Local stores the routed-expert block scales as lossless CSF slab pairs instead of the
# plain E4M3 matrix; decode_csf_scales expands each pair back into the matrix the banks read.
_CSF_KINDS = ("weight_scale.nvfp4_csf_fixed", "weight_scale.nvfp4_csf_exceptions")

# The vision tower bypasses the scheme-driven reader (its leaves are not in _LINEAR_LEAVES),
# so its role-named ModelOpt scales are renamed here instead; text dense roles go through
# QuantConfig storage aliases in _DenseReader.
_VISION_PREFIXES = ("model.visual.", "visual.")
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
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
    if raw_name.startswith(_VISION_PREFIXES) and raw_name.endswith(_SCALE_SUFFIXES):
        return _rename_scaled(raw_name)
    return rename_vl_prefix(raw_name)


def _rename_scaled(raw_name: str) -> str | None:
    """Keep a vision scale when the scheme that stores it names it after the model's role.

    The vision tower bypasses _DenseReader, so nvidia's MIXED_PRECISION role names are mapped
    here: the MXFP8 block scale ``weight_scale`` -> ``weight_scale_inv``, the NVFP4 per-tensor
    global ``weight_scale_2`` -> ``weight_global``. Routed experts read their scales through the
    bank readers, so a scheme there keeps nothing.
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
    """Expand a scalar vision NVFP4 ``weight_global`` to the model's ``[out]`` fp32 buffer.

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
            rows = self._rows.get(name[: -len(".weight_global")])
            if rows is not None and tensor.numel() != rows:
                raise ValueError(
                    f"{name} holds {tensor.numel()} elements for a weight with {rows} output rows"
                )
            return name, tensor
        module = name[: -len(".weight_global")]
        if module in self._rows:
            return self._expand(module, tensor, self._rows[module])
        self._scalars[module] = tensor
        return None

    def check_complete(self) -> None:
        if self._scalars:
            raise ValueError(f"scalar weight_global without a weight: {sorted(self._scalars)}")

    @staticmethod
    def _expand(module: str, scalar: torch.Tensor, rows: int) -> tuple[str, torch.Tensor]:
        return module + ".weight_global", _per_row_scale(scalar, rows)


def _emit_vision(globals_: _GlobalExpander, name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one vision tensor, flushing a scalar ``weight_global`` once its weight is seen."""
    if name.endswith(".weight_global"):
        pair = globals_.on_global(name, tensor)
        if pair is not None:
            yield pair
        return
    yield name, tensor
    if name.endswith(".weight"):
        pair = globals_.on_weight(name.rpartition(".")[0], tensor.shape[0])
        if pair is not None:
            yield pair


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
    globals_ = _GlobalExpander()  # the vision tower's scalar weight_globals bypass the reader
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
                if name.startswith(VISION_KEY_PREFIXES):
                    yield from _emit_vision(globals_, name, tensor)
                    continue
                emitted = reader.add(name, tensor)
                if emitted is None:
                    yield name, tensor
                else:
                    yield from emitted

    globals_.check_complete()
    if reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


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
                yield from _emit_vision(globals_, name, tensor)
    globals_.check_complete()


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor scale.

    FP8 tables hold e4m3 values and a scalar ``weight_scale``; bf16 tables hold bf16 values and
    a scale of 1; NVFP4 tables hold packed e2m1 values in ``bank`` and the per-16 e4m3 scale
    rows in ``scale_bank``, with ``weight_scale`` reading the checkpoint's global ``weight_scale_2``.
    """

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16 or fp32)
    scale_bank: HostBank | None = None  # [total_rows, ngram_head_dim // 16] e4m3, NVFP4 tables

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim // (2 if packed else 1)]`` view of the value bank in the stored dtype."""
        return self.bank.tensor


_PLE_ST_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16}


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
    table is ~47.7 GiB (fp8) and must not also sit in the page cache while the bank holds the same bytes.
    An fp8 table needs the checkpoint's scalar ``weight_scale``; a bf16 table without one is unscaled.
    """
    encoding = getattr(qwen4_args, "ple_embedding_dtype", "fp8")
    if encoding not in ("fp8", "bf16", "nvfp4"):
        raise ValueError(f"unsupported ple_embedding_dtype {encoding!r}")
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale_parts: dict[int, tuple[str, int, int]] = {}
    shapes: dict[int, tuple[int, ...]] = {}
    scale_shapes: dict[int, tuple[int, ...]] = {}
    scale: torch.Tensor | None = None
    dtype: torch.dtype | None = None
    # fp8 stays dtype-driven (F8_E4M3 or a bf16 table); nvfp4 packs e2m1, bf16 demands bf16
    _want_st = {"nvfp4": ("U8",), "bf16": ("BF16",), "fp8": ("F8_E4M3", "BF16")}[encoding]
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
                if meta["dtype"] not in _want_st:
                    raise ValueError(
                        f"PLE table shard {key} has dtype {meta['dtype']}, "
                        f"expected {' or '.join(_want_st)} for ple_embedding_dtype={encoding}"
                    )
                shard_dtype = {**_PLE_ST_DTYPES, "U8": torch.uint8}[meta["dtype"]]
                if dtype not in (None, shard_dtype):
                    raise ValueError(f"PLE table shard {key} mixes dtypes: {meta['dtype']} after {dtype}")
                dtype = shard_dtype
                index = int(match.group("shard"))
                begin, end = meta["data_offsets"]
                parts[index] = (path, base + begin, end - begin)
                shapes[index] = tuple(meta["shape"])
                continue
            if encoding == "nvfp4":
                match = _PLE_SHARD_SCALE_RE.search(key)
                if match is not None:
                    if meta["dtype"] != "F8_E4M3":
                        raise ValueError(f"PLE scale shard {key} has dtype {meta['dtype']}, expected F8_E4M3")
                    index = int(match.group("shard"))
                    begin, end = meta["data_offsets"]
                    scale_parts[index] = (path, base + begin, end - begin)
                    scale_shapes[index] = tuple(meta["shape"])

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if len(set(shapes.values())) > 1:
        raise ValueError(f"PLE table shards disagree on shape: {sorted(set(shapes.values()))[:8]}")
    rows, cols = shapes[min(shapes)]
    head_dim = qwen4_args.ngram_head_dim
    if encoding == "nvfp4":
        if head_dim % 16:
            raise ValueError(f"nvfp4 PLE heads must be a multiple of 16, got {head_dim}")
        if cols != head_dim // 2:
            raise ValueError(
                f"PLE table row is {cols} packed bytes, config says {head_dim // 2} for a {head_dim}-dim head"
            )
        if sorted(scale_parts) != list(range(expected)):
            raise ValueError(
                f"PLE table needs scale shards 0..{expected - 1}, found {len(scale_parts)}: {sorted(scale_parts)[:8]}"
            )
        want = (rows, cols // 8)
        for shard in sorted(scale_shapes):
            if scale_shapes[shard] != want:
                raise ValueError(f"PLE scale shard {shard} is {list(scale_shapes[shard])}, expected {list(want)}")
    elif cols != head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {head_dim}")
    if scale is None:
        if encoding == "nvfp4":
            raise ValueError("PLE table has no weight_scale_2")
        if dtype is torch.float8_e4m3fn:
            raise ValueError("PLE table has no weight_scale")
        scale = torch.tensor(1.0, dtype=torch.bfloat16)

    bank = HostBank((expected * rows, cols), dtype)
    shard_bytes = rows * cols * dtype.itemsize
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            if nbytes != shard_bytes:
                raise ValueError(f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}")
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
                if nbytes != scale_shard_bytes:
                    raise ValueError(f"PLE scale shard {shard} is {nbytes} B, expected {scale_shard_bytes}")
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


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor).

    FP4-Local exports the block scales as CSF slab pairs; scan the weight map and decode them
    through the shared decoder so the banks still see plain e4m3 matrices.
    """
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    weight_map = safetensors_weight_map(download_hf_weight(model_path))
    csf = any(name.endswith(".nvfp4_csf_fixed") for name in weight_map)
    kinds = list(kind_map)
    if csf:
        kinds += list(_CSF_KINDS)
        kind_map = {**kind_map, **{kind: kind for kind in _CSF_KINDS}}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kinds)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.8-Flash-Next NVFP4 experts ({quant.dialect}{' CSF' if csf else ''})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
        scale_pair=_CSF_KINDS if csf else None,
        scale_decoder=decode_csf_scales if csf else None,
    )


__all__ = [
    "nvfp4_expert_spec",
    "PleTable",
    "iter_weights",
    "load_ple_table",
]
