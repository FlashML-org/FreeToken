"""Qwen3.8-Flash-Next checkpoint reader (the NVFP4 and the official block-fp8 releases).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named, for the offload cache's expert reader.

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``) and ``model.visual.*`` (served text-only).
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Callable, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import download_hf_weight
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
# ``.weight_scale_inv`` is NOT here: it is the 128x128 block-FP8 scale, and whether it is
# dropped or emitted depends on how the module it belongs to is served (see ``_rename``).
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
_SCALE_INV = ".weight_scale_inv"

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
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

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


def _rename(raw_name: str, is_block_fp8: Callable[[str], bool] | None = None) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip.

    ``is_block_fp8`` answers, for one checkpoint module name, whether the model serves it as
    block-FP8. A ``weight_scale_inv`` is kept only for those modules -- their linears declare
    the matching buffer -- and dropped for every other module, whose weight this reader
    dequantizes to bf16 instead (see :func:`_load_maybe_block_fp8`). Emitting it either way
    would trip ``load_state_dict``'s strict unexpected-key check.
    """
    if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_INV):
        if is_block_fp8 is None or not is_block_fp8(raw_name[: -len(_SCALE_INV)]):
            return None
    elif raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]],
    table: dict[str, tuple[tuple[str, ...], int]] | None = None,
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in (table or _FUSIONS).items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def _load_maybe_block_fp8(f, raw_name: str, keyset: set[str]) -> torch.Tensor:
    """Load ``raw_name``, dequantizing 128x128 block-FP8 to bf16 when a sibling
    ``weight_scale_inv`` sits in the same shard; pass everything else through unchanged.

    The official releases keep the dense attn/GDN/HC/PLE projections bf16 (they sit on the
    quant ``ignore`` list), but a community requant can store them as block-FP8 without saying
    so per module -- and an undeclared module gets no scheme, so the model builds a plain bf16
    linear for it. Dequantizing here is what keeps the two agreeing. Without it those weights
    reach ``_try_fuse`` as fp8 and die on the fp8-with-bf16 ``torch.cat``.
    """
    tensor = f.get_tensor(raw_name)
    if raw_name.endswith(".weight") and raw_name[: -len(".weight")] + _SCALE_INV in keyset:
        from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

        scale = f.get_tensor(raw_name[: -len(".weight")] + _SCALE_INV)
        return dequant_block_fp8(tensor, scale).to(torch.bfloat16)
    return tensor


# Serving the dense side natively as block-FP8 changes which buffers the model expects. The
# four-way in_proj fusion cannot survive it: b|a are num_v_heads rows wide, and
# Fp8BlockLinearMethod requires every output size to be a whole number of 128-row blocks, so
# gdn.py splits the projection into an fp8 qkv|z GEMM plus a small bf16 b|a GEMM -- the split
# sglang and vLLM use. Each fp8 group fuses its ``weight_scale_inv`` on the same axis as its
# ``weight``; every fp8 part is a whole number of 128-row blocks (10240|6144 for qkv|z,
# 12288|512|512 for q|k|v), so the per-block scales concatenate exactly alongside the rows
# they describe.
_SPLIT_FP8: dict[str, tuple[str, tuple[str, ...]]] = {
    "in_proj": (
        ".linear_attn.in_proj_qkvz",
        (".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z"),
    ),
    "qkv_proj": (
        ".self_attn.qkv_proj",
        (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ),
}
_SPLIT_BF16: dict[str, tuple[str, tuple[str, ...]]] = {
    "in_proj": (
        ".linear_attn.in_proj_ba",
        (".linear_attn.in_proj_b", ".linear_attn.in_proj_a"),
    ),
}
# The bf16 fusion each group replaces when it is served natively.
_SPLIT_REPLACES = {
    "in_proj": ".linear_attn.in_proj.weight",
    "qkv_proj": ".self_attn.qkv_proj.weight",
}
# (layer type carrying the group, the attribute path the model builds for it). The probe asks
# the QuantConfig the same question gdn.py and attention.py ask when they build the linear.
_SPLIT_PROBE = {
    "in_proj": ("linear_attention", "linear_attn.in_proj_qkvz"),
    "qkv_proj": ("full_attention", "self_attn.qkv_proj"),
}


def _fusions_for(groups: frozenset[str]) -> dict[str, tuple[tuple[str, ...], int]]:
    """``_FUSIONS`` with each block-FP8 group replaced by its split, weights and scales."""
    table = {k: v for k, v in _FUSIONS.items() if k not in {_SPLIT_REPLACES[g] for g in groups}}
    for group in groups:
        fused, parts = _SPLIT_FP8[group]
        for kind in (".weight", _SCALE_INV):
            table[fused + kind] = (tuple(part + kind for part in parts), 0)
        if group in _SPLIT_BF16:
            fused_bf16, parts_bf16 = _SPLIT_BF16[group]
            table[fused_bf16 + ".weight"] = (tuple(part + ".weight" for part in parts_bf16), 0)
    return table


def _declares_quant(model_path: str) -> bool:
    """Whether a local checkpoint directory carries any quantization declaration at all."""
    if os.path.exists(os.path.join(model_path, "hf_quant_config.json")):
        return True  # ModelOpt < 0.41 keeps it only in the sidecar
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return False
    text = config.get("text_config") or {}
    return bool(config.get("quantization_config") or text.get("quantization_config"))


def _block_fp8_dense(model_path: str) -> tuple[frozenset[str], Callable[[str], bool]]:
    """Which fused attention groups the checkpoint declares as block-FP8, plus a predicate
    over checkpoint module names for the un-fused ones (o_proj, out_proj, shared expert).

    Both read the family's own :class:`QuantConfig`, built from the same ModelSpec name map
    the engine hands the model, so the buffers this reader emits cannot disagree with the
    modules the model built. That is not just tidiness: the block-FP8 linears have no
    tensor-parallel variant, so a rank that downgrades must downgrade on both sides at once.
    """
    from freetoken.engine.config import checkpoint_quant_config
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.qwen4_exp.config import _layer_types
    from freetoken.models.register import get_model_spec
    from freetoken.utils import cached_load_hf_config

    # A local checkpoint that declares no quantization has nothing to serve natively, and
    # answering that from the file avoids the full AutoConfig resolution behind
    # cached_load_hf_config -- which the reader's own fixtures cannot satisfy, since they
    # write safetensors shards and no config.json. Non-local paths take the full route.
    if os.path.isdir(model_path) and not _declares_quant(model_path):
        return frozenset(), lambda _name: False

    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    quant = checkpoint_quant_config(model_path, hf_config, spec)
    if quant is None:
        return frozenset(), lambda _name: False

    def is_block(scheme) -> bool:
        return scheme is not None and scheme.kind is QuantKind.FP8_BLOCK

    layer_types = _layer_types(getattr(hf_config, "text_config", hf_config))
    groups = set()
    for group, (layer_type, leaf) in _SPLIT_PROBE.items():
        layer_id = next((i for i, t in enumerate(layer_types) if t == layer_type), None)
        if layer_id is not None and is_block(quant.scheme_for(f"model.layers.{layer_id}.{leaf}")):
            groups.add(group)
    return frozenset(groups), lambda name: is_block(quant.scheme_for_name(name))


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Most releases quantize only the routed experts --
    every skip list (modelopt ``ignore``, fp8 ``modules_to_not_convert``) covers the rest -- so
    attention, GDN, HC, PLE, the shared expert and lm_head arrive as plain bf16 (the n-gram hash
    constants stay int64). A modelopt ``MIXED_PRECISION`` build can instead declare the dense
    attention and GDN projections ``FP8_PB_WO``; the model then builds block-FP8 linears for them,
    so their weights and ``weight_scale_inv`` pass through un-dequantized and the in_proj fusion
    splits to match (see :func:`_block_fp8_dense`). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from the offload cache's expert reader.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    # A declared block-FP8 dense side is served natively; anything else keeps the dequant path.
    groups, is_block_fp8 = _block_fp8_dense(model_path)
    fusions = _fusions_for(groups) if groups else _FUSIONS
    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                name = _rename(raw_name, is_block_fp8)
                if name is None:
                    continue
                tensor = (
                    f.get_tensor(raw_name)
                    if is_block_fp8(raw_name.rpartition(".")[0])
                    else _load_maybe_block_fp8(f, raw_name, keyset)
                )
                fused = _try_fuse(name, tensor, fuse_buf, fusions)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


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
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
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
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = [
    "nvfp4_expert_spec",
    "PleTable",
    "iter_weights",
    "load_ple_table",
]
