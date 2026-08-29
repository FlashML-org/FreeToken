"""Qwen3.8-Flash-Next (RadixArk NVFP4) checkpoint reader.

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, into the offload cache's source banks.

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``) and ``model.visual.*`` (served text-only).
"""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
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
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"

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


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
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


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: the modelopt
    ``ignore`` list covers everything except those experts, so attention, GDN, HC, PLE, the shared
    expert and lm_head are all plain bf16 (the n-gram hash constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from :func:`load_nvfp4_expert_sources`.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
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
                tensor = f.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
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


@dataclass(frozen=True)
class PleShard:
    """A PLE byte range in a safetensor file."""

    path: str
    offset: int
    nbytes: int
    rows: int
    cols: int


@dataclass(frozen=True)
class PleLayout:
    """Validated on-disk PLE layout."""

    shards: tuple[PleShard, ...]
    weight_scale: torch.Tensor
    rows_per_shard: int
    head_dim: int


class MmapPleStorage:
    """Memory-map PLE safetensor ranges and gather selected rows."""

    def __init__(self, layout: PleLayout) -> None:
        self.rows_per_shard = layout.rows_per_shard
        self.head_dim = layout.head_dim
        self.num_rows = len(layout.shards) * self.rows_per_shard
        self.nbytes = sum(shard.nbytes for shard in layout.shards)
        self._files: dict[str, BinaryIO] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._shards: list[torch.Tensor] = []
        try:
            for shard in layout.shards:
                mapping = self._maps.get(shard.path)
                if mapping is None:
                    fh = open(shard.path, "rb")
                    mapping = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_COPY)
                    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                        mapping.madvise(mmap.MADV_RANDOM)
                    self._files[shard.path] = fh
                    self._maps[shard.path] = mapping
                tensor = torch.frombuffer(
                    mapping,
                    dtype=torch.uint8,
                    count=shard.nbytes,
                    offset=shard.offset,
                ).reshape(shard.rows, shard.cols)
                self._shards.append(tensor)
        except Exception:
            self.close()
            raise

    def gather(self, row_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Copy valid rows into a CPU buffer; invalid IDs produce zeros."""
        ids = row_ids.reshape(-1)
        if ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("mmap PLE row ids must be a CPU int64 tensor")
        expected = (ids.numel(), self.head_dim)
        if out.device.type != "cpu" or out.dtype != torch.uint8 or tuple(out.shape) != expected:
            raise ValueError(
                f"mmap PLE output must be CPU uint8 {expected}, got {out.device} "
                f"{out.dtype} {tuple(out.shape)}"
            )
        out.zero_()
        if ids.numel() == 0:
            return out

        valid = (ids >= 0) & (ids < self.num_rows)
        positions = valid.nonzero().reshape(-1)
        if positions.numel() == 0:
            return out
        valid_ids = ids.index_select(0, positions)
        shard_ids = torch.div(valid_ids, self.rows_per_shard, rounding_mode="floor")
        for shard_id in torch.unique(shard_ids).tolist():
            in_shard = (shard_ids == shard_id).nonzero().reshape(-1)
            dst_positions = positions.index_select(0, in_shard)
            local_ids = valid_ids.index_select(0, in_shard).remainder(self.rows_per_shard)
            rows = self._shards[shard_id].index_select(0, local_ids)
            out.index_copy_(0, dst_positions, rows)
        return out

    def close(self) -> None:
        self._shards.clear()
        for mapping in self._maps.values():
            mapping.close()
        self._maps.clear()
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class MmapPleTable:
    """Mapped PLE storage and its FP8 scale."""

    storage: MmapPleStorage
    weight_scale: torch.Tensor


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


def _ple_layout(model_path: str, qwen4_args) -> PleLayout:
    """Parse and validate the PLE shards."""
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

    shards = tuple(
        PleShard(path=parts[i][0], offset=parts[i][1], nbytes=parts[i][2], rows=rows, cols=cols)
        for i in range(expected)
    )
    shard_bytes = rows * cols
    for shard_id, shard in enumerate(shards):
        if shard.nbytes != shard_bytes:
            raise ValueError(
                f"PLE shard {shard_id} is {shard.nbytes} B, expected {shard_bytes}"
            )
    return PleLayout(
        shards=shards,
        weight_scale=scale,
        rows_per_shard=rows,
        head_dim=cols,
    )


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    layout = _ple_layout(model_path, qwen4_args)

    bank = HostBank((len(layout.shards) * layout.rows_per_shard, layout.head_dim),
                    torch.float8_e4m3fn)
    bar = byte_bar(sum(shard.nbytes for shard in layout.shards), "Loading PLE table")
    try:
        buf = bank.memoryview()
        dest_offset = 0
        for shard in layout.shards:
            read_range_into(buf, shard.path, file_offset=shard.offset, nbytes=shard.nbytes,
                            dest_offset=dest_offset, workers=workers, chunk=chunk)
            dest_offset += shard.nbytes
            bar.update(shard.nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=layout.weight_scale)


def load_mmap_ple_table(model_path: str, qwen4_args) -> MmapPleTable:
    """Map the PLE safetensor ranges."""
    layout = _ple_layout(model_path, qwen4_args)
    return MmapPleTable(
        storage=MmapPleStorage(layout),
        weight_scale=layout.weight_scale,
    )


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
) -> dict:
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "MmapPleStorage",
    "MmapPleTable",
    "PleTable",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_mmap_ple_table",
    "load_ple_table",
]
