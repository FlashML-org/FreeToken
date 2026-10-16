from __future__ import annotations

import collections
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.utils import download_hf_weight
from tqdm import tqdm

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]


@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str
    # Maps checkpoint tensor-kind names onto the canonical (modelopt) kinds, e.g.
    # compressed-tensors' weight_packed -> weight, weight_global_scale -> weight_scale_2.
    kind_map: dict[str, str] | None = None
    # The checkpoint stores the QUANT-side global scale (local fp8 scales were
    # multiplied by it before the cast); the banks keep its reciprocal.
    global_reciprocal: bool = False
    # Tensor kinds present in the checkpoint but never fetched (per-expert input
    # scales are runtime quantities, not stored here).
    ignored_kinds: frozenset[str] = frozenset()
    # The checkpoint stores the block scales as a lossless CSF slab pair instead of
    # the plain E4M3 matrix (e.g. "weight_scale.nvfp4_csf_fixed" /
    # "weight_scale.nvfp4_csf_exceptions"); scale_decoder expands the parts into the
    # plain matrix the banks consume.
    scale_pair: tuple[str, ...] | None = None
    scale_decoder: Callable[..., torch.Tensor] | None = None


def _canon_kind(spec: "Nvfp4ExpertSourceSpec", kind: str) -> str:
    return spec.kind_map.get(kind, kind) if spec.kind_map else kind


def _ingest_global(spec: "Nvfp4ExpertSourceSpec", tensor: torch.Tensor) -> torch.Tensor:
    if spec.global_reciprocal:
        tensor = 1.0 / tensor.float()
    return tensor.to(torch.float16)


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Nvfp4ExpertSourceSpec, layer: int, config) -> int | None:
    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    num_layers = _num_moe_layers(config)
    if bank_layer < 0 or bank_layer >= num_layers:
        raise ValueError(
            f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} "
            f"is outside [0, {num_layers})"
        )
    return bank_layer


def _kind_suffix(kind: str) -> str:
    return {"weight": "", "weight_scale": "_scale", "weight_scale_2": "_global"}[kind]


def decode_csf_scales(fixed: torch.Tensor, exceptions: torch.Tensor) -> torch.Tensor:
    """Expand the CSF slab encoding of an NVFP4 per-16 scale into the plain E4M3 matrix (CPU).

    ``fixed`` is U8 ``[slabs, 16 * (1 + row_bytes)]``: the first 16 bytes of each slab hold the
    per-row scale bases (one byte per row of the slab's 16 rows), the rest are row-major code
    bytes with column c at bit 4c. ``exceptions`` is U32 records, ``pos = rec & 0xFFFFFF``
    (row-major scale index, below 2**24) and ``value = rec >> 24`` (the exact E4M3 byte),
    applied after base + offset.
    """
    if fixed.ndim != 2 or fixed.dtype != torch.uint8:
        raise ValueError(f"CSF fixed slab tensor is {tuple(fixed.shape)} {fixed.dtype}, expected U8 [slabs, slab_bytes]")
    slabs, slab_bytes = fixed.shape
    row_bytes = slab_bytes // 16 - 1
    if row_bytes < 1 or slab_bytes != 16 * (1 + row_bytes):
        raise ValueError(f"CSF slab width {slab_bytes} does not hold 16 bases plus 16 * row_bytes code bytes")
    rows = slabs * 16
    cols = row_bytes * 2
    data = fixed.to(torch.int16)
    base = data[:, :16].reshape(rows, 1)
    codes = data[:, 16:].reshape(rows, row_bytes)
    bits = 4 * torch.arange(cols)
    scale = (base + ((codes[:, bits // 8] >> bits % 8) & 0x0F)) & 0xFF
    if exceptions.numel():
        # CPU torch has no uint32 shifts; the same bit math in int32 is well defined,
        # including (rec >> 24) & 0xFF for records with the sign bit set.
        rec = exceptions.reshape(-1)
        if rec.dtype == torch.uint8:
            rec = rec.view(torch.int32)
        elif rec.dtype != torch.int32:
            rec = rec.to(torch.int32)
        scale = scale.to(torch.int32)
        flat = scale.reshape(-1)
        flat.scatter_(0, (rec & 0xFFFFFF).to(torch.int64), (rec >> 24) & 0xFF)
    return scale.to(torch.uint8).view(torch.float8_e4m3fn)


def iter_nvfp4_expert_pieces(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    drop_page_cache: DropPageCache | None = None,
    primary: bool = True,
):
    """One piece per routed expert: ``gate`` / ``up`` / ``down`` codes plus their ``_scale``
    (fp8 block scales) and ``_global`` (the per-tensor scale, reciprocal for quant-side dialects,
    fp16) companions, straight from the safetensors shards.

    Serial reads walk the shards in order; ``parallel`` uses the chunked O_DIRECT reader. Either
    way tensors of one expert may span shards, so they are grouped by (layer, expert) as they land.
    """
    from freetoken.models.loader import drop_page_cache as _drop
    from freetoken.models.loader import safetensors_weight_map
    from freetoken.moe.expert_pieces import per_expert_pieces

    drop = drop_page_cache or _drop
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)

    wanted: dict[str, tuple[int, int, str]] = {}
    pair_kinds = spec.scale_pair
    for name in weight_map:
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = _canon_kind(spec, match.group("kind"))
        role = spec.proj_to_role[proj]
        if kind in spec.ignored_kinds:
            continue
        if pair_kinds is not None:
            if kind in pair_kinds:
                wanted[name] = (bank_layer, int(match.group("expert")), f"{role}~{kind}")
            elif kind in ("weight", "weight_scale_2"):
                wanted[name] = (bank_layer, int(match.group("expert")), role + _kind_suffix(kind))
            else:
                raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")
        elif kind not in ("weight", "weight_scale", "weight_scale_2"):
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")
        else:
            wanted[name] = (bank_layer, int(match.group("expert")), role + _kind_suffix(kind))
    per_expert = 9 if pair_kinds is None else 3 * (2 + len(pair_kinds))
    expected = _num_moe_layers(config) * config.num_experts * per_expert
    if len(wanted) != expected:
        raise ValueError(f"{spec.desc}: found {len(wanted)} expert tensors, expected {expected}")

    def _serial():
        by_shard: dict[str, list[str]] = collections.defaultdict(list)
        for name, shard in weight_map.items():
            if name in wanted:
                by_shard[shard].append(name)
        for shard in tqdm(sorted(by_shard), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            drop(path)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name in by_shard[shard]:
                    tensor = f.get_tensor(name)
                    if wanted[name][2].endswith("_global"):
                        tensor = _ingest_global(spec, tensor)
                    yield name, tensor
            drop(path)

    def _parallel():
        from freetoken.models.weight import iter_expert_tensors_parallel

        for name, tensor in iter_expert_tensors_parallel(folder, lambda n: n in wanted, workers=workers, chunk=chunk):
            if wanted[name][2].endswith("_global"):
                tensor = _ingest_global(spec, tensor)
            yield name, tensor

    if pair_kinds is None:
        return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=9)
    if spec.scale_decoder is None:
        raise ValueError(f"{spec.desc}: scale_pair requires a scale_decoder")
    decoder = spec.scale_decoder

    # CSF checkpoints stream the slab parts as separate tensors; group them back into the
    # same 9 logical roles, decoding the scale pair on the fly.
    roles = list(spec.proj_to_role.values())
    slots_per_expert = {
        slot
        for role in roles
        for slot in [f"{role}", f"{role}_global", *(f"{role}~{kind}" for kind in pair_kinds)]
    }

    def _csf_pieces():
        pending: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for name, tensor in _parallel() if parallel else _serial():
            layer, expert, slot = wanted[name]
            piece = pending.setdefault((layer, expert), {})
            piece[slot] = tensor
            if len(piece) != len(slots_per_expert):
                continue
            del pending[(layer, expert)]
            grouped: dict[str, torch.Tensor] = {}
            for role in roles:
                grouped[role] = piece[role].unsqueeze(0)
                grouped[f"{role}_global"] = piece[f"{role}_global"].reshape(1, -1)
                grouped[f"{role}_scale"] = decoder(
                    *[piece[f"{role}~{kind}"] for kind in pair_kinds]
                ).unsqueeze(0)
            yield layer, expert, expert + 1, grouped
        if pending:
            raise ValueError(f"{spec.desc}: incomplete expert tensors for (layer, expert) {sorted(pending)[:8]}")

    return _csf_pieces()


__all__ = ["Nvfp4ExpertSourceSpec", "decode_csf_scales", "iter_nvfp4_expert_pieces"]
