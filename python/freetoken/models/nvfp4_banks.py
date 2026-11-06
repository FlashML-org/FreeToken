from __future__ import annotations

import collections
import os
import re
from dataclasses import dataclass
from typing import Callable, Collection

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
    return tensor.to(torch.float32)


def _activation_scale():
    """The checkpoint tensor of the NVFP4 static activation scale under the installed dialect, or None."""
    from freetoken.layers.quantization import QuantKind, get_quant_config

    try:
        quant = get_quant_config()
    except RuntimeError:
        return None
    if quant is None or QuantKind.NVFP4 not in quant.STORAGE:
        return None
    return quant.stored_tensors(QuantKind.NVFP4).get("input_scale")


def _ingest(spec: "Nvfp4ExpertSourceSpec", act, role: str, tensor: torch.Tensor) -> torch.Tensor:
    if role.endswith("_global"):
        return _ingest_global(spec, tensor)
    if role.endswith("_input"):
        tensor = tensor.float()
        return 1.0 / tensor if act.reciprocal else tensor
    return tensor


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
        # Records are U32 on disk or raw U8 bytes; the bit math runs in int64 because CPU torch
        # has no uint32 shifts and uint32 -> int32 casts wrap records with the sign bit set.
        rec = exceptions.reshape(-1)
        if rec.dtype == torch.uint8:
            rec = rec.view(torch.uint32)
        if rec.dtype not in (torch.uint32, torch.int32):
            raise ValueError(f"CSF exceptions are U32 records (or U8 bytes), got {rec.dtype}")
        rec = rec.to(torch.int64)
        scale = scale.to(torch.int32)
        flat = scale.view(-1)
        flat.scatter_(0, rec & 0xFFFFFF, ((rec >> 24) & 0xFF).to(torch.int32))
    return scale.to(torch.uint8).view(torch.float8_e4m3fn)


def iter_nvfp4_expert_pieces(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    layers: Collection[int] | None = None,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    drop_page_cache: DropPageCache | None = None,
    primary: bool = True,
):
    """One piece per routed expert: ``gate`` / ``up`` / ``down`` codes plus their ``_scale``
    (fp8 block scales) and ``_global`` (the per-tensor scale, reciprocal for quant-side dialects,
    fp32) companions, straight from the safetensors shards. When every expert also stores the
    dialect's static activation scale, the pieces carry it as fp32 ``_input`` (dequant side).

    Serial reads walk the shards in order; ``parallel`` uses the chunked O_DIRECT reader. Either
    way tensors of one expert may span shards, so they are grouped by (layer, expert) as they land.
    ``layers`` (bank layer ids, default all) reads only those layers' experts.
    """
    from freetoken.models.loader import drop_page_cache as _drop
    from freetoken.models.loader import safetensors_weight_map
    from freetoken.moe.expert_pieces import per_expert_pieces

    drop = drop_page_cache or _drop
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)

    keep = None if layers is None else set(layers)
    act = _activation_scale()
    wanted: dict[str, tuple[int, int, str]] = {}
    pair_kinds = spec.scale_pair
    for name in weight_map:
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None or (keep is not None and bank_layer not in keep):
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = _canon_kind(spec, match.group("kind"))
        role = spec.proj_to_role[proj]
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
        if kind == "weight" and act is not None:
            act_name = name[: match.start("kind")] + act.name
            if act_name in weight_map:
                wanted[act_name] = (bank_layer, int(match.group("expert")), role + "_input")
    experts = (_num_moe_layers(config) if keep is None else len(keep)) * config.num_experts
    acts = [name for name, (_, _, slot) in wanted.items() if slot.endswith("_input")]
    # a partial set serves no W4A4 kernel; the pieces then carry none
    if len(acts) != experts * 3:
        for name in acts:
            del wanted[name]
        acts = []
    # A CSF checkpoint keeps every input_scale in its own trailing shard; streaming them
    # with the weights makes the first piece wait on that shard while every other tensor
    # piles up unread in the queue. The CSF path reads them out of band instead.
    act_keys: dict[str, tuple[int, int, str]] = {}
    if pair_kinds is not None:
        for name in acts:
            act_keys[name] = wanted.pop(name)
        acts = list(act_keys)
    per_expert = (3 * (2 + len(pair_kinds)) + (3 if acts else 0)) if pair_kinds is not None else (12 if acts else 9)
    streamed = per_expert - (3 if pair_kinds is not None and acts else 0)
    if len(wanted) != experts * streamed:
        raise ValueError(f"{spec.desc}: found {len(wanted)} expert tensors, expected {experts * streamed}")

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
                    yield name, _ingest(spec, act, wanted[name][2], f.get_tensor(name))
            drop(path)

    def _parallel():
        from freetoken.models.weight import iter_expert_tensors_parallel

        for name, tensor in iter_expert_tensors_parallel(folder, lambda n: n in wanted, workers=workers, chunk=chunk):
            yield name, _ingest(spec, act, wanted[name][2], tensor)

    if pair_kinds is None:
        return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=per_expert)
    if spec.scale_decoder is None:
        raise ValueError(f"{spec.desc}: scale_pair requires a scale_decoder")
    decoder = spec.scale_decoder

    # CSF checkpoints stream the slab parts as separate tensors; group them back into the
    # same logical roles (plus the W4A4 input scales when the set is complete), decoding
    # the scale pair on the fly.
    roles = list(spec.proj_to_role.values())
    slots_per_expert = {
        slot
        for role in roles
        for slot in [f"{role}", f"{role}_global", *(f"{role}~{kind}" for kind in pair_kinds)]
    }
    act_index = {key: name for name, key in act_keys.items()}

    def _csf_pieces():
        acts_in: dict[str, torch.Tensor] = {}
        if act_keys:
            act_shards: dict[str, list[str]] = collections.defaultdict(list)
            for name in act_keys:
                act_shards[weight_map[name]].append(name)
            for shard in tqdm(sorted(act_shards), desc=f"Loading {spec.desc} scales", disable=not primary):
                path = os.path.join(folder, shard)
                _drop(path)
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    for name in act_shards[shard]:
                        acts_in[name] = _ingest(spec, act, act_keys[name][2], f.get_tensor(name))
                _drop(path)

        def _assemble(layer: int, expert: int, parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            grouped: dict[str, torch.Tensor] = {}
            for role in roles:
                grouped[role] = parts[role].unsqueeze(0)
                grouped[f"{role}_global"] = parts[f"{role}_global"].reshape(1, -1)
                if acts:
                    grouped[f"{role}_input"] = acts_in[act_index[(layer, expert, f"{role}_input")]].reshape(1, -1)
                grouped[f"{role}_scale"] = decoder(
                    *[parts[f"{role}~{kind}"] for kind in pair_kinds]
                ).unsqueeze(0)
            return grouped

        yield from per_expert_pieces(
            _parallel() if parallel else _serial(),
            wanted.get,
            tensors_per_expert=len(slots_per_expert),
            assemble=_assemble,
        )

    return _csf_pieces()


__all__ = ["Nvfp4ExpertSourceSpec", "decode_csf_scales", "iter_nvfp4_expert_pieces"]
