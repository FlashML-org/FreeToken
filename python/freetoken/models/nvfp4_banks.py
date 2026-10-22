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
        if kind not in ("weight", "weight_scale", "weight_scale_2"):
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")
        wanted[name] = (bank_layer, int(match.group("expert")), spec.proj_to_role[proj] + _kind_suffix(kind))
        if kind == "weight" and act is not None:
            act_name = name[: match.start("kind")] + act.name
            if act_name in weight_map:
                wanted[act_name] = (bank_layer, int(match.group("expert")), spec.proj_to_role[proj] + "_input")
    experts = (_num_moe_layers(config) if keep is None else len(keep)) * config.num_experts
    acts = [name for name, (_, _, role) in wanted.items() if role.endswith("_input")]
    # a partial set serves no W4A4 kernel; the pieces then carry none
    if len(acts) != experts * 3:
        for name in acts:
            del wanted[name]
        acts = []
    per_expert = 12 if acts else 9
    if len(wanted) != experts * per_expert:
        raise ValueError(f"{spec.desc}: found {len(wanted)} expert tensors, expected {experts * per_expert}")

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

    return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=per_expert)


__all__ = ["Nvfp4ExpertSourceSpec", "iter_nvfp4_expert_pieces"]
