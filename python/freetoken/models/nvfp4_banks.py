from __future__ import annotations

import collections
import json
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


def _tp_slice() -> tuple[int, int] | None:
    """``(i_local, i_lo)``: this rank's slice of the expert intermediate axis, or None at TP=1.

    TP shards every routed expert along I (gate/up output rows, down input columns), so each
    rank caches only its own half and the MoE layer all-reduces the partial sum. The expert
    kernel sizes its banks from ``MoEConfig.local_intermediate``, which is the same split.
    """
    from freetoken.distributed import try_get_tp_info

    tp = try_get_tp_info()
    if tp is None or tp.size == 1:
        return None
    return tp.size, tp.rank


def _tp_shard(role: str, tensor: torch.Tensor, inter: int, tp_size: int, tp_rank: int):
    """This rank's I-slice of one checkpoint expert tensor. ``_global`` is a per-tensor scalar
    and has no I axis; gate/up carry I on the row axis, down on the column axis (halved for the
    packed FP4 codes, sixteenthed for the fp8 block scales)."""
    if role.endswith("_global"):
        return tensor
    n = inter // tp_size
    assert n % 16 == 0, f"NVFP4 TP shard {n} must cover whole 16-wide scale blocks"
    lo = tp_rank * n
    if role.startswith("down"):
        d = 2 if role == "down" else 16
        return tensor[:, lo // d : (lo + n) // d]
    return tensor[lo : lo + n]


def _kind_suffix(kind: str) -> str:
    return {"weight": "", "weight_scale": "_scale", "weight_scale_2": "_global"}[kind]


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
    from freetoken.moe.expert_pieces import per_expert_pieces

    drop = drop_page_cache or _drop
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    wanted: dict[str, tuple[int, int, str]] = {}
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
        if kind not in ("weight", "weight_scale", "weight_scale_2"):
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")
        wanted[name] = (bank_layer, int(match.group("expert")), spec.proj_to_role[proj] + _kind_suffix(kind))
    expected = _num_moe_layers(config) * config.num_experts * 9
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

    stream = _parallel() if parallel else _serial()
    shard = _tp_slice()
    if shard is not None:
        tp_size, tp_rank = shard
        inter = config.moe_intermediate_size
        base = stream

        def _sharded():
            for name, tensor in base:
                yield name, _tp_shard(wanted[name][2], tensor, inter, tp_size, tp_rank)

        stream = _sharded()
    return per_expert_pieces(stream, wanted.get, tensors_per_expert=9)


__all__ = ["Nvfp4ExpertSourceSpec", "iter_nvfp4_expert_pieces"]
