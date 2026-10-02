from __future__ import annotations

import collections
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.distributed import get_tp_info
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


def _kind_suffix(kind: str) -> str:
    return {"weight": "", "weight_scale": "_scale", "weight_scale_2": "_global"}[kind]


# gate/up and their block scales carry I on the row axis, down and its scales on the column axis; the
# per-expert global scale has no I axis and is shared by every rank.
_I_AXIS = {"gate": 0, "up": 0, "gate_scale": 0, "up_scale": 0, "down": 1, "down_scale": 1}


def shard_nvfp4_piece(role: str, t: torch.Tensor, *, rank: int, tp_size: int) -> torch.Tensor:
    """This rank's block of the intermediate dim in one NVFP4 expert tensor, in its own storage."""
    axis = _I_AXIS.get(role)
    if tp_size == 1 or axis is None:
        return t
    total = t.shape[axis]
    if total % tp_size:
        raise ValueError(f"NVFP4 expert piece {role!r} has {total} entries on axis {axis}, not divisible by TP={tp_size}")
    step = total // tp_size
    return t.narrow(axis, rank * step, step).clone(memory_format=torch.contiguous_format)


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

    tp = get_tp_info()

    def _local(name: str, tensor: torch.Tensor) -> torch.Tensor:
        role = wanted[name][2]
        if role.endswith("_global"):
            return _ingest_global(spec, tensor)
        return shard_nvfp4_piece(role, tensor, rank=tp.rank, tp_size=tp.size)

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
                    piece = _local(name, f.get_tensor(name))
                    yield name, piece
                    del piece
            drop(path)

    def _parallel():
        from freetoken.models.weight import iter_expert_tensors_parallel

        for name, tensor in iter_expert_tensors_parallel(folder, lambda n: n in wanted, workers=workers, chunk=chunk):
            yield name, _local(name, tensor)

    return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=9)


__all__ = ["Nvfp4ExpertSourceSpec", "iter_nvfp4_expert_pieces"]
