from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.layers.quantization import QuantKind
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    iter_merged_tensors,
    iter_root_safetensor_files_from_index,
    shard_tensor,
)
from freetoken.utils import cached_load_hf_config

from .config import parse_config

_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
}


def local_mxfp4_intermediate_range(
    intermediate_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int, int]:
    if intermediate_size % 32 != 0:
        raise ValueError("GPT-OSS MXFP4 intermediate size must be divisible by 32")
    blocks = intermediate_size // 32
    blocks_per_rank = (blocks + world_size - 1) // world_size
    local_intermediate = blocks_per_rank * 32
    start = rank * local_intermediate
    end = min((rank + 1) * local_intermediate, intermediate_size)
    return start, end, local_intermediate


def _shard_dim(value: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * value.ndim
    slices[dim] = slice(start, end)
    return value[tuple(slices)].clone()


def _shard_dim_pad(
    value: torch.Tensor,
    dim: int,
    start: int,
    end: int,
    target_size: int,
) -> torch.Tensor:
    shard = _shard_dim(value, dim, start, end)
    pad_size = target_size - shard.shape[dim]
    if pad_size <= 0:
        return shard
    pad_shape = list(shard.shape)
    pad_shape[dim] = pad_size
    return torch.cat([shard, shard.new_zeros(pad_shape)], dim=dim)


def _attn_range(
    *,
    kind: str,
    rank: int,
    world_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    if kind == "q":
        if num_q_heads % world_size != 0:
            raise ValueError("GPT-OSS query heads must be divisible by TP size")
        heads_per_rank = num_q_heads // world_size
        head_idx = rank * heads_per_rank
    else:
        if world_size > num_kv_heads:
            if world_size % num_kv_heads != 0:
                raise ValueError("GPT-OSS TP size must divide or replicate KV heads")
            heads_per_rank = 1
            head_idx = rank * num_kv_heads // world_size
        else:
            if num_kv_heads % world_size != 0:
                raise ValueError("GPT-OSS KV heads must be divisible by TP size")
            heads_per_rank = num_kv_heads // world_size
            head_idx = rank * heads_per_rank
    return head_idx * head_dim, (head_idx + heads_per_rank) * head_dim


def shard_gpt_oss_tensor(
    name: str,
    value: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> torch.Tensor:
    if name.endswith(".self_attn.sinks"):
        start, end = _attn_range(
            kind="q",
            rank=rank,
            world_size=world_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=1,
        )
        return value[start:end].clone()

    for suffix, kind in (
        (".self_attn.q_proj.weight", "q"),
        (".self_attn.q_proj.bias", "q"),
        (".self_attn.k_proj.weight", "kv"),
        (".self_attn.k_proj.bias", "kv"),
        (".self_attn.v_proj.weight", "kv"),
        (".self_attn.v_proj.bias", "kv"),
    ):
        if name.endswith(suffix):
            start, end = _attn_range(
                kind=kind,
                rank=rank,
                world_size=world_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            return _shard_dim(value, 0, start, end)

    if name.endswith(".self_attn.o_proj.weight"):
        start, end = _attn_range(
            kind="q",
            rank=rank,
            world_size=world_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        return _shard_dim(value, 1, start, end)
    if name.endswith(".self_attn.o_proj.bias"):
        return value.clone() if rank == 0 else torch.zeros_like(value)

    if ".mlp.experts." in name:
        tp_start, tp_end, local_intermediate = local_mxfp4_intermediate_range(
            intermediate_size,
            rank=rank,
            world_size=world_size,
        )
        if name.endswith(".mlp.experts.gate_up_proj_blocks"):
            return _shard_dim_pad(
                value,
                1,
                2 * tp_start,
                2 * tp_end,
                2 * local_intermediate,
            )
        if name.endswith(".mlp.experts.gate_up_proj_scales"):
            return _shard_dim_pad(
                value,
                1,
                2 * tp_start,
                2 * tp_end,
                2 * local_intermediate,
            )
        if name.endswith(".mlp.experts.gate_up_proj_bias"):
            return _shard_dim_pad(
                value,
                1,
                2 * tp_start,
                2 * tp_end,
                2 * local_intermediate,
            )
        if name.endswith(".mlp.experts.down_proj_blocks"):
            return _shard_dim_pad(
                value,
                2,
                tp_start // 32,
                tp_end // 32,
                local_intermediate // 32,
            )
        if name.endswith(".mlp.experts.down_proj_scales"):
            return _shard_dim_pad(
                value,
                value.ndim - 1,
                tp_start // 32,
                tp_end // 32,
                local_intermediate // 32,
            )
        if name.endswith(".mlp.experts.down_proj_bias"):
            return value.clone() if rank == 0 else torch.zeros_like(value)

    if "embed_tokens" in name or "lm_head" in name:
        return shard_tensor(
            name,
            value,
            rank=rank,
            world_size=world_size,
            num_kv_heads=num_kv_heads,
        )
    return value.clone()


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in iter_root_safetensor_files_from_index(model_path):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    is_expert = ".mlp.experts." in raw_name
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue
                    raw = f.get_tensor(raw_name)
                    yield raw_name, shard_gpt_oss_tensor(
                        raw_name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_q_heads=config.num_qo_heads,
                        num_kv_heads=config.num_kv_heads,
                        head_dim=config.head_dim,
                        intermediate_size=config.moe_intermediate_size,
                    )

    yield from iter_merged_tensors(
        sharded_tensors(),
        _MERGE_RULES,
        model_name="gpt_oss",
    )


def _expert_layer_and_name(key: str) -> tuple[int, str] | None:
    parts = key.split(".")
    if len(parts) < 6 or parts[0] != "model" or parts[1] != "layers":
        return None
    if parts[4] != "experts":
        return None
    try:
        layer_id = int(parts[2])
    except ValueError:
        return None
    return layer_id, parts[5]


def _read_safetensor_slice(
    f,
    name: str,
    slices: tuple[slice, ...],
) -> torch.Tensor:
    if all(s == slice(None) for s in slices):
        return f.get_tensor(name)
    return f.get_slice(name)[slices]


_EXPERT_SOURCES = (
    "gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
    "down_proj_blocks", "down_proj_scales", "down_proj_bias",
)


def _source_slices(config, tp_info) -> dict[str, tuple[str, tuple[slice, ...]]]:
    """HF expert tensor -> (piece role, this rank's slice of the intermediate axis)."""
    tp_start, tp_end, _ = local_mxfp4_intermediate_range(
        config.moe_intermediate_size, rank=tp_info.rank, world_size=tp_info.size
    )
    tp_end = max(tp_start, tp_end)
    rows = slice(2 * tp_start, 2 * tp_end)
    cols = slice(tp_start // 32, tp_end // 32)
    return {
        "gate_up_proj_blocks": ("gate_up", (slice(None), rows, slice(None), slice(None))),
        "gate_up_proj_scales": ("gate_up_scale", (slice(None), rows, slice(None))),
        "gate_up_proj_bias": ("gate_up_bias", (slice(None), rows)),
        "down_proj_blocks": ("down", (slice(None), slice(None), cols, slice(None))),
        "down_proj_scales": ("down_scale", (slice(None), slice(None), cols)),
        "down_proj_bias": ("down_bias", (slice(None), slice(None))),
    }


def iter_expert_pieces(model_path: str, config, kind: QuantKind, *, parallel: bool = False, workers: int = 8, chunk: int = 8 << 20):
    """gpt-oss experts as HF ships them, one piece per layer: stacked ``[E, ...]`` mxfp4
    ``_blocks`` / ``_scales`` / ``_bias`` tensors sliced to this rank's intermediate range."""
    if kind is not QuantKind.MXFP4:
        return None
    if config.moe_weight_format != "mxfp4":
        raise ValueError("GPT-OSS offload requires MXFP4 expert weights")
    tp_info = get_tp_info()
    slices = _source_slices(config, tp_info)
    num_layers, num_experts = config.num_layers, config.num_experts

    def _is_expert(name: str) -> bool:
        info = _expert_layer_and_name(name)
        return info is not None and 0 <= info[0] < num_layers

    def _tensors():
        if parallel:
            from freetoken.models.weight import iter_expert_tensors_parallel

            for name, whole in iter_expert_tensors_parallel(model_path, _is_expert, workers=workers, chunk=chunk):
                _, source = _expert_layer_and_name(name)
                yield name, whole[slices[source][1]]
            return
        for file in iter_root_safetensor_files_from_index(model_path):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    info = _expert_layer_and_name(name)
                    if info is None:
                        continue
                    layer_id, source = info
                    if not 0 <= layer_id < num_layers:
                        raise ValueError(f"Unexpected GPT-OSS expert layer in checkpoint: {name}")
                    if source not in slices:
                        raise ValueError(f"Unexpected GPT-OSS expert source: {name}")
                    yield name, _read_safetensor_slice(f, name, slices[source][1])

    def _pieces():
        pending: dict[int, dict[str, torch.Tensor]] = {}
        for name, tensor in _tensors():
            layer_id, source = _expert_layer_and_name(name)
            role = slices[source][0]
            if role == "down_bias" and tp_info.rank != 0:
                tensor = torch.zeros_like(tensor)  # the TP reduction adds the bias once, on rank 0
            piece = pending.setdefault(layer_id, {})
            piece[role] = tensor
            if len(piece) == len(_EXPERT_SOURCES):
                del pending[layer_id]
                yield layer_id, 0, num_experts, piece
        if pending:
            raise ValueError(f"Missing GPT-OSS expert tensors for layers {sorted(pending)[:8]}")

    return _pieces()


__all__ = [
    "iter_weights",
    "iter_expert_pieces",
    "local_mxfp4_intermediate_range",
    "shard_gpt_oss_tensor",
]
