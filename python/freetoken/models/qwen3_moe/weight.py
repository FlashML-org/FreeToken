from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    iter_merged_tensors,
    iter_stacked_experts,
    iter_weight_files,
    shard_tensor,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


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
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = raw_name.removeprefix("language_model.")
                    is_expert = _EXPERT_PATTERN.match(name) is not None
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue

                    raw = f.get_tensor(raw_name)
                    tensor = shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, tensor

    merged = iter_merged_tensors(
        sharded_tensors(),
        _MERGE_RULES,
        model_name="qwen3_moe",
    )
    yield from iter_stacked_experts(
        merged,
        num_experts=config.num_experts,
        model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only iter_weights: raw experts read via the common chunked multi-threaded
    O_DIRECT reader, then same merge+stack pipeline."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def _is_expert(raw_name: str) -> bool:
        return _EXPERT_PATTERN.match(raw_name.removeprefix("language_model.")) is not None

    def raw_experts() -> Iterator[tuple[str, torch.Tensor]]:
        for raw_name, raw in iter_expert_tensors_parallel(
            model_path, _is_expert, workers=workers, chunk=chunk
        ):
            name = raw_name.removeprefix("language_model.")
            tensor = shard_tensor(
                name, raw, rank=tp_info.rank, world_size=tp_info.size,
                num_kv_heads=config.num_kv_heads,
            )
            yield name, tensor

    merged = iter_merged_tensors(raw_experts(), _MERGE_RULES, model_name="qwen3_moe")
    yield from iter_stacked_experts(
        merged, num_experts=config.num_experts, model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


__all__ = ["iter_weights", "iter_weights_parallel"]
