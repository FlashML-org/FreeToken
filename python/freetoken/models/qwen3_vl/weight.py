from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import MergeRule, iter_merged_tensors, iter_weight_files, shard_tensor
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}
_FUSED_EXPERT_KEYS = ("gate_up_proj", "down_proj")


def rename_vl_prefix(raw_name: str) -> str:
    """Checkpoint wrapper prefixes -> state-dict keys: tower under visual., text under model."""
    if raw_name.startswith(("model.visual.", "visual.")):
        return "visual." + raw_name.removeprefix("model.").removeprefix("visual.")
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _is_fused_experts(name: str) -> bool:
    parts = name.split(".")
    return len(parts) > 2 and parts[-2] == "experts" and parts[-1] in _FUSED_EXPERT_KEYS


def _linear_layout(tensor: torch.Tensor) -> torch.Tensor:
    # the checkpoint stacks experts as [E, in, out]; the banks and fused kernels take [E, out, in] and copy from the strided view
    return tensor.transpose(1, 2)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = rename_vl_prefix(raw_name)
                    if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                        continue
                    is_expert = _is_fused_experts(name)
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue
                    raw = f.get_tensor(raw_name)
                    if is_expert:
                        if tp_info.size > 1:
                            raise NotImplementedError("Qwen3-VL-MoE experts are not tensor-parallel sharded")
                        yield name, _linear_layout(raw)
                        continue
                    if name.startswith("visual.") and tp_info.size > 1:
                        raise NotImplementedError("Qwen VL vision tower weights are not tensor-parallel sharded")
                    tensor = shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, tensor

    yield from iter_merged_tensors(tensors(), _MERGE_RULES, model_name="qwen3_vl")


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only iter_weights over the common chunked multi-threaded O_DIRECT reader."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_vl parallel reader is experts-only (used by load_moe_expert_sources)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen3-VL-MoE experts are not tensor-parallel sharded")
    for raw_name, raw in iter_expert_tensors_parallel(
        model_path, _is_fused_experts, workers=workers, chunk=chunk
    ):
        yield rename_vl_prefix(raw_name), _linear_layout(raw)


__all__ = ["iter_weights", "iter_weights_parallel", "rename_vl_prefix"]
