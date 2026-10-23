from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch

from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule, drop_page_cache, iter_merged_tensors, iter_stacked_experts,
    iter_weight_files, safetensors_weight_map, shard_tensor,
)
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.utils import cached_load_hf_config, download_hf_weight

from .config import parse_config

_EXPERT = re.compile(r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_MERGE = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}
_NVFP4 = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: (
        layer - config.first_k_dense_replace if config.first_k_dense_replace <= layer < config.num_layers else None
    ),
    desc="Flint NVFP4 experts",
)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    return _NVFP4


def iter_weights(
    model_path: str, device: torch.device, *, include_moe_experts: bool, include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    model_path = download_hf_weight(model_path)
    config = parse_config(cached_load_hf_config(model_path))
    if include_moe_experts and config.expert_quant != "none":
        raise ValueError("Flint NVFP4 uses the offload, cpu or hybrid MoE strategy")
    tp = get_tp_info()
    names = safetensors_weight_map(model_path)

    def tensors():
        for file in iter_weight_files(model_path):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as reader:
                for name in reader.keys():
                    expert = _EXPERT.match(name) is not None
                    if (expert and not include_moe_experts) or (not expert and not include_non_moe):
                        continue
                    raw = reader.get_tensor(name)
                    if name.endswith(".softmax_offset"):
                        raw = raw.chunk(tp.size)[tp.rank].float()
                    elif name.endswith(".expert_bias"):
                        raw = raw.float()
                    else:
                        raw = shard_tensor(name, raw, rank=tp.rank, world_size=tp.size, num_kv_heads=config.num_kv_heads)
                    yield name, raw
            drop_page_cache(file)
        if include_non_moe:
            for layer in range(config.first_k_dense_replace, config.num_layers):
                name = f"model.layers.{layer}.mlp.gate.expert_bias"
                if name not in names:
                    yield name, torch.zeros(config.num_experts, dtype=torch.float32, device=device)

    merged = iter_merged_tensors(tensors(), _MERGE, model_name="flint")
    yield from iter_stacked_experts(merged, num_experts=config.num_experts, model_name="flint", expert_pattern=_EXPERT)
