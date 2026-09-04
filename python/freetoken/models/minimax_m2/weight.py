from __future__ import annotations

import re
from typing import Iterator, Tuple

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    drop_page_cache,
    iter_merged_tensors,
    iter_weight_files,
    shard_tensor,
)
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_RE = re.compile(r"\.block_sparse_moe\.experts\.\d+\.")
_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"w1": "gate", "w3": "up", "w2": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="NVFP4 experts",
)


_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
}


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """yield dense BF16 weights only; NVFP4 experts go to the offload cache via load_nvfp4_expert_sources."""
    assert not include_moe_experts, (
        "MiniMax-M2 stores experts as NVFP4 and only supports the offload MoE backend; "
        "experts are loaded into the offload cache, not the dense model."
    )
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def raw() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading dense weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for name in f.keys():
                    if _EXPERT_RE.search(name):
                        continue
                    # skip attention calibration scales (FP8 kv_cache_scheme): unused with
                    # BF16 KV/attention, and ``.k_proj``/``.v_proj`` in their names make the
                    # qkv merge see incomplete groups.
                    if name.endswith((".k_scale", ".v_scale")) or (
                        ".self_attn." in name and name.endswith("_scale")
                    ):
                        continue
                    if not include_non_moe:
                        continue
                    tensor = shard_tensor(
                        name,
                        f.get_tensor(name),
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    yield name, tensor

    yield from iter_merged_tensors(raw(), _MERGE_RULES, model_name="minimax_m2")



def nvfp4_expert_source_spec(model_path: str, config):
    """The source spec the disk tier must index this checkpoint with.

    Same object the loader passes to ``load_nvfp4_expert_source_banks``, exposed so the
    shared provider can build the disk index without knowing the family: the index has to
    read the rows the loader placed, so one spec has to serve both."""
    return _NVFP4_SOURCE_SPEC

def load_nvfp4_expert_sources(
    model_path: str,
    config,
    *,
    layer_sink=None, disk_tier=None,
) -> dict[str, torch.Tensor]:
    """CPU NVFP4 expert source banks for the offload cache; see load_nvfp4_expert_source_banks."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
        disk_tier=disk_tier,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None, disk_tier=None
):
    """parallel: same NVFP4 source banks via the common chunked multi-threaded O_DIRECT reader."""
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
        disk_tier=disk_tier,
    )


__all__ = ["iter_weights", "load_nvfp4_expert_sources", "load_nvfp4_expert_sources_parallel"]
