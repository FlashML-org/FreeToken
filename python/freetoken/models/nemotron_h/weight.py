from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch

from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config


_EXPERT_RE = re.compile(r"^backbone\.layers\.\d+\.mixer\.experts\.\d+\.")
_EXPERT_KEY_RE = re.compile(
    r"^backbone\.layers\.(?P<layer>\d+)\.mixer\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)


def _layer_to_bank(layer: int, config) -> int | None:
    try:
        return (config.moe_layer_ids or ()).index(layer)
    except ValueError:
        return None


_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="Nemotron-H NVFP4 experts",
    gated=False,
    hidden_size_attr="expert_hidden_size",
)


def _dequant_nvfp4(weight, scale, global_scale):
    from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4_weight

    return _dequant_nvfp4_weight(weight, scale, global_scale)


def _skip(name: str) -> bool:
    return (
        name.startswith("mtp.")
        or name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale"))
    )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    del include_moe_experts  # routed experts always live in native NVFP4 offload banks
    if not include_non_moe:
        return
    if get_tp_info().size > 1:
        raise NotImplementedError("Nemotron-H currently supports TP=1 only")

    config = parse_config(cached_load_hf_config(model_path))
    qkv: dict[str, dict[str, torch.Tensor]] = {}

    for file in tqdm(
        iter_weight_files(model_path), desc="Loading Nemotron-H weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keys = set(f.keys())
            for name in f.keys():
                if _skip(name) or _EXPERT_RE.match(name):
                    continue
                if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                    continue
                tensor = f.get_tensor(name)
                if name.endswith(".weight"):
                    base = name[:-7]
                    scale_name = base + ".weight_scale"
                    global_name = base + ".weight_scale_2"
                    if global_name in keys:
                        tensor = _dequant_nvfp4(
                            tensor, f.get_tensor(scale_name), f.get_tensor(global_name)
                        )
                    elif scale_name in keys:
                        # Keep checkpoint-native E4M3 and turn its scalar into the per-row
                        # vector consumed by FreeToken's W8A16/W8A8 linear.
                        yield name, tensor
                        scale = f.get_tensor(scale_name).reshape(1).float()
                        yield base + ".weight_scale", scale.expand(tensor.shape[0]).contiguous()
                        input_name = base + ".input_scale"
                        if input_name in keys:
                            yield input_name, f.get_tensor(input_name).reshape(()).float()
                        continue

                    match = re.match(
                        r"^(backbone\.layers\.\d+\.mixer)\.(q_proj|k_proj|v_proj)\.weight$",
                        name,
                    )
                    if match:
                        prefix, role = match.groups()
                        slots = qkv.setdefault(prefix, {})
                        slots[role] = tensor
                        if len(slots) == 3:
                            yield prefix + ".qkv_proj.weight", torch.cat(
                                [slots[r] for r in ("q_proj", "k_proj", "v_proj")], dim=0
                            )
                            del qkv[prefix]
                        continue
                yield name, tensor
        drop_page_cache(file)
    assert not qkv, f"incomplete Nemotron-H qkv fusions: {list(qkv)}"


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path, config, _SOURCE_SPEC, drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(), layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    return load_nvfp4_expert_source_banks_parallel(
        model_path, config, _SOURCE_SPEC, drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(), workers=workers, chunk=chunk,
        layer_sink=layer_sink,
    )


def dummy_nvfp4_expert_sources(config):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    L, E = config.num_moe_layers, config.num_experts
    H, I = config.expert_hidden_size, config.moe_intermediate_size
    fp8 = torch.float8_e4m3fn

    def bank(*shape, dtype):
        return [alloc_pinned_tensor(*shape, dtype=dtype) for _ in range(L)]

    sources = {
        "gate_up_packed": bank(E, I, H // 2, dtype=torch.uint8),
        "gate_up_scale": bank(E, I, H // 16, dtype=fp8),
        "gate_up_global": bank(E, I, dtype=torch.float16),
        "down_packed": bank(E, H, I // 2, dtype=torch.uint8),
        "down_scale": bank(E, H, I // 16, dtype=fp8),
        "down_global": bank(E, H, dtype=torch.float16),
    }
    for tensor in sources["gate_up_packed"] + sources["down_packed"]:
        tensor.random_(0, 256)
    for tensor in sources["gate_up_scale"] + sources["down_scale"]:
        tensor.fill_(1.0)
    for tensor in sources["gate_up_global"] + sources["down_global"]:
        tensor.fill_(0.01)
    return sources


__all__ = [
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "dummy_nvfp4_expert_sources",
]
