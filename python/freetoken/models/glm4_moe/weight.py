from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config

# routed experts go to the offload cache, not the dense model.
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_ROUTED_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_ROUTED_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    ),
    desc="GLM NVFP4 experts",
)


# --------------------------------------------------------------------------------------
# Dense / resident weight streaming.
# --------------------------------------------------------------------------------------
class _ShardReader:
    """Opens safetensors shards on demand (mmap) and serves tensors on ``device``."""

    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _iter_nvfp4_resident(
    reader: _ShardReader, src_prefix: str, dst_prefix: str
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield native NVFP4 buffers for an always-resident Linear.

    ``weight`` (packed fp4) + ``weight_scale`` (fp8 block scale) verbatim; per-tensor
    ``weight_scale_2`` broadcast to per-row fp32 ``weight_global`` as the NVFP4 linear
    method / dequant_nvfp4 expect. lossless vs checkpoint; same dequant math as routed experts.
    """
    packed = reader.get(f"{src_prefix}.weight")  # [OUT, IN//2] uint8
    scale = reader.get(f"{src_prefix}.weight_scale")  # [OUT, IN//16] fp8-e4m3
    g = reader.get(f"{src_prefix}.weight_scale_2").reshape(()).to(torch.float32)  # scalar
    yield f"{dst_prefix}.weight", packed
    yield f"{dst_prefix}.weight_scale", scale
    yield f"{dst_prefix}.weight_global", g.expand(packed.shape[0]).contiguous()
    if reader.has(f"{src_prefix}.input_scale"):
        yield f"{dst_prefix}.input_scale", reader.get(f"{src_prefix}.input_scale").reshape(()).to(torch.float32)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the resident (non routed-expert) weights for GLM-4 MoE.

    - qkvo (+ bias), embedding, router gate, norms, lm_head: bf16 as stored; the router's
      e_score_correction_bias in fp32.
    - leading dense MLP layers + each MoE layer's shared expert: native NVFP4 (dequant in
      forward), faithful and smallest footprint.
    - MTP layer (index num_layers) and routed experts are skipped.
    """
    assert not include_moe_experts, (
        "GLM-4 MoE stores experts as NVFP4 and only supports the offload backend; experts "
        "are loaded into the offload cache from their NVFP4 pieces."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    try:
        yield from _iter_resident_weights(reader, config, primary)
    finally:
        # drop shard page cache so the ~176GB expert banks (allocated next) start clean.
        reader.close()


def _iter_resident_weights(reader, config, primary) -> Iterator[tuple[str, torch.Tensor]]:
    L = config.num_layers
    dense = config.first_k_dense_replace

    for layer in tqdm(range(L), desc="Loading GLM dense weights", disable=not primary):
        a = f"model.layers.{layer}.self_attn"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield f"{a}.{proj}.weight", reader.get(f"{a}.{proj}.weight").to(torch.bfloat16)
            bias_name = f"{a}.{proj}.bias"
            if reader.has(bias_name):
                yield bias_name, reader.get(bias_name).to(torch.bfloat16)
        for norm in ("q_norm", "k_norm"):
            name = f"{a}.{norm}.weight"
            if reader.has(name):
                yield name, reader.get(name)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield (
                f"model.layers.{layer}.{norm}.weight",
                reader.get(f"model.layers.{layer}.{norm}.weight"),
            )

        m = f"model.layers.{layer}.mlp"
        if layer < dense:
            # dense SwiGLU MLP as native NVFP4, separate gate/up/down.
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield from _iter_nvfp4_resident(reader, f"{m}.{proj}", f"{m}.{proj}")
        else:
            # router (bf16 gate + fp32 selection bias) and shared expert.
            yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight")
            yield (
                f"{m}.e_score_correction_bias",
                reader.get(f"{m}.gate.e_score_correction_bias").to(torch.float32),
            )
            s = f"{m}.shared_experts"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield from _iter_nvfp4_resident(reader, f"{s}.{proj}", f"{s}.{proj}")

    yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight").to(torch.bfloat16)
    yield "model.norm.weight", reader.get("model.norm.weight")
    yield "lm_head.weight", reader.get("lm_head.weight")


# --------------------------------------------------------------------------------------
# Routed expert host banks (NVFP4) for the offload cache.
# --------------------------------------------------------------------------------------
def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = ["iter_weights", "nvfp4_expert_spec"]
