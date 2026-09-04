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
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config
from .df11_embedding import compress_df11_embedding
from .df11_linear import compress_df11_weight

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
    ``weight_scale_2`` broadcast to per-row fp16 ``weight_global`` as LinearNVFP4 /
    dequant_nvfp4 expect. lossless vs checkpoint; same dequant math as routed experts.
    """
    packed = reader.get(f"{src_prefix}.weight")  # [OUT, IN//2] uint8
    scale = reader.get(f"{src_prefix}.weight_scale")  # [OUT, IN//16] fp8-e4m3
    g = reader.get(f"{src_prefix}.weight_scale_2").reshape(()).to(torch.float16)  # scalar
    yield f"{dst_prefix}.weight", packed
    yield f"{dst_prefix}.weight_scale", scale
    yield f"{dst_prefix}.weight_global", g.expand(packed.shape[0]).contiguous()


def _iter_attn_df11(
    reader: _ShardReader, prefix: str, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield DF11 buffers for an attention projection from its bf16 checkpoint weight.

    qkvo are bf16 in the checkpoint; DF11 compresses them losslessly (~10.7 bits/weight) to
    fit a 32 GB VRAM target, decoding bit-for-bit.
    """
    w = reader.get(f"{prefix}.weight").to(torch.bfloat16)
    bundle = compress_df11_weight(w)
    for name in ("low8", "bitstream", "chunk_start", "lut"):
        yield f"{prefix}.{name}", bundle[name]


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the resident (non routed-expert) weights for GLM-4 MoE.

    - qkvo: lossless DF11 (+ bf16 .bias) so the ~25GB of bf16 attention fits a 32 GB VRAM target.
    - leading dense MLP layers + each MoE layer's shared expert: native NVFP4 (dequant in
      forward), faithful and smallest footprint.
    - embedding: row-contiguous DF11, decode only looked-up rows. router gate (+
      e_score_correction_bias), norms, lm_head pass through.
    - MTP layer (index num_layers) and routed experts are skipped.
    """
    assert not include_moe_experts, (
        "GLM-4 MoE stores experts as NVFP4 and only supports the offload backend; experts "
        "are loaded into the offload cache via load_nvfp4_expert_sources()."
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
    device = reader._device
    L = config.num_layers
    dense = config.first_k_dense_replace

    for layer in tqdm(range(L), desc="Loading GLM dense weights", disable=not primary):
        a = f"model.layers.{layer}.self_attn"
        # DF11 projections (+ qkv bias, bias-free o_proj).
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield from _iter_attn_df11(reader, f"{a}.{proj}", device)
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
            # router (bf16 gate + fp32 selection bias -> bf16) and shared expert.
            yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight")
            yield (
                f"{m}.e_score_correction_bias",
                reader.get(f"{m}.gate.e_score_correction_bias").to(torch.bfloat16),
            )
            s = f"{m}.shared_experts"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield from _iter_nvfp4_resident(reader, f"{s}.{proj}", f"{s}.{proj}")

    # bf16 embedding -> row-contiguous DF11 (~30% smaller), decode only looked-up rows.
    embed = reader.get("model.embed_tokens.weight").to(torch.bfloat16)
    for name, buf in compress_df11_embedding(embed).items():
        yield f"model.embed_tokens.{name}", buf
    del embed
    yield "model.norm.weight", reader.get("model.norm.weight")
    # lm_head stays bf16: full-vocab matmul needs a decode scratch as big as the weight, so
    # DF11 nets no savings (unlike the gathered embedding lookup).
    yield "lm_head.weight", reader.get("lm_head.weight")


# --------------------------------------------------------------------------------------
# Routed expert host banks (NVFP4) for the offload cache.
# --------------------------------------------------------------------------------------

def nvfp4_expert_source_spec(model_path: str, config):
    """The source spec the disk tier must index this checkpoint with.

    Same object the loader passes to ``load_nvfp4_expert_source_banks``, exposed so the
    shared provider can build the disk index without knowing the family: the index has to
    read the rows the loader placed, so one spec has to serve both."""
    return _NVFP4_SOURCE_SPEC

def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None, disk_tier=None) -> dict[str, torch.Tensor]:
    """Build the pinned CPU NVFP4 banks for GLM-4's routed experts.

    experts exist only for layers [first_k_dense_replace, num_layers) and pack by MoE layer
    index (layer - first_k_dense_replace) so banks have no holes for leading dense layers.
    MTP layer excluded.
    """
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
