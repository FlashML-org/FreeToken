from __future__ import annotations

import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    drop_page_cache,
    iter_merged_tensors,
    iter_stacked_experts,
    safetensors_weight_map,
    shard_tensor,
)
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
_SINK_KEY_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.attention_sink_bias$")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_ROUTED_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    # bank = layer - first_k_dense_replace (moe_layer_freq = [0, 1x47] => layer 1 -> bank 0)
    layer_to_bank=lambda layer, config: (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    ),
    desc="MiMo NVFP4 experts",
    # modelopt weight_scale_2 is the dequant-side global (same as the GLM banks); the
    # reciprocal is only for compressed-tensors weight_global_scale (nvfp4_banks.py).
    kind_map=None,
    global_reciprocal=False,
    validate_weight_map=lambda weight_map, config: _validate_weight_map(weight_map, config),
)

# unquantized checkpoints: per-expert tensors (source dtype) merge (gate|up) and stack (E)
# into the resident MoELayer's experts.gate_up_proj / experts.down_proj (qwen3_moe's format).
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_EXPERT_MERGE_RULES = {
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


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


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the resident weights for MiMo v2, expert form chosen by the checkpoint's quant dialect.

    - fused qkv_proj (ONE checkpoint tensor/layer, kept fused on the merged layer) and the
      layer-0 dense MLP are F32 in the checkpoint and pass through in source dtype;
      o_proj / embed / norms / router gate / lm_head pass through. Tensors are yielded
      UNCONVERTED -- the engine's single conversion to EngineConfig.dtype
      (``_materialize_loaded_weight_state_dict``) is the only dtype decision.
    - attention sinks: SWA-only BF16 ``attention_sink_bias`` -> engine ``sinks`` cast to fp32.
    - ``model.mtp.*``, ``visual.*``, ``audio_encoder.*`` and ``speech_embeddings.*`` are
      never yielded; the strict loader would reject them.
    - MTP beyond the main layers is always skipped.
    - unquantized checkpoints (``expert_quant == "none"``) keep experts resident:
      ``include_moe_experts=True`` yields per-layer stacked ``experts.gate_up_proj`` /
      ``experts.down_proj``, the format the resident MoELayer consumes.
    - NVFP4 checkpoints serve experts only through the offload cache's expert pieces, so
      ``include_moe_experts=True`` stays rejected there.
    """
    tp_size = get_tp_info().size
    if tp_size != 1:
        raise ValueError(f"MiMo v2 supports tensor parallel size 1 only; got TP={tp_size}")
    config = parse_config(cached_load_hf_config(model_path))
    if config.expert_quant == "none":
        assert include_moe_experts or include_non_moe
        folder = download_hf_weight(model_path)
        weight_map = safetensors_weight_map(folder)
        _validate_weight_map(weight_map, config)
        reader = _ShardReader(folder, weight_map, device)
        primary = get_tp_info().is_primary()
        try:
            if include_non_moe:
                yield from _iter_resident_weights(reader, config, primary)
            if include_moe_experts:
                yield from _iter_stacked_unquantized_experts(reader, config, primary)
        finally:
            reader.close()
        return
    assert not include_moe_experts, (
        "MiMo v2 stores experts as NVFP4 and only supports the offload backend; experts "
        "are loaded into the offload cache from their NVFP4 pieces."
    )
    assert include_non_moe
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    _validate_weight_map(weight_map, config)
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    try:
        yield from _iter_resident_weights(reader, config, primary)
    finally:
        # drop shard page cache so the expert banks (allocated next) start clean.
        reader.close()


def _validate_weight_map(weight_map: dict[str, str], config) -> None:
    expected_sinks = {
        f"model.layers.{layer}.self_attn.attention_sink_bias"
        for group in config.attention_groups
        if group.name == "swa"
        for layer in group.layer_ids
    }
    actual_sinks = {name for name in weight_map if _SINK_KEY_RE.fullmatch(name)}
    if actual_sinks != expected_sinks:
        raise ValueError(
            "MiMo v2 checkpoint must contain attention_sink_bias only on every SWA layer "
            f"(missing={sorted(expected_sinks - actual_sinks)[:3]}, "
            f"unexpected={sorted(actual_sinks - expected_sinks)[:3]})"
        )
    if config.expert_quant == "nvfp4":
        unsupported_scales = {
            name
            for name in weight_map
            if ".mlp.experts." in name
            and name.endswith(
                (".input_scale", ".weight_global_scale", ".weight_global", ".weight_scale_inv")
            )
        }
        if unsupported_scales:
            raise ValueError(
                "MiMo v2 W4A16_NVFP4 expects ModelOpt weight_scale_2 scales without "
                f"activation scales; found {sorted(unsupported_scales)[:3]}"
            )


def _iter_stacked_unquantized_experts(reader, config, primary) -> Iterator[tuple[str, torch.Tensor]]:
    """Per-expert tensors in source dtype -> TP-sharded, gate|up-merged, expert-stacked resident tensors."""
    tp_info = get_tp_info()

    def sharded() -> Iterator[tuple[str, torch.Tensor]]:
        for layer in range(config.first_k_dense_replace, config.num_layers):
            m = f"model.layers.{layer}.mlp"
            for e in range(config.num_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    name = f"{m}.experts.{e}.{proj}.weight"
                    raw = reader.get(name)
                    yield name, shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw

    merged = iter_merged_tensors(sharded(), _EXPERT_MERGE_RULES, model_name="mimo_v2")
    stacked = iter_stacked_experts(
        merged,
        num_experts=config.num_experts,
        model_name="mimo_v2",
        expert_pattern=_EXPERT_PATTERN,
    )
    yield from tqdm(stacked, desc="Loading MiMo expert weights", disable=not primary)


def _iter_resident_weights(reader, config, primary) -> Iterator[tuple[str, torch.Tensor]]:
    L = config.num_layers
    dense = config.first_k_dense_replace

    for layer in tqdm(range(L), desc="Loading MiMo dense weights", disable=not primary):
        a = f"model.layers.{layer}.self_attn"
        # fused [Q | K | V] qkv stays one fused tensor; the checkpoint stores it F32 and it
        # is yielded verbatim (the engine casts once to the engine dtype).
        yield f"{a}.qkv_proj.weight", reader.get(f"{a}.qkv_proj.weight")
        yield f"{a}.o_proj.weight", reader.get(f"{a}.o_proj.weight")
        sink = f"{a}.attention_sink_bias"
        if reader.has(sink):
            # engine sinks are fp32; the checkpoint ships BF16 on the 39 SWA layers only.
            sink_weight = reader.get(sink)
            expected_shape = (config.num_qo_heads,)
            if tuple(sink_weight.shape) != expected_shape:
                raise ValueError(
                    f"MiMo v2 {sink} has shape {tuple(sink_weight.shape)}, expected {expected_shape}"
                )
            yield f"{a}.sinks", sink_weight.to(torch.float32)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield (
                f"model.layers.{layer}.{norm}.weight",
                reader.get(f"model.layers.{layer}.{norm}.weight"),
            )

        m = f"model.layers.{layer}.mlp"
        if layer < dense:
            # dense SwiGLU MLP, separate gate/up/down; F32 in the checkpoint, yielded verbatim.
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield f"{m}.{proj}.weight", reader.get(f"{m}.{proj}.weight")
        else:
            # router (bf16 gate + fp32 selection bias); no shared experts.
            yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight")
            yield (
                f"{m}.e_score_correction_bias",
                reader.get(f"{m}.gate.e_score_correction_bias").to(torch.float32),
            )

    yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight")
    yield "model.norm.weight", reader.get("model.norm.weight")
    # tied heads reuse embed_tokens as the output head (ParallelLMHead.state_dict omits
    # lm_head.weight when tied): never require or yield the redundant tensor, even if stored
    if not config.tie_word_embeddings:
        yield "lm_head.weight", reader.get("lm_head.weight")


# --------------------------------------------------------------------------------------
# Routed expert host banks (NVFP4) for the offload cache.
# --------------------------------------------------------------------------------------
def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = ["iter_weights", "nvfp4_expert_spec"]
