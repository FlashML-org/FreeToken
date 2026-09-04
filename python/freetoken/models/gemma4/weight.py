from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.models.loader import (
    MergeRule,
    drop_page_cache,
    iter_weight_files,
)
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_PACKED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.feed_forward\.experts\.(?P<name>gate_up_proj|down_proj)$"
)

# NVFP4 routed experts (nvidia modelopt checkpoint): per-expert, un-fused, under the raw
# ``model.language_model.layers.N.experts.E.{proj}`` key (no .mlp./.feed_forward. infix).
# Matched against the RAW weight_map key in nvfp4_banks (it never sees the renamed key).
_NVFP4_EXPERT_RE = re.compile(r"\.experts\.\d+\.")
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE (no dense prefix)
    desc="Gemma4 NVFP4 experts",
)
_LAYER_INDEX_PATTERN = re.compile(r"layers\.(\d+)\.")
_LAYER_FF_PREFIX_PATTERN = re.compile(r"^(model\.layers\.\d+)\.")
_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}
_FEED_FORWARD_PREFIXES = (
    "experts.",
    "router.",
    "layer_scalar",
    "post_feedforward_layernorm.",
    "post_feedforward_layernorm_1.",
    "post_feedforward_layernorm_2.",
    "pre_feedforward_layernorm_2.",
)

# modelopt-NVFP4 dense MLP (nvidia/Gemma-4-31B-IT-NVFP4): mlp.{gate,up,down}_proj are W4A16
# FP4 -- uint8 weight + fp8-e4m3 block weight_scale + per-tensor weight_scale_2 + input_scale.
# The scales are consumed with their .weight; input_scale is unused (W4A16). Mirrors the
# qwen3_5_moe native-NVFP4 dense loader.
_NVFP4_DENSE_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
_NVFP4_DENSE_MLP_RE = re.compile(r"\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def _nvfp4_dense_parts(f, raw_base: str):
    """Load an NVFP4 dense weight as the W4A16 kernel's buffers: (weight uint8 [O, IN//2],
    weight_scale fp8-e4m3 block [O, IN//16], weight_global fp16 [O] from the per-tensor
    weight_scale_2 broadcast per output row)."""
    w = f.get_tensor(raw_base + ".weight")
    s = f.get_tensor(raw_base + ".weight_scale")
    g = f.get_tensor(raw_base + ".weight_scale_2").reshape(1).to(torch.float16)
    g = g.expand(w.shape[0]).contiguous()
    assert (
        w.dtype is torch.uint8
        and s.dtype is torch.float8_e4m3fn
        and g.dtype is torch.float16
    ), f"unexpected NVFP4 dense dtypes at {raw_base}: {w.dtype}/{s.dtype}/{g.dtype}"
    return w, s, g


def _emit_nvfp4_dense_mlp(f, base: str, raw_base: str, buf: dict):
    """(key, tensor) triples for an NVFP4 dense MLP projection: down_proj standalone;
    gate_proj/up_proj merged output-wise into gate_up_proj (each keeps its own scales, so the
    fused weight is exact). Returns [] while a gate/up merge is still buffered."""
    w, s, g = _nvfp4_dense_parts(f, raw_base)
    if base.endswith(".down_proj"):
        return [(base + ".weight", w), (base + ".weight_scale", s), (base + ".weight_global", g)]
    is_gate = base.endswith(".gate_proj")
    prefix = base[: -len(".gate_proj")] if is_gate else base[: -len(".up_proj")]
    slots = buf.setdefault(prefix, {})
    slots["gate" if is_gate else "up"] = (w, s, g)
    if "gate" not in slots or "up" not in slots:
        return []
    gw, gs, gg = slots["gate"]
    uw, us, ug = slots["up"]
    del buf[prefix]
    pre = prefix + ".gate_up_proj"
    return [
        (pre + ".weight", torch.cat([gw, uw], dim=0)),
        (pre + ".weight_scale", torch.cat([gs, us], dim=0)),
        (pre + ".weight_global", torch.cat([gg, ug], dim=0)),
    ]


def _rename_language_key(raw_name: str) -> str:
    name = raw_name.removeprefix("model.language_model.")
    name = "model." + name.removeprefix("language_model.")

    match = _LAYER_FF_PREFIX_PATTERN.match(name)
    if match is None:
        return name

    layer_prefix = match.group(1)
    layer_key = name[match.end() :]
    if layer_key.startswith("mlp."):
        return f"{layer_prefix}.feed_forward.shared_mlp.{layer_key.removeprefix('mlp.')}"
    if layer_key.startswith(_FEED_FORWARD_PREFIXES):
        return f"{layer_prefix}.feed_forward.{layer_key}"
    return name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    def rename_key(raw_name: str, *, include_vision: bool) -> str | None:
        prefix = "model.language_model."
        if raw_name.startswith(prefix):
            return _rename_language_key(raw_name)
        if raw_name.startswith("language_model."):
            return _rename_language_key(raw_name)
        if include_vision:
            if raw_name.startswith("model.vision_tower."):
                return ("vision_tower." + raw_name[len("model.vision_tower.") :]).replace(
                    ".linear.",
                    ".",
                )
            if raw_name.startswith("model.embed_vision."):
                return "embed_vision." + raw_name[len("model.embed_vision.") :]
        return None

    def merge_info(key: str) -> tuple[str, MergeRule] | None:
        for suffix, rule in _MERGE_RULES.items():
            if key.endswith(suffix + ".weight") or key.endswith(suffix):
                return key.replace(suffix, rule.fused_suffix), rule
        return None

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()
    if tp_info.size > 1:
        raise NotImplementedError("Gemma 4 weight loading currently supports TP=1 only")

    include_vision = config.is_multimodal
    k_eq_v_layers = {
        layer_id
        for layer_id in range(config.num_layers)
        if isinstance(config.attention_group_for_layer(layer_id), FullAttentionGroupConfig)
        and config.attention_group_for_layer(layer_id).k_eq_v
    }
    merge_buf: dict[str, dict[str, torch.Tensor]] = {}
    gateup_buf: dict[str, dict[str, tuple]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not tp_info.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                name = rename_key(raw_name, include_vision=include_vision)
                if name is None:
                    continue

                # Per-expert NVFP4 tensors go to the offload cache (load_nvfp4_expert_sources),
                # not this dense pass; fused bf16/q4_0 experts lack ".experts.<int>." so are unaffected.
                if _NVFP4_EXPERT_RE.search(raw_name):
                    continue

                # NVFP4 dense-MLP scales are consumed with their .weight (below), never yielded.
                if raw_name.endswith(_NVFP4_DENSE_SCALE_SUFFIXES):
                    continue

                is_vision = name.startswith(("vision_tower.", "embed_vision."))
                is_expert = (
                    not is_vision and _PACKED_EXPERT_PATTERN.match(name) is not None
                )
                if is_expert and not include_moe_experts:
                    continue
                if not is_expert and not include_non_moe:
                    continue

                # Native W4A16 NVFP4 dense MLP: the .weight is FP4-packed and carries block +
                # per-tensor scales. The keyset guard (weight_scale_2 sibling present) is
                # defense-in-depth beyond config.dense_quant -- the sibling MoE checkpoint's
                # bf16 shared_mlp has no such sibling, so it falls through to the bf16 path.
                if (
                    config.dense_quant == "nvfp4"
                    and not is_vision
                    and not is_expert
                    and _NVFP4_DENSE_MLP_RE.search(raw_name)
                    and raw_name[: -len(".weight")] + ".weight_scale_2" in keyset
                ):
                    yield from _emit_nvfp4_dense_mlp(
                        f, name[: -len(".weight")], raw_name[: -len(".weight")], gateup_buf
                    )
                    continue

                tensor = f.get_tensor(raw_name)
                if is_vision or is_expert:
                    yield name, tensor
                    continue

                info = merge_info(name)
                if info is None:
                    yield name, tensor
                    continue

                merged_key, rule = info
                slots = merge_buf.setdefault(merged_key, {})
                slots[rule.slot] = tensor
                if rule.slot == "k" and k_eq_v_layers:
                    layer_match = _LAYER_INDEX_PATTERN.search(name)
                    if (
                        layer_match is not None
                        and int(layer_match.group(1)) in k_eq_v_layers
                    ):
                        slots["v"] = tensor
                if not all(slot in slots for slot in rule.slots):
                    continue
                parts = [slots[slot] for slot in rule.slots]
                del merge_buf[merged_key]
                yield merged_key, torch.cat(parts, dim=0)

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not gateup_buf, f"Incomplete NVFP4 gate/up merges: {list(gateup_buf.keys())}"


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only parallel reader: gemma-4 packs experts per-layer
    (``feed_forward.experts.{gate_up_proj,down_proj}``), so no merge needed; same key
    rename as iter_weights, read via the common chunked O_DIRECT reader."""
    assert include_moe_experts and not include_non_moe, (
        "gemma4 parallel reader is experts-only (used by load_moe_expert_sources)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    if get_tp_info().size > 1:
        raise NotImplementedError("Gemma 4 weight loading currently supports TP=1 only")

    def _expert_name(raw_name: str) -> str | None:
        if raw_name.startswith("model.language_model.") or raw_name.startswith("language_model."):
            name = _rename_language_key(raw_name)
            if _PACKED_EXPERT_PATTERN.match(name) is not None:
                return name
        return None

    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, lambda rn: _expert_name(rn) is not None, workers=workers, chunk=chunk
    ):
        yield _expert_name(raw_name), tensor



def nvfp4_expert_source_spec(model_path: str, config):
    """The source spec the disk tier must index this checkpoint with.

    Same object the loader passes to ``load_nvfp4_expert_source_banks``, exposed so the
    shared provider can build the disk index without knowing the family: the index has to
    read the rows the loader placed, so one spec has to serve both."""
    return _NVFP4_SOURCE_SPEC

def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None, disk_tier=None
) -> dict[str, list[torch.Tensor]]:
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
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
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


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
