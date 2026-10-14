from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.models.loader import (
    MergeRule,
    ShardReader,
    drop_page_cache,
    iter_weight_files,
)
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
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
_CT_FP8_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale)$"
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
# The scales are consumed with their .weight.
_NVFP4_DENSE_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
_NVFP4_DENSE_MLP_RE = re.compile(r"\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
_CT_FP8_SCALE_SUFFIX = ".weight_scale"


def _nvfp4_dense_parts(reader: ShardReader, raw_base: str):
    """Load an NVFP4 dense weight as the NVFP4 linear method's buffers: weight uint8 [O, IN//2], weight_scale fp8-e4m3 block [O, IN//16], weight_global fp16 [O] (the per-tensor weight_scale_2 per output row), input_scale fp32 scalar or None when the export has none."""
    w = reader.get_tensor(raw_base + ".weight")
    s = reader.get_tensor(raw_base + ".weight_scale")
    g = reader.get_tensor(raw_base + ".weight_scale_2").reshape(1).to(torch.float16)
    g = g.expand(w.shape[0]).contiguous()
    assert (
        w.dtype is torch.uint8
        and s.dtype is torch.float8_e4m3fn
        and g.dtype is torch.float16
    ), f"unexpected NVFP4 dense dtypes at {raw_base}: {w.dtype}/{s.dtype}/{g.dtype}"
    a = reader.get_tensor(raw_base + ".input_scale").reshape(()).to(torch.float32) if reader.has(raw_base + ".input_scale") else None
    return w, s, g, a


def _dequant_ct_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize compressed-tensors channel-wise FP8 weights to BF16."""
    if scale.numel() == weight.shape[0]:
        scale = scale.reshape(weight.shape[0], 1)
    elif scale.numel() != 1:
        raise ValueError(
            f"unexpected Gemma compressed-tensors scale shape {tuple(scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )
    return weight.to(torch.bfloat16) * scale.to(torch.bfloat16)


def _emit_nvfp4_dense_mlp(reader: ShardReader, base: str, raw_base: str, buf: dict):
    """(key, tensor) pairs for an NVFP4 dense MLP projection: down_proj standalone;
    gate_proj/up_proj merged output-wise into gate_up_proj (each keeps its own scales, so the
    fused weight is exact). Returns [] while a gate/up merge is still buffered."""
    w, s, g, a = _nvfp4_dense_parts(reader, raw_base)
    if base.endswith(".down_proj"):
        out = [(base + ".weight", w), (base + ".weight_scale", s), (base + ".weight_global", g)]
        return out + ([(base + ".input_scale", a)] if a is not None else [])
    is_gate = base.endswith(".gate_proj")
    prefix = base[: -len(".gate_proj")] if is_gate else base[: -len(".up_proj")]
    slots = buf.setdefault(prefix, {})
    slots["gate" if is_gate else "up"] = (w, s, g, a)
    if "gate" not in slots or "up" not in slots:
        return []
    gw, gs, gg, ga = slots["gate"]
    uw, us, ug, ua = slots["up"]
    del buf[prefix]
    pre = prefix + ".gate_up_proj"
    out = [
        (pre + ".weight", torch.cat([gw, uw], dim=0)),
        (pre + ".weight_scale", torch.cat([gs, us], dim=0)),
        (pre + ".weight_global", torch.cat([gg, ug], dim=0)),
    ]
    if ga is not None and ua is not None:
        # both parts read the same activation; the larger range covers both
        out.append((pre + ".input_scale", torch.maximum(ga, ua)))
    return out


def _rename_vision_key(raw_name: str) -> str | None:
    """Encoder checkpoint key -> state-dict key, or None for anything else; the tower's clipped-linear wrapper is dropped."""
    if raw_name.startswith("model.vision_tower."):
        return ("vision_tower." + raw_name[len("model.vision_tower.") :]).replace(".linear.", ".")
    if raw_name.startswith("model.embed_vision."):
        return "embed_vision." + raw_name[len("model.embed_vision.") :]
    if raw_name.startswith("model.vision_embedder."):
        return "vision_embedder." + raw_name[len("model.vision_embedder.") :]
    return None


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision encoder alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename_vision_key(raw_name)
                if name is not None:
                    yield name, f.get_tensor(raw_name)


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
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    def rename_key(raw_name: str) -> str | None:
        prefix = "model.language_model."
        if raw_name.startswith(prefix):
            return _rename_language_key(raw_name)
        if raw_name.startswith("language_model."):
            return _rename_language_key(raw_name)
        return _rename_vision_key(raw_name)

    def merge_info(key: str) -> tuple[str, MergeRule] | None:
        for suffix, rule in _MERGE_RULES.items():
            if key.endswith(suffix + ".weight") or key.endswith(suffix + ".weight_scale") or key.endswith(suffix):
                return key.replace(suffix, rule.fused_suffix), rule
        return None

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()
    if tp_info.size > 1:
        raise NotImplementedError("Gemma 4 weight loading currently supports TP=1 only")

    k_eq_v_layers = {
        layer_id
        for layer_id in range(config.num_layers)
        if isinstance(config.attention_group_for_layer(layer_id), FullAttentionGroupConfig)
        and config.attention_group_for_layer(layer_id).k_eq_v
    }
    merge_buf: dict[str, dict[str, torch.Tensor]] = {}
    gateup_buf: dict[str, dict[str, tuple]] = {}
    ct_reader = ShardReader(model_path, device) if config.expert_quant == "compressed-tensors" else None
    files = ct_reader.files() if ct_reader is not None else iter_weight_files(model_path)
    try:
        for file in tqdm(
            files,
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            names = ct_reader.names_in(file) if ct_reader is not None else None
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                keyset = set(names if names is not None else f.keys())
                for raw_name in keyset:
                    name = rename_key(raw_name)
                    if name is None:
                        continue

                    # Per-expert NVFP4 tensors go to the offload cache (load_nvfp4_expert_sources),
                    # not this dense pass; fused bf16/q4_0 experts lack ".experts.<int>." so are unaffected.
                    if _NVFP4_EXPERT_RE.search(raw_name):
                        continue

                    if config.expert_quant == "compressed-tensors":
                        # Dense Linear layers use the FP8 quant method and must retain the
                        # raw FP8 weight plus its scale. Routed Gemma experts bypass that
                        # method and are dequantized into the BF16 offload banks instead.
                        is_ct_expert = _CT_FP8_EXPERT_KEY_RE.match(raw_name) is not None
                        is_shared_mlp = ".feed_forward.shared_mlp." in name
                        if (is_ct_expert or is_shared_mlp) and raw_name.endswith(_CT_FP8_SCALE_SUFFIX):
                            continue
                        if (is_ct_expert or is_shared_mlp) and raw_name.endswith(".weight"):
                            scale_name = raw_name.removesuffix(".weight") + _CT_FP8_SCALE_SUFFIX
                            tensor = _dequant_ct_fp8_weight(
                                ct_reader.get_tensor(raw_name),
                                ct_reader.get_tensor(scale_name),
                            )
                        else:
                            tensor = f.get_tensor(raw_name)
                            if raw_name.endswith(_CT_FP8_SCALE_SUFFIX):
                                tensor = tensor.reshape(-1)
                    else:
                        tensor = f.get_tensor(raw_name)

                    # NVFP4 dense-MLP scales are consumed with their .weight (below), never yielded.
                    if config.dense_quant == "nvfp4" and raw_name.endswith(_NVFP4_DENSE_SCALE_SUFFIXES):
                        continue

                    is_vision = name.startswith(("vision_tower.", "embed_vision.", "vision_embedder."))
                    if is_vision and not include_vision:
                        continue
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

    finally:
        if ct_reader is not None:
            ct_reader.close()

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
        "gemma4 parallel reader is experts-only (used by the expert piece reader)"
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


def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """CPU NVFP4 expert source banks for the offload cache; see load_nvfp4_expert_source_banks."""
    if model_path.endswith(".gguf"):
        from .gguf import load_nvfp4_expert_sources as load_gguf_nvfp4_expert_sources

        return load_gguf_nvfp4_expert_sources(model_path, config, layer_sink=layer_sink)
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def _load_ct_fp8_expert_banks(model_path: str, config, *, layer_sink=None):
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import LayerCompletionTracker, alloc_layer_banks, pin_banks

    L, E, H, I = (
        config.num_layers,
        config.num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
    )
    banks = alloc_layer_banks(
        {
            "gate_up": ((E, 2 * I, H), torch.bfloat16),
            "down": ((E, H, I), torch.bfloat16),
        },
        L,
    )
    tracker = LayerCompletionTracker(2, banks, layer_sink) if layer_sink is not None else None
    seen: set[tuple[int, int, str]] = set()
    primary = get_tp_info().is_primary()

    reader = ShardReader(model_path, torch.device("cpu"))
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading Gemma compressed-tensors FP8 experts",
            disable=not primary,
        ):
            for raw_name in reader.names_in(file):
                match = _CT_FP8_EXPERT_KEY_RE.match(raw_name)
                if match is None or match["kind"] != "weight":
                    continue
                layer = int(match["layer"])
                expert = int(match["expert"])
                proj = match["proj"]
                base = raw_name.removesuffix(".weight")
                scale_name = base + ".weight_scale"
                try:
                    scale = reader.get_tensor(scale_name)
                except KeyError as exc:
                    raise ValueError(f"missing scale tensor {scale_name!r}") from exc
                value = _dequant_ct_fp8_weight(reader.get_tensor(raw_name), scale)
                if proj == "gate_proj":
                    banks["gate_up"][layer].tensor[expert, :I].copy_(value)
                elif proj == "up_proj":
                    banks["gate_up"][layer].tensor[expert, I:].copy_(value)
                else:
                    banks["down"][layer].tensor[expert].copy_(value)
                seen.add((layer, expert, proj))
    finally:
        reader.close()

    expected = {(layer, expert, proj) for layer in range(L) for expert in range(E)
                for proj in ("gate_proj", "up_proj", "down_proj")}
    missing = expected - seen
    if missing:
        raise ValueError(f"missing Gemma compressed-tensors expert weights: {sorted(missing)[:4]}")

    if tracker is not None:
        for layer in range(L):
            tracker.note(layer)
            tracker.note(layer)
    else:
        pin_banks(banks)
    return ExpertBanks(
        "bf16",
        {name: [bank.tensor for bank in layer_banks] for name, layer_banks in banks.items()},
        streamed=layer_sink is not None,
    )


def setup_offload_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    decode_target: str = "gpu",
    layer_sink=None,
):
    """Load Gemma's compressed-tensors per-channel FP8 experts as BF16 banks."""
    from freetoken.moe.expert_banks import ExpertBanks

    if getattr(model_config, "expert_quant", "none") == "compressed-tensors":
        if dummy:
            from freetoken.models.weight import dummy_moe_expert_sources

            gate_up, down = dummy_moe_expert_sources(model_config, dtype=torch.bfloat16)
            return ExpertBanks("bf16", {"gate_up": gate_up, "down": down})
        return _load_ct_fp8_expert_banks(model_path, model_config, layer_sink=layer_sink)

    from freetoken.moe.expert_banks import _PROVIDERS

    provider = _PROVIDERS[model_config.expert_quant]
    return provider(
        model_path,
        model_config,
        device,
        dtype,
        dummy,
        parallel=parallel,
        workers=workers,
        chunk=chunk,
        decode_target=decode_target,
        layer_sink=layer_sink,
    )


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = [
    "nvfp4_expert_spec",
    "iter_weights",
    "iter_weights_parallel",
    "setup_offload_expert_banks",
]
