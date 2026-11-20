"""Inkling 2 Mini (model_type ``inkling2_loop``): config parsing, the loop schedule and the checkpoint's expert quantization.

Layers ``loop_start:loop_end`` run ``num_passes`` times per token with shared weights. Every run of a layer
("access") keeps its own KV and convolution state, so the KV pool and the state pool hold one cache layer
per access: weight layer ``L`` on pass 0 uses cache layer ``L``, later passes of a loop layer use cache
layers after the weight layers (pass ``p``: ``num_layers + (p - 1) * num_loop_layers + L - loop_start``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Tuple

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
    SWAAttentionGroupConfig,
)

# per-request convolution windows riding the linear-state slots, one tensor per convolution of a cache layer
CONV_STATES = ("inkling2_conv_k", "inkling2_conv_v", "inkling2_conv_attn", "inkling2_conv_mlp")


@dataclass(frozen=True)
class Inkling2LoopArgs:
    num_layers: int
    sliding_layers: frozenset[int]
    loop_start: int
    loop_end: int
    num_passes: int
    gate_rank: int
    block_size: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    swa_num_heads: int
    swa_num_kv_heads: int
    swa_head_dim: int
    sliding_window: int
    d_rel: int
    rel_extent: int
    log_n_floor: int
    log_alpha: float
    qk_clip: float
    rel_clip: float
    conv_kernel: int
    gate_softcap: float
    up_softcap: float
    route_scale: float
    unpadded_vocab_size: int
    rms_norm_eps: float

    @property
    def num_loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def num_cache_layers(self) -> int:
        return self.num_layers + (self.num_passes - 1) * self.num_loop_layers

    @property
    def num_accesses(self) -> int:
        return self.num_cache_layers

    def is_loop_layer(self, layer: int) -> bool:
        return self.loop_start <= layer < self.loop_end

    def is_sliding(self, layer: int) -> bool:
        return layer in self.sliding_layers

    def cache_layer(self, layer: int, pass_idx: int) -> int:
        if pass_idx == 0:
            return layer
        assert self.is_loop_layer(layer) and pass_idx < self.num_passes, (layer, pass_idx)
        return self.num_layers + (pass_idx - 1) * self.num_loop_layers + layer - self.loop_start

    def access(self, layer: int, pass_idx: int) -> int:
        """Execution-order index of one run of ``layer``: 0.. for the prelude, then the passes, then the coda."""
        if layer < self.loop_start:
            return layer
        if layer >= self.loop_end:
            return layer + (self.num_passes - 1) * self.num_loop_layers
        return self.loop_start + pass_idx * self.num_loop_layers + layer - self.loop_start

    def schedule(self) -> Tuple[Tuple[int, int], ...]:
        """``(weight layer, pass)`` of every access, in execution order."""
        prelude = [(layer, 0) for layer in range(self.loop_start)]
        passes = [(layer, p) for p in range(self.num_passes) for layer in range(self.loop_start, self.loop_end)]
        coda = [(layer, 0) for layer in range(self.loop_end, self.num_layers)]
        return tuple(prelude + passes + coda)

    def cache_layers_of(self, sliding: bool) -> Tuple[int, ...]:
        return tuple(sorted(self.cache_layer(layer, p) for layer, p in self.schedule() if self.is_sliding(layer) == sliding))

    def kv_dim(self, layer: int) -> int:
        return (self.swa_num_kv_heads * self.swa_head_dim) if self.is_sliding(layer) else (self.num_kv_heads * self.head_dim)

    def conv_widths(self) -> Tuple[int, int, int, int]:
        """Channels of the k, v, post-attention and post-MLP convolutions; the state pool sizes one width per cache layer."""
        kv = max(self.kv_dim(layer) for layer in range(self.num_layers))
        return kv, kv, self.hidden_size, self.hidden_size


def _get(cfg: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)
        if value is not None:
            return value
    return default


def _loop_section(text: Any) -> dict:
    loop = _get(text, "loop_config")
    if loop is None:
        # a transformers build that knows the model flattens loop_config into the text config
        return {
            "core_start": _get(text, "loop_layer_index_start"),
            "core_end": _get(text, "loop_layer_index_end"),
            "num_passes": _get(text, "loop_num_passes"),
            "gate_rank": _get(text, "loop_gate_rank"),
        }
    return loop if isinstance(loop, dict) else (loop.to_dict() if hasattr(loop, "to_dict") else dict(vars(loop).get("_data", {})))


def parse_args(text: Any) -> Inkling2LoopArgs:
    num_layers = int(_get(text, "num_hidden_layers"))
    local = _get(text, "local_layer_ids")
    if local is None:
        layer_types = _get(text, "layer_types")
        local = [i for i, t in enumerate(layer_types) if t == "hybrid_sliding"]
    loop = _loop_section(text)
    if loop.get("is_varloop"):
        raise NotImplementedError("variable-loop Inkling 2 checkpoints are not served; this build runs a fixed number of passes")
    args = Inkling2LoopArgs(
        num_layers=num_layers,
        sliding_layers=frozenset(int(i) for i in local),
        loop_start=int(loop["core_start"]),
        loop_end=int(loop["core_end"]),
        num_passes=int(loop["num_passes"]),
        gate_rank=int(loop["gate_rank"]),
        block_size=int(_get(text, "n_layers_per_attn_res_block", default=6)),
        hidden_size=int(_get(text, "hidden_size")),
        num_heads=int(_get(text, "num_attention_heads")),
        num_kv_heads=int(_get(text, "num_key_value_heads")),
        head_dim=int(_get(text, "head_dim")),
        swa_num_heads=int(_get(text, "swa_num_attention_heads")),
        swa_num_kv_heads=int(_get(text, "swa_num_key_value_heads")),
        swa_head_dim=int(_get(text, "swa_head_dim")),
        sliding_window=int(_get(text, "sliding_window_size", "sliding_window")),
        d_rel=int(_get(text, "d_rel")),
        rel_extent=int(_get(text, "rel_extent")),
        log_n_floor=int(_get(text, "log_scaling_n_floor")),
        log_alpha=float(_get(text, "log_scaling_alpha")),
        qk_clip=float(_get(text, "qk_l2_clip_threshold")),
        rel_clip=float(_get(text, "rel_l2_clip_threshold")),
        conv_kernel=int(_get(text, "sconv_kernel_size", "conv_kernel_size", default=4)),
        gate_softcap=float(_get(text, "mlp_gate_softcap")),
        up_softcap=float(_get(text, "mlp_up_softcap")),
        route_scale=float(_get(text, "route_scale", "routed_scaling_factor")),
        unpadded_vocab_size=int(_get(text, "unpadded_vocab_size")),
        rms_norm_eps=float(_get(text, "rms_norm_eps", default=1e-6)),
    )
    if args.loop_start <= 0 or args.loop_start % args.block_size or args.loop_end % args.block_size:
        raise ValueError(f"loop layers {args.loop_start}:{args.loop_end} must start and end on attention-residual block boundaries")
    if args.swa_num_heads != args.num_heads or args.swa_head_dim != args.head_dim:
        raise NotImplementedError("Inkling 2 checkpoints whose sliding and global layers differ in query heads or head size are not served")
    if _get(text, "n_shared_experts", default=0):
        raise NotImplementedError("Inkling 2 checkpoints with shared experts are not served")
    return args


def conv_slot_states(args: Inkling2LoopArgs) -> Tuple[SlotStateSpec, ...]:
    from .conv import ConvWindowVerify  # conv imports this module

    layers = tuple(range(args.num_cache_layers))
    return tuple(
        SlotStateSpec(name=name, shape=(args.conv_kernel - 1, width), layer_ids=layers, verify=ConvWindowVerify)
        for name, width in zip(CONV_STATES, args.conv_widths())
    )


def parse_config(hf_config: Any) -> ModelConfig:
    from .vision import parse_vision_config  # vision imports the layers

    text = _get(hf_config, "text_config", default=hf_config)
    args = parse_args(text)
    # no rope: positions only size the page table; the attention reads them as relative distances
    rotary = RotaryConfig(
        head_dim=args.head_dim,
        rotary_dim=args.head_dim,
        max_position=int(_get(text, "model_max_length", "max_position_embeddings", default=1 << 20)),
        base=1.0,
        scaling=None,
    )
    architectures = list(_get(hf_config, "architectures", default=None) or ["Inkling2LoopForConditionalGeneration"])
    return ModelConfig(
        num_layers=args.num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        vocab_size=int(_get(text, "vocab_size")),
        intermediate_size=int(_get(text, "moe_intermediate_size", "intermediate_size")),
        rms_norm_eps=args.rms_norm_eps,
        rotary_config=rotary,
        hidden_act="swiglu_softcap",
        tie_word_embeddings=False,
        num_experts=int(_get(text, "n_routed_experts", "num_local_experts")),
        num_experts_per_tok=int(_get(text, "num_experts_per_tok")),
        moe_intermediate_size=int(_get(text, "moe_intermediate_size", "intermediate_size")),
        norm_topk_prob=True,
        model_type="inkling2_loop",
        architectures=architectures,
        moe_enabled=True,
        expert_quant="nvfp4" if _get(hf_config, "quantization_config") is not None else "none",
        has_router_bias=True,
        routed_scaling_factor=args.route_scale,
        final_logit_softcapping=float(_get(text, "final_logit_softcapping")),
        vision_config=parse_vision_config(hf_config),
        image_token_id=int(_get(hf_config, "image_token_id", default=200054)),
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=args.cache_layers_of(sliding=False),
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                rotary_config=rotary,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=args.cache_layers_of(sliding=True),
                num_kv_heads=args.swa_num_kv_heads,
                head_dim=args.swa_head_dim,
                rotary_config=rotary,
                sliding_window=args.sliding_window,
            ),
        ),
        num_kv_layers=args.num_cache_layers,
        slot_states=conv_slot_states(args),
        no_prefix_reuse=(
            "the convolution windows of each cache layer are not snapshotted at prefix boundaries, "
            "so a cached prefix cannot be resumed"
        ),
        inkling2_args=args,
    )


_EXPERT_TENSOR = re.compile(r"^(?P<module>.+\.mlp\.experts)\.w(13|2)_weight$")
# hf_quant_config.json compute dtype -> the ModelOpt algo FreeToken serves it as; experts in other formats
# (MXFP8) load through the bf16 path, dequantized by iter_expert_pieces
_SERVED_ALGOS = {"nvfp4": "W4A16_NVFP4"}


def quant_config(model_path: str, hf_config: Any, *, name_map, unquantized):
    """The checkpoint's expert formats as a ModelOpt MIXED_PRECISION config.

    The config.json says ``modelopt`` / ``NVFP4`` for the whole model, but only the routed experts of some
    layers are NVFP4; ``hf_quant_config.json`` lists the tensors of each format. The activation
    quantization it declares for the NVFP4 experts is not applied (W4A16), as in the reference."""
    from freetoken.layers.quantization import ModelOptConfig, QuantConfig
    from freetoken.utils.hf import optional_hf_file

    path = optional_hf_file(model_path, "hf_quant_config.json")
    if path is None:
        return QuantConfig.from_hf(hf_config, name_map=name_map, unquantized=unquantized)
    with open(path, encoding="utf-8") as f:
        groups = (json.load(f).get("model") or {}).get("parameter_groups") or []
    layers: dict[str, dict] = {}
    for group in groups:
        algo = _SERVED_ALGOS.get(str(group.get("compute_dtype")))
        if algo is None:
            continue
        for tensor in group.get("tensors") or ():
            if (m := _EXPERT_TENSOR.match(tensor)) is not None:
                layers[m.group("module")] = {"quant_algo": algo}
    q = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}
    if not layers:
        from freetoken.layers.quantization import NoQuantConfig

        return NoQuantConfig(name_map, unquantized)
    return ModelOptConfig(q, hf_config, name_map=name_map, unquantized=unquantized)


__all__ = ["CONV_STATES", "Inkling2LoopArgs", "conv_slot_states", "parse_args", "parse_config", "quant_config"]
