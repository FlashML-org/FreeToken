"""Inkling 2 checkpoint reader: the dense tensors under FreeToken names, the routed experts as pieces.

The checkpoint keeps Thinking Machines' own tensor names (``model.llm.*``, ``model.visual.*``).
Stacked experts store gate and up rows interleaved (``w13`` row ``2i`` is gate row ``i``,
``2i + 1`` up row ``i``); NVFP4 layers carry a per-expert global ``amax / outer_scale_denominator``,
MXFP8 layers ue8m0 scales per 32 inputs.
"""

from __future__ import annotations

import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import ShardReader
from tqdm import tqdm

from .config import Inkling2Args

_LLM = "model.llm."
_LAYER = re.compile(r"model\.llm\.layers\.(\d+)\.(.+)")
# checkpoint controller -> (model gate group, the group's controllers in order)
_GATES = {
    "attn": ("attn_gates", ("attn", "k", "v")),
    "k": ("attn_gates", ("attn", "k", "v")),
    "v": ("attn_gates", ("attn", "k", "v")),
    "mlp": ("mlp_gate", ("mlp",)),
}


def _args(model_path: str) -> Inkling2Args:
    from freetoken.utils import cached_load_hf_config

    from .config import _text_config, parse_args

    return parse_args(_text_config(cached_load_hf_config(model_path)))


def _lag_taps(weight: torch.Tensor) -> torch.Tensor:
    """``[C, 1, K]`` conv1d weight (tap ``K - 1`` on the current row) -> ``[C, K]`` fp32 by backward lag."""
    return weight[:, 0, :].flip(-1).float().contiguous()


def _deinterleave(w13: torch.Tensor) -> torch.Tensor:
    """``[E, 2I, ...]`` gate/up-interleaved rows -> ``[gate; up]`` halves."""
    return torch.cat([w13[:, 0::2], w13[:, 1::2]], dim=1)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return  # experts load as pieces (iter_expert_pieces)
    if get_tp_info().size > 1:
        raise NotImplementedError("inkling2 weight loading supports TP=1 only")
    args = _args(model_path)
    reader = ShardReader(model_path, device)
    get = reader.get_tensor
    try:
        names = [n for f in reader.files() for n in reader.names_in(f)]
        for name in tqdm(names, desc="Loading weights", disable=not get_tp_info().is_primary()):
            if name.startswith("model.visual."):
                if include_vision:
                    yield from _vision_tensor(name, get)
                continue
            m = _LAYER.fullmatch(name)
            if m is not None:
                yield from _layer_tensor(int(m.group(1)), m.group(2), name, get, args)
                continue
            leaf = name[len(_LLM):]
            if leaf == "embed.weight":
                yield "model.embed_tokens.weight", get(name)
            elif leaf == "embed_norm.weight":
                yield "model.embed_norm.weight", get(name)
            elif leaf == "norm.weight":
                yield "model.norm", get(name).view(1, -1)
            elif leaf == "unembed.weight":
                yield "lm_head.weight", get(name)[: args.unpadded_vocab_size].contiguous()
            elif leaf == "attn_res.output_query":
                yield "model.final_query", get(name).view(1, -1)
            elif leaf.startswith("attn_res.attn_queries."):
                # attention residual queries start at layer 1: layer 0 reads its input directly
                yield f"model.layers.{int(leaf.rsplit('.', 1)[1]) + 1}.attention_input", get(name).view(-1, args.hidden_size)
            elif leaf.startswith("attn_res.mlp_queries."):
                yield f"model.layers.{int(leaf.rsplit('.', 1)[1])}.mlp_input", get(name).view(1, -1)
            elif leaf == "middle_cycle_cross_pass_queries":
                # rows (2j, 2j + 1) are loop layer j's query and MLP messages, the kv tensor its key and value ones
                qm = get(name).unflatten(0, (-1, 2))
                kv = get(_LLM + "middle_cycle_cross_pass_kv_queries").unflatten(0, (-1, 2))
                yield "model.message_queries", torch.cat([qm[:, :1], kv, qm[:, 1:]], dim=1).contiguous()
            elif leaf == "middle_cycle_cross_pass_kv_queries":
                continue
            else:
                raise ValueError(f"unexpected inkling2 tensor {name}")
    finally:
        reader.close()


def _layer_tensor(layer: int, leaf: str, name: str, get, args: Inkling2Args) -> Iterator[tuple[str, torch.Tensor]]:
    p = f"model.layers.{layer}."
    base = name[: -len(leaf)]
    swa = args.is_swa(layer)
    if leaf.startswith("mlp.experts."):
        return
    if leaf == "attn.wq_du.weight":
        parts = [get(base + f"attn.{n}.weight") for n in ("wq_du", "wr_du", "gate_attn", "attn_sink_proj")]
        yield p + "self_attn.q_side.weight", torch.cat(parts, dim=0)
    elif leaf in ("attn.wk_dv.weight", "attn.wv_dv.weight"):
        kind = leaf[len("attn.w")]
        parts = [get(name)] + ([] if swa else [get(base + f"attn.{kind}_sconv.weight_proj")])
        yield p + f"self_attn.{kind}_side.weight", torch.cat(parts, dim=0)
    elif leaf in ("attn.k_sconv.weight", "attn.v_sconv.weight"):
        yield p + f"self_attn.{leaf[5]}_conv", _lag_taps(get(name))
    elif leaf in ("attn.k_sconv.static_w", "attn.v_sconv.static_w"):
        # the per-token convolution's static taps are already indexed by backward lag
        yield p + f"self_attn.{leaf[5]}_conv", get(name).t().float().contiguous()
    elif leaf == "attn.wo_ud.weight":
        yield p + "self_attn.o_proj.weight", get(name)
    elif leaf == "attn.q_l2_clip.scale":
        yield p + "self_attn.q_scale", get(name)
    elif leaf == "attn.rel_l2_clip.scale":
        yield p + "self_attn.rel_scale", get(name)
    elif leaf == "attn.rel_logits_proj.proj":
        yield p + "self_attn.rel_proj", get(name)
    elif leaf in ("attn.wr_du.weight", "attn.gate_attn.weight", "attn.attn_sink_proj.weight",
                  "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj"):
        return  # fused above
    elif leaf == "attn_norm.weight":
        yield p + "input_norm.weight", get(name)
    elif leaf == "q_input_norm.weight":
        yield p + "qkv_input_norm", torch.stack([get(base + f"{kind}_input_norm.weight") for kind in "qkv"])
    elif leaf in ("k_input_norm.weight", "v_input_norm.weight"):
        return  # stacked with q's
    elif leaf == "mlp_norm.weight":
        yield p + "mlp_norm", get(name).view(1, -1)
    elif leaf == "attn_sconv.weight":
        yield p + "attn_conv", _lag_taps(get(name))
    elif leaf == "mlp_sconv.weight":
        yield p + "mlp_conv", _lag_taps(get(name))
    elif leaf == "mlp.gate.weight":
        yield p + "mlp.router_weight", get(name)
    elif leaf == "mlp.gate.bias":
        yield p + "mlp.router_bias", get(name).float()
    elif leaf.startswith("controllers."):
        # the query, key and value gates run side by side; the MLP gate on its own
        _, which, w = leaf.split(".")
        if which not in _GATES or w not in ("down_weight", "forget_weight", "gain_weight"):
            raise ValueError(f"unexpected inkling2 tensor {name}")
        group, members = _GATES[which]
        if which != members[0] or w == "gain_weight":
            return
        ctl = base + "controllers."
        if w == "down_weight":
            yield p + f"{group}.down", torch.stack([get(ctl + f"{m}.down_weight") for m in members])
        else:
            up = [torch.cat([get(ctl + f"{m}.forget_weight"), get(ctl + f"{m}.gain_weight")]) for m in members]
            yield p + f"{group}.up", torch.stack(up)
    else:
        raise ValueError(f"unexpected inkling2 tensor {name}")


_VISION_LAYER = re.compile(r"model\.visual\.backbone\.layers\.(\d+)\.(.+)")
_VISION_LAYER_LEAVES = {
    "ln_1": "layernorm_before", "ln_2": "layernorm_after", "mlp.c_fc": "fc1", "mlp.c_proj": "fc2", "self_attn.out_proj": "out_proj",
}
_VISION_TOP = {
    "model.visual.backbone.embeddings.patch_embedding.weight": "visual.patch_embedding.weight",
    "model.visual.backbone.embeddings.position_embedding": "visual.position_embedding",
    "model.visual.backbone.embeddings.class_embedding": "visual.class_embedding",
    "model.visual.backbone.embeddings.cls_position_embedding": "visual.cls_position_embedding",
    "model.visual.backbone.ln_pre.weight": "visual.pre_layernorm.weight",
    "model.visual.backbone.ln_pre.bias": "visual.pre_layernorm.bias",
    "model.visual.proj_in.weight": "visual.merger_in.weight",
    "model.visual.proj_out.weight": "visual.merger_out.weight",
}


def _vision_tensor(name: str, get) -> Iterator[tuple[str, torch.Tensor]]:
    if name in _VISION_TOP:
        yield _VISION_TOP[name], get(name)
        return
    m = _VISION_LAYER.fullmatch(name)
    if m is None:
        raise ValueError(f"unexpected inkling2 vision tensor {name}")
    p, leaf = f"visual.layers.{m.group(1)}.", m.group(2)
    stem, _, kind = leaf.rpartition(".")
    if stem == "self_attn.q_proj":
        base = name[: -len(leaf)]
        yield p + f"qkv.{kind}", torch.cat([get(base + f"self_attn.{x}_proj.{kind}") for x in "qkv"])
    elif stem in ("self_attn.k_proj", "self_attn.v_proj"):
        return  # fused with q
    elif leaf in ("ls_1.gamma", "ls_2.gamma"):
        yield p + f"lambda_{leaf[3]}", get(name)
    elif stem in _VISION_LAYER_LEAVES:
        yield p + f"{_VISION_LAYER_LEAVES[stem]}.{kind}", get(name)
    else:
        raise ValueError(f"unexpected inkling2 vision tensor {name}")


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    reader = ShardReader(model_path, device)
    try:
        for f in reader.files():
            for name in reader.names_in(f):
                if name.startswith("model.visual."):
                    yield from _vision_tensor(name, reader.get_tensor)
    finally:
        reader.close()


def iter_expert_pieces(model_path: str, config, kind: QuantKind, *, layers=None, parallel=False, workers: int = 8, chunk: int = 8 << 20):
    """NVFP4 layers as NVFP4 pieces, MXFP8 layers as e4m3 rows with their ue8m0 codes."""
    if kind not in (QuantKind.NVFP4, QuantKind.MXFP8):
        return None
    if parallel:
        raise NotImplementedError("inkling2 has no parallel expert reader")
    return _iter_expert_pieces(model_path, config, kind, layers)


def _iter_expert_pieces(model_path: str, config, kind: QuantKind, layers) -> Iterator:
    args: Inkling2Args = config.inkling2_args
    want = range(args.num_layers) if layers is None else sorted(layers)
    reader = ShardReader(model_path, torch.device("cpu"))
    try:
        for layer in want:
            p = f"{_LLM}layers.{layer}.mlp.experts."
            stored_fp4 = reader.has(p + "w13_weight_amax")
            if stored_fp4 != (kind is QuantKind.NVFP4):
                raise ValueError(f"layer {layer} experts are stored {'nvfp4' if stored_fp4 else 'mxfp8'} but were asked for as {kind}")
            piece = {}
            for proj, role in (("w13", "gate_up"), ("w2", "down")):
                w, s = reader.get_tensor(p + f"{proj}_weight"), reader.get_tensor(p + f"{proj}_weight_scale")
                if proj == "w13":
                    w, s = _deinterleave(w), _deinterleave(s)
                piece[role], piece[f"{role}_scale"] = w, s
                if stored_fp4:
                    amax = reader.get_tensor(p + f"{proj}_weight_amax").float()
                    denom = reader.get_tensor(p + f"{proj}_weight_outer_scale_denominator").float()
                    piece[f"{role}_global"] = amax / denom
            yield layer, 0, args.num_experts, piece
    finally:
        reader.close()


__all__ = ["iter_expert_pieces", "iter_vision_weights", "iter_weights"]
