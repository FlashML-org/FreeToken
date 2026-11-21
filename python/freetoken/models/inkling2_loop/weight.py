"""Inkling 2 Mini checkpoint reader: the original ``model.llm.*`` names, renamed to the FreeToken modules.

Routed experts are stacked per layer (``mlp.experts.w13_weight`` [E, 2I, ...] with gate and up rows
interleaved, ``w2_weight`` [E, H, ...]). NVFP4 layers carry fp8 block scales and a per-expert global scale
``w*_weight_amax / w*_weight_outer_scale_denominator``; MXFP8 layers carry e8m0 block scales and are dequantized
to bf16 here, since FreeToken has no MXFP8 expert kernel.
"""

from __future__ import annotations

import re
from typing import Collection, Iterator

import torch

from freetoken.layers.quantization import QuantKind

_LLM = "model.llm."
_VISUAL = "model.visual."
_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")
_EXPERT = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight(?:_(scale|amax|outer_scale_denominator))?$")
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# layer-relative checkpoint suffix -> FreeToken suffix, for tensors that only change name
_LAYER_RENAMES = {
    "attn_norm.weight": "input_layernorm",
    "q_input_norm.weight": "q_input_layernorm",
    "k_input_norm.weight": "k_input_layernorm",
    "v_input_norm.weight": "v_input_layernorm",
    "mlp_norm.weight": "post_attention_layernorm",
    "attn.wq_du.weight": "self_attn.q_proj.weight",
    "attn.wk_dv.weight": "self_attn.k_proj.weight",
    "attn.wv_dv.weight": "self_attn.v_proj.weight",
    "attn.wo_ud.weight": "self_attn.o_proj.weight",
    "attn.wr_du.weight": "self_attn.rel_proj.weight",
    "attn.gate_attn.weight": "self_attn.gate_proj.weight",
    "attn.attn_sink_proj.weight": "self_attn.sink_proj.weight",
    "attn.q_l2_clip.scale": "self_attn.q_scale",
    "attn.rel_l2_clip.scale": "self_attn.rel_scale",
    "attn.rel_logits_proj.proj": "self_attn.rel_table",
    "attn.k_sconv.static_w": "self_attn.k_conv",
    "attn.v_sconv.static_w": "self_attn.v_conv",
    "attn.k_sconv.weight_proj": "self_attn.k_conv_proj.weight",
    "attn.v_sconv.weight_proj": "self_attn.v_conv_proj.weight",
    "mlp.gate.weight": "mlp.gate_weight",
    "mlp.gate.bias": "mlp.gate_bias",
}
# static depthwise convolutions are stored as conv1d [C, 1, K] weights (cross-correlation): taps by lag are the flip
_CONV_RENAMES = {
    "attn.k_sconv.weight": "self_attn.k_conv",
    "attn.v_sconv.weight": "self_attn.v_conv",
    "attn_sconv.weight": "attn_conv",
    "mlp_sconv.weight": "mlp_conv",
}
_GATES = {"attn": "q_message_gate", "k": "k_message_gate", "v": "v_message_gate", "mlp": "mlp_message_gate"}
_GATE_LEAVES = {"down_weight": "down_proj.weight", "forget_weight": "forget_proj", "gain_weight": "message_proj"}


def _conv_taps(weight: torch.Tensor) -> torch.Tensor:
    """conv1d [C, 1, K] -> [K, C] taps by lag (row 0 weights the current token)."""
    return weight[:, 0, :].flip(-1).t().contiguous()


def rename(raw: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    """One text-tower tensor in FreeToken naming, or None for the experts and the message queries (merged elsewhere)."""
    if raw == "model.llm.unembed.weight":
        return "lm_head.weight", tensor
    if not raw.startswith(_LLM):
        return None
    name = raw[len(_LLM):]
    simple = {"embed.weight": "model.embed_tokens.weight", "embed_norm.weight": "model.embed_norm", "norm.weight": "model.norm"}
    if name in simple:
        return simple[name], tensor
    if name == "attn_res.output_query":
        return "model.final_query", tensor.view(1, -1)
    if m := re.fullmatch(r"attn_res\.attn_queries\.(\d+)", name):
        # the attention queries start at layer 1: layer 0 reads the embeddings directly
        return f"model.layers.{int(m.group(1)) + 1}.attention_query", tensor.view(-1, tensor.shape[-1])
    if m := re.fullmatch(r"attn_res\.mlp_queries\.(\d+)", name):
        return f"model.layers.{m.group(1)}.mlp_query", tensor.view(1, -1)
    m = _LAYER.match(name)
    if m is None:
        return None
    layer, leaf = m.group(1), m.group(2)
    prefix = f"model.layers.{layer}."
    if leaf in _LAYER_RENAMES:
        return prefix + _LAYER_RENAMES[leaf], tensor
    if leaf in _CONV_RENAMES:
        return prefix + _CONV_RENAMES[leaf], _conv_taps(tensor)
    if gm := re.fullmatch(r"controllers\.(attn|k|v|mlp)\.(down_weight|forget_weight|gain_weight)", leaf):
        return f"{prefix}{_GATES[gm.group(1)]}.{_GATE_LEAVES[gm.group(2)]}", tensor
    if leaf.startswith("mlp.experts."):
        return None
    raise ValueError(f"unexpected Inkling 2 tensor {raw}")


def merge_message_queries(queries: torch.Tensor, kv_queries: torch.Tensor) -> torch.Tensor:
    """The two [2 * loop layers, H] checkpoint tensors ([query, mlp] and [key, value] per loop layer) -> [loop layers, 4, H] (q, k, v, mlp)."""
    q = queries.unflatten(0, (-1, 2))
    kv = kv_queries.unflatten(0, (-1, 2))
    return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1).contiguous()


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    import safetensors

    from freetoken.distributed import get_tp_info
    from freetoken.models.loader import iter_weight_files

    if get_tp_info().size > 1:
        raise NotImplementedError("Inkling 2 Mini weight loading supports TP=1 only")
    if include_moe_experts:
        raise NotImplementedError("Inkling 2 experts load as expert banks (iter_expert_pieces), not as dense weights")
    if not include_non_moe:
        return
    queries: dict[str, torch.Tensor] = {}
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw in f.keys():
                if raw.startswith(_VISUAL):
                    if include_vision:
                        yield "visual." + raw[len(_VISUAL):], f.get_tensor(raw)
                    continue
                if _EXPERT.match(raw):
                    continue
                if raw.endswith(("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries")):
                    queries[raw.rsplit(".", 1)[-1]] = f.get_tensor(raw)
                    continue
                out = rename(raw, f.get_tensor(raw))
                if out is not None:
                    yield out
    if set(queries) != {"middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"}:
        raise ValueError(f"{model_path}: missing the loop message queries (found {sorted(queries)})")
    yield "model.message_queries", merge_message_queries(
        queries["middle_cycle_cross_pass_queries"], queries["middle_cycle_cross_pass_kv_queries"]
    )


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    import safetensors

    from freetoken.models.loader import iter_weight_files

    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw in f.keys():
                if raw.startswith(_VISUAL):
                    yield "visual." + raw[len(_VISUAL):], f.get_tensor(raw)


def deinterleave_rows(t: torch.Tensor) -> torch.Tensor:
    """[E, 2I, ...] with gate and up rows alternating -> [gate rows; up rows]."""
    raw = t.view(torch.uint8) if t.element_size() == 1 else t
    return torch.cat([raw[:, 0::2], raw[:, 1::2]], dim=1).view(t.dtype)


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """[E, R, C/2] e2m1 pairs (low nibble first), [E, R, C/16] fp8 scales, [E] globals -> [E, R, C] fp32."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = packed.to(torch.int64)
    vals = torch.stack([lut[codes & 0xF], lut[codes >> 4]], dim=-1).flatten(-2)
    vals = vals.unflatten(-1, (-1, 16)) * scale.float()[..., None]
    return vals.flatten(-2) * global_scale.float()[:, None, None]


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[E, R, C] e4m3 with [E, R, C/32] e8m0 exponents -> [E, R, C] fp32."""
    vals = weight.float().unflatten(-1, (-1, 32)) * torch.exp2(scale.float() - 127)[..., None]
    return vals.flatten(-2)


def _layer_tensors(f, layer: int) -> dict[str, torch.Tensor]:
    base = f"{_LLM}layers.{layer}.mlp.experts."
    return {name[len(base):]: f.get_tensor(name) for name in f.keys() if name.startswith(base)}


def iter_expert_pieces(model_path, config, kind: QuantKind, *, layers: Collection[int] | None = None, parallel=False, workers=8, chunk=8 << 20):
    """``(layer, 0, E, {role: tensor})`` per MoE layer whose experts are served as ``kind``: NVFP4 rows as stored
    (gate / up de-interleaved, the per-expert global from amax / denominator), MXFP8 dequantized to bf16 for ``NONE``."""
    import safetensors

    from freetoken.models.loader import iter_weight_files

    if kind not in (QuantKind.NVFP4, QuantKind.NONE):
        return None
    want = set(range(config.num_moe_layers)) if layers is None else set(layers)

    def pieces():
        seen: set[int] = set()
        for file in iter_weight_files(model_path):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                found = sorted({int(m.group(1)) for name in f.keys() if (m := _EXPERT.match(name))})
                for layer in found:
                    if layer not in want:
                        continue
                    t = _layer_tensors(f, layer)
                    nvfp4 = "w13_weight_amax" in t
                    if kind is QuantKind.NVFP4 and not nvfp4:
                        raise ValueError(f"layer {layer} experts are not NVFP4; the quant config serves them as NVFP4")
                    if nvfp4:
                        g13 = t["w13_weight_amax"].float() / t["w13_weight_outer_scale_denominator"].float()
                        g2 = t["w2_weight_amax"].float() / t["w2_weight_outer_scale_denominator"].float()
                        if kind is QuantKind.NVFP4:
                            piece = {
                                "gate_up": deinterleave_rows(t["w13_weight"]),
                                "gate_up_scale": deinterleave_rows(t["w13_weight_scale"]),
                                "gate_up_global": g13.view(-1, 1),
                                "down": t["w2_weight"],
                                "down_scale": t["w2_weight_scale"],
                                "down_global": g2.view(-1, 1),
                            }
                        else:
                            piece = {
                                "gate_up": deinterleave_rows(dequant_nvfp4(t["w13_weight"], t["w13_weight_scale"], g13)).to(torch.bfloat16),
                                "down": dequant_nvfp4(t["w2_weight"], t["w2_weight_scale"], g2).to(torch.bfloat16),
                            }
                    else:
                        if t["w13_weight"].dtype != torch.float8_e4m3fn:
                            raise ValueError(f"layer {layer} experts are neither NVFP4 nor MXFP8 ({t['w13_weight'].dtype})")
                        piece = {
                            "gate_up": deinterleave_rows(dequant_mxfp8(t["w13_weight"], t["w13_weight_scale"])).to(torch.bfloat16),
                            "down": dequant_mxfp8(t["w2_weight"], t["w2_weight_scale"]).to(torch.bfloat16),
                        }
                    seen.add(layer)
                    yield layer, 0, config.num_experts, piece
        if missing := sorted(want - seen):
            raise ValueError(f"{model_path}: no routed experts for MoE layers {missing}")

    return pieces()


__all__ = ["dequant_mxfp8", "dequant_nvfp4", "deinterleave_rows", "iter_expert_pieces", "iter_vision_weights", "iter_weights", "merge_message_queries", "rename"]
