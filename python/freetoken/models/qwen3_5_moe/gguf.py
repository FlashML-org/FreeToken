"""Native Qwen3.6 GGUF loading for the Q4_K_M control checkpoint.

The checkpoint remains block-quantized end to end.  Q8_0 and Q6_K dense
projections are retained as packed tensors and execute through FreeToken's
native ggml HIP kernels.  Routed expert gate/up rows stay Q4_K and down rows
stay Q5_K in the AMD offload cache.  Only scalar parameters such as norms,
router weights, and the Gated DeltaNet recurrence parameters are materialized
as bf16 or fp32, because they are stored as F32 in the GGUF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import ModelConfig
from freetoken.models.gguf.dequant import (
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)


# F32 GGUF tensors whose runtime parameter has a direct one-to-one mapping.
# The Gated DeltaNet alpha/beta naming describes the recurrence semantics:
# alpha maps to the softplus ``a`` input and beta maps to the sigmoid ``b`` input.
_SCALAR_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
_GDN_BA_SUFFIXES = {"ssm_alpha.weight": "a", "ssm_beta.weight": "b"}


def _to_bf16(t) -> torch.Tensor:
    """Dequantize one GGUF scalar tensor to its logical torch shape."""
    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape)


def _require_weight_tp1() -> None:
    """Reject TP before loading unsharded GGUF packed rows."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen3.5 GGUF weight loading currently supports TP=1 only")


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield every non-routed-expert Qwen GGUF parameter in runtime key order.

    Full-attention Q/K/V and Gated DeltaNet qkv/z/b/a each arrive as individual
    GGUF tensors.  FreeToken executes them as fused projections, so their packed
    rows are concatenated only on the output axis.  This is byte preserving because
    every fused member has the same input width and quantization type (Q8_0).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert not include_moe_experts, "Qwen GGUF routed experts are supplied by the offload cache"
    assert include_non_moe
    _require_weight_tp1()

    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    gdn_buf: dict[int, dict[str, torch.Tensor]] = {}
    shared_buf: dict[int, dict[str, torch.Tensor]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            if t.ggml_type != GGML_Q8_0:
                raise ValueError(f"{name} expected Q8_0, got {t.ggml_type}")
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            if t.ggml_type != GGML_Q6_K:
                raise ValueError(f"{name} expected Q6_K, got {t.ggml_type}")
            yield "lm_head.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t) + 1.0
            continue
        if not name.startswith("blk."):
            continue

        parts = name.split(".")
        layer = int(parts[1])
        suffix = ".".join(parts[2:])
        base = f"model.layers.{layer}"
        if suffix in _EXPERT_SUFFIXES:
            continue
        if suffix in _GDN_BA_SUFFIXES:
            # The split GGUF path keeps qkv|z packed Q8_0, while recurrence b|a
            # remains a conventional dense fused projection.  The runtime order is
            # explicitly b then a, matching Qwen3_5GatedDeltaNet._in_proj_split.
            gdn_buf.setdefault(layer, {})[_GDN_BA_SUFFIXES[suffix]] = _to_bf16(t)
            slots = gdn_buf[layer]
            if all(key in slots for key in ("b", "a")):
                yield f"{base}.linear_attn.in_proj_ba.weight", torch.cat(
                    [slots.pop("b"), slots.pop("a")], dim=0
                )
                if not slots:
                    del gdn_buf[layer]
            continue
        if suffix in _SCALAR_MAP:
            tensor = _to_bf16(t)
            rel = _SCALAR_MAP[suffix]
            if suffix == "ssm_conv1d.weight":
                # GGUF stores depthwise filters as [channels, kernel]; FreeToken's
                # causal-convolution holder uses the PyTorch depthwise layout
                # [channels, 1, kernel].
                tensor = tensor.unsqueeze(1)
            elif suffix == "ffn_gate_inp_shexp.weight":
                # The single shared-expert gate is stored as a vector in GGUF but
                # executes as a one-row replicated linear projection.
                tensor = tensor.unsqueeze(0)
            # Gemma-style norms carry the delta from unity in GGUF.  GDN's gated RMS
            # norm is conventional and intentionally excluded from this adjustment.
            if rel.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                             "self_attn.q_norm.weight", "self_attn.k_norm.weight")):
                tensor = tensor + 1.0
            if rel.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
                tensor = tensor.to(torch.float32)
            yield f"{base}.{rel}", tensor
            continue

        if suffix == "attn_q.weight":
            qkv_buf.setdefault(layer, {})["qg"] = t.packed()
        elif suffix == "attn_k.weight":
            qkv_buf.setdefault(layer, {})["k"] = t.packed()
        elif suffix == "attn_v.weight":
            qkv_buf.setdefault(layer, {})["v"] = t.packed()
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
        elif suffix == "attn_qkv.weight":
            gdn_buf.setdefault(layer, {})["qkv"] = t.packed()
        elif suffix == "attn_gate.weight":
            gdn_buf.setdefault(layer, {})["z"] = t.packed()
        elif suffix == "ssm_out.weight":
            yield f"{base}.linear_attn.out_proj.qweight", t.packed()
        elif suffix == "ffn_gate_shexp.weight":
            shared_buf.setdefault(layer, {})["gate"] = t.packed()
        elif suffix == "ffn_up_shexp.weight":
            shared_buf.setdefault(layer, {})["up"] = t.packed()
        elif suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
        else:
            raise ValueError(f"unmapped Qwen3.5 GGUF tensor: {name}")

        slots = qkv_buf.get(layer)
        if slots is not None and all(key in slots for key in ("qg", "k", "v")):
            yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                [slots["qg"], slots["k"], slots["v"]], dim=0
            )
            del qkv_buf[layer]
        slots = gdn_buf.get(layer)
        if slots is not None and all(key in slots for key in ("qkv", "z")):
            # qkv|z is quantized; b|a are F32 tensors and are loaded below as dense.
            yield f"{base}.linear_attn.in_proj_qkvz.qweight", torch.cat(
                [slots["qkv"], slots["z"]], dim=0
            )
            del slots["qkv"], slots["z"]
            if not slots:
                del gdn_buf[layer]
        slots = shared_buf.get(layer)
        if slots is not None and all(key in slots for key in ("gate", "up")):
            yield f"{base}.mlp.shared_expert.gate_up_proj.qweight", torch.cat(
                [slots["gate"], slots["up"]], dim=0
            )
            del shared_buf[layer]

    assert not qkv_buf, f"incomplete Qwen attention QKV groups: {sorted(qkv_buf)}"
    assert not gdn_buf, f"incomplete Qwen GDN qkv/z groups: {sorted(gdn_buf)}"
    assert not shared_buf, f"incomplete Qwen shared gate/up groups: {sorted(shared_buf)}"


class GGUFLMHead(BaseOP):
    """Untied Q6_K language head that preserves last-token prefill semantics."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, GGML_Q6_K), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, GGML_Q6_K)


def is_gguf_model(config: ModelConfig) -> bool:
    """Return whether this model uses the Qwen packed-GGUF runtime path."""
    return getattr(config, "moe_weight_format", None) == "q4_k_q5_k"


def convert_qwen3_5_to_gguf(model, config: ModelConfig) -> None:
    """Replace Qwen dense projections with packed GGUF HIP operators in place."""
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    def swap_linear(owner, attr: str, quant_type: int, in_features: int, out_features: int):
        old = getattr(owner, attr)
        setattr(owner, attr, GGUFLinear(in_features, out_features, quant_type, old.bias is not None))

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(
        config.vocab_size, config.hidden_size, GGML_Q8_0, embed_scale=None
    )
    for layer in inner.layers.op_list:
        if layer._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            # The GDN constructor already creates the matching qkv|z GGUF projection
            # and a dense b|a projection when config.attn_quant is ``gguf_q8``.
            assert hasattr(layer.linear_attn, "in_proj_qkvz")
            assert hasattr(layer.linear_attn, "in_proj_ba")
            swap_linear(
                layer.linear_attn, "out_proj", GGML_Q8_0,
                layer.linear_attn.value_dim, config.hidden_size,
            )
        else:
            swap_linear(
                layer.self_attn, "qkv_proj", GGML_Q8_0,
                config.hidden_size, sum(layer.self_attn._qkv_split),
            )
            swap_linear(
                layer.self_attn, "o_proj", GGML_Q8_0,
                layer.self_attn.qo_attn_dim, config.hidden_size,
            )
        shared = layer.mlp.shared_expert
        swap_linear(
            shared, "gate_up_proj", GGML_Q8_0,
            config.hidden_size, 2 * config.shared_expert_intermediate_size,
        )
        swap_linear(
            shared, "down_proj", GGML_Q8_0,
            config.shared_expert_intermediate_size, config.hidden_size,
        )
    model.lm_head = GGUFLMHead(config.vocab_size, config.hidden_size)


def _require_tp1() -> None:
    """Reject unsupported tensor parallelism before allocating unsharded GGUF banks."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen3.5 GGUF expert banks currently support TP=1 only")


def _expert_specs(config) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return host-bank shapes expressed in exact packed GGML row bytes."""
    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    return {
        "gate_up": ((experts, 2 * intermediate, row_bytes(hidden, GGML_Q4_K)), torch.uint8),
        "down": ((experts, hidden, row_bytes(intermediate, GGML_Q5_K)), torch.uint8),
    }


def _q6_down_specs(config) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """One Q6_K down bank for each exceptional Qwen GGUF layer."""
    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    return {
        "down": ((experts, hidden, row_bytes(intermediate, GGML_Q6_K)), torch.uint8),
    }


@dataclass(frozen=True)
class QwenGGUFExpertSources:
    """Primary Q4_K/Q5_K banks plus exact Q6_K down-only exceptional banks.

    ``primary`` stays shape-uniform for the existing cache.  Its Q5_K down rows
    for ``q6_layer_ids`` are deliberately unused placeholders.  ``q6_down`` has
    only the actual Q6_K layers in the same order as ``q6_layer_ids`` and feeds a
    small auxiliary cache, avoiding any conversion between the two GGML layouts.
    """

    primary: dict[str, list[torch.Tensor]]
    q6_down: list[torch.Tensor]
    q6_layer_ids: tuple[int, ...]


def load_q4_k_q5_k_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> QwenGGUFExpertSources:
    """Load byte-exact Qwen GGUF experts into per-layer host banks.

    The loader fuses separately stored `ffn_gate_exps` and `ffn_up_exps` rows
    along their output dimension, which is safe because both use the same Q4_K
    input-row geometry.  Most down rows remain Q5_K.  The explicit Q6_K late
    layers are held in a compact side list for an auxiliary cache instead of
    being coerced into the primary Q5_K bank.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1()
    layers = int(config.num_layers)
    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    gate_row_bytes = row_bytes(hidden, GGML_Q4_K)
    down_row_bytes = row_bytes(intermediate, GGML_Q5_K)
    q6_down_row_bytes = row_bytes(intermediate, GGML_Q6_K)
    q6_layer_ids = tuple(int(layer) for layer in getattr(config, "gguf_q6_down_layer_ids", ()))
    q6_index = {layer: index for index, layer in enumerate(q6_layer_ids)}
    if layer_sink is not None and q6_layer_ids:
        raise NotImplementedError(
            "Qwen GGUF FTW conversion does not yet serialize the auxiliary Q6_K down banks"
        )
    host_banks = alloc_layer_banks(_expert_specs(config), layers)
    banks = {name: [bank.tensor for bank in host_banks[name]] for name in host_banks}
    q6_host_banks = alloc_layer_banks(_q6_down_specs(config), len(q6_layer_ids))
    q6_down = [bank.tensor for bank in q6_host_banks["down"]]
    gate_seen: set[int] = set()
    up_seen: set[int] = set()
    down_seen: set[int] = set()
    completed_gate_up: set[int] = set()

    def load(sink) -> None:
        # A completed layer consists of a fused Q4_K gate/up bank and one Q5_K down bank.
        tracker = LayerCompletionTracker(2, host_banks, sink) if sink is not None else None
        for tensor in iter_gguf_tensors(model_path):
            if not tensor.name.startswith("blk."):
                continue
            parts = tensor.name.split(".")
            layer = int(parts[1])
            suffix = ".".join(parts[2:])
            if suffix == "ffn_gate_exps.weight":
                if tensor.ggml_type != GGML_Q4_K:
                    raise ValueError(f"{tensor.name} expected Q4_K, got {tensor.ggml_type}")
                banks["gate_up"][layer][:, :intermediate].copy_(
                    tensor.packed().reshape(experts, intermediate, gate_row_bytes)
                )
                gate_seen.add(layer)
            elif suffix == "ffn_up_exps.weight":
                if tensor.ggml_type != GGML_Q4_K:
                    raise ValueError(f"{tensor.name} expected Q4_K, got {tensor.ggml_type}")
                banks["gate_up"][layer][:, intermediate:].copy_(
                    tensor.packed().reshape(experts, intermediate, gate_row_bytes)
                )
                up_seen.add(layer)
            elif suffix == "ffn_down_exps.weight":
                if layer in q6_index:
                    if tensor.ggml_type != GGML_Q6_K:
                        raise ValueError(f"{tensor.name} expected Q6_K, got {tensor.ggml_type}")
                    q6_down[q6_index[layer]].copy_(
                        tensor.packed().reshape(experts, hidden, q6_down_row_bytes)
                    )
                    # The primary cache must retain one uniform Q5_K bank shape. The
                    # Q6 layers never read this placeholder because their execution
                    # uses the auxiliary Q6_K cache.
                    banks["down"][layer].zero_()
                else:
                    if tensor.ggml_type != GGML_Q5_K:
                        raise ValueError(f"{tensor.name} expected Q5_K, got {tensor.ggml_type}")
                    banks["down"][layer].copy_(
                        tensor.packed().reshape(experts, hidden, down_row_bytes)
                    )
                down_seen.add(layer)
                if tracker is not None:
                    tracker.note(layer)
            else:
                continue
            if layer in gate_seen and layer in up_seen and layer not in completed_gate_up:
                completed_gate_up.add(layer)
                if tracker is not None:
                    tracker.note(layer)

    if layer_sink is not None:
        load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            load(pins)
            for bank in q6_host_banks["down"]:
                pins.submit(bank)
    else:
        load(None)

    wanted = set(range(layers))
    assert gate_seen == wanted and up_seen == wanted and down_seen == wanted, (
        "incomplete Qwen GGUF expert tensors: "
        f"gate={sorted(wanted - gate_seen)}, up={sorted(wanted - up_seen)}, "
        f"down={sorted(wanted - down_seen)}"
    )
    return QwenGGUFExpertSources(banks, q6_down, q6_layer_ids)


def dummy_q4_k_q5_k_expert_sources(config) -> QwenGGUFExpertSources:
    """Build correctly shaped random packed banks for loader and cache tests."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    host_banks = alloc_layer_banks(_expert_specs(config), int(config.num_layers))
    banks = {name: [bank.tensor for bank in host_banks[name]] for name in host_banks}
    q6_layer_ids = tuple(int(layer) for layer in getattr(config, "gguf_q6_down_layer_ids", ()))
    q6_host_banks = alloc_layer_banks(_q6_down_specs(config), len(q6_layer_ids))
    q6_down = [bank.tensor for bank in q6_host_banks["down"]]
    for tensor in banks["gate_up"] + banks["down"]:
        tensor.random_(0, 256)
    for tensor in q6_down:
        tensor.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(host_banks)
        pin_banks(q6_host_banks)
    return QwenGGUFExpertSources(banks, q6_down, q6_layer_ids)


__all__ = [
    "iter_gguf_weights",
    "is_gguf_model",
    "convert_qwen3_5_to_gguf",
    "load_q4_k_q5_k_expert_sources",
    "dummy_q4_k_q5_k_expert_sources",
]
