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
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
_GDN_BA_SUFFIXES = {"ssm_alpha.weight": "a", "ssm_beta.weight": "b"}


def _ssm_a_to_a_log(ssm_a: torch.Tensor) -> torch.Tensor:
    """Recover HF ``A_log`` from llama.cpp's precomputed negative decay.

    The GGUF Qwen3.5 exporter writes ``ssm_a = -exp(A_log)`` because llama.cpp
    multiplies that value directly by the softplus alpha gate.  FreeToken's Gated
    DeltaNet instead owns the equivalent ``-A_log.exp()`` expression.  Loading the
    GGUF value as ``A_log`` would exponentiate it a second time and destabilize every
    linear-attention layer, so invert the exporter transformation exactly here.
    """
    value = ssm_a.to(torch.float32)
    if not torch.isfinite(value).all() or not torch.all(value < 0):
        raise ValueError("Qwen GGUF ssm_a must contain finite negative -exp(A_log) values")
    return torch.log(-value)


def _restore_gdn_value_head_order(value: torch.Tensor, num_key_heads: int) -> torch.Tensor:
    """Convert llama.cpp's grouped GDN value-head order to Qwen's interleaved order.

    Qwen3.5 uses more value than key heads.  GGUF places all value heads belonging
    to the first position of each key-head group before the second position, while
    the Hugging Face checkpoint and FreeToken's Gated DeltaNet use consecutive
    per-key-head values.  This applies both to one scalar per value head (``ssm_a``
    and ``ssm_dt``) and to matrices whose output axis is the value-head axis
    (``ssm_alpha`` and ``ssm_beta``).
    """
    if value.ndim < 1:
        raise ValueError("Qwen GDN value-head tensor must have at least one dimension")
    num_value_heads = value.shape[0]
    if num_key_heads <= 0 or num_value_heads % num_key_heads:
        raise ValueError(
            "Qwen GDN value-head tensor is incompatible with the GGUF key-head count: "
            f"{tuple(value.shape)} vs {num_key_heads}"
        )
    head_ratio = num_value_heads // num_key_heads
    if head_ratio == 1:
        return value
    # [ratio, key_head, ...] in GGUF becomes [key_head, ratio, ...], then a
    # contiguous leading output axis matching the HF/FreeToken projection order.
    return value.reshape(head_ratio, num_key_heads, *value.shape[1:]).transpose(0, 1).reshape_as(value)


def _restore_gdn_value_head_rows(
    value: torch.Tensor,
    num_key_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Restore grouped GGUF rows whose leading axis contains complete value heads.

    A GDN projection can contain a prefix of key-head rows followed by its value
    rows, such as the Q|K|V and depthwise-convolution projections.  Only the
    value suffix needs the llama.cpp-to-HF permutation.  ``head_dim`` is the
    number of consecutive rows occupied by one value head, so this routine also
    handles projections such as ``z`` where each head spans 128 output rows.
    """
    if value.ndim < 1 or head_dim <= 0 or value.shape[0] % head_dim:
        raise ValueError(
            "Qwen GDN value-head rows require a positive whole-head leading axis: "
            f"{tuple(value.shape)} with head_dim={head_dim}"
        )
    num_value_heads = value.shape[0] // head_dim
    ordered = _restore_gdn_value_head_order(
        value.reshape(num_value_heads, head_dim, *value.shape[1:]), num_key_heads
    )
    return ordered.reshape_as(value)


def _restore_gdn_value_head_input_blocks(
    packed: torch.Tensor,
    num_key_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Restore GDN value-head order along a Q8_0 packed projection input axis.

    ``ssm_out`` consumes all value heads as its input.  Q8_0 stores independent
    32-element blocks along each output row, and Qwen's 128-element value heads
    therefore occupy four complete byte blocks.  Reordering those blocks is exact:
    it neither dequantizes weights nor changes their Q8 scales or integers.
    """
    if packed.ndim != 2:
        raise ValueError(f"Qwen GDN packed output projection must be rank 2, got {tuple(packed.shape)}")
    bytes_per_head = row_bytes(head_dim, GGML_Q8_0)
    if head_dim <= 0 or packed.shape[1] % bytes_per_head:
        raise ValueError(
            "Qwen GDN packed output projection does not contain complete value-head blocks: "
            f"{tuple(packed.shape)} with head_dim={head_dim}"
        )
    num_value_heads = packed.shape[1] // bytes_per_head
    grouped = packed.reshape(packed.shape[0], num_value_heads, bytes_per_head)
    # The generic helper operates on the leading head axis.  Transpose the packed
    # view so the same explicit permutation is applied to every output row.
    return _restore_gdn_value_head_order(grouped.transpose(0, 1), num_key_heads).transpose(0, 1).reshape_as(packed)


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
    from freetoken.models.gguf.reader import load_gguf_metadata

    assert not include_moe_experts, "Qwen GGUF routed experts are supplied by the offload cache"
    assert include_non_moe
    _require_weight_tp1()

    metadata = load_gguf_metadata(model_path)
    arch = metadata.get("general.architecture")
    prefix = "qwen35moe" if arch == "qwen35moe" else "qwen35"
    dense_model = prefix == "qwen35"
    gdn_num_key_heads = int(metadata[f"{prefix}.ssm.group_count"])
    gdn_num_value_heads = int(metadata[f"{prefix}.ssm.time_step_rank"])
    gdn_inner_size = int(metadata[f"{prefix}.ssm.inner_size"])
    if gdn_num_value_heads <= 0 or gdn_inner_size % gdn_num_value_heads:
        raise ValueError(
            "Qwen GGUF GDN metadata has an invalid value-head geometry: "
            f"inner_size={gdn_inner_size}, time_step_rank={gdn_num_value_heads}"
        )
    gdn_value_head_dim = gdn_inner_size // gdn_num_value_heads

    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    gdn_buf: dict[int, dict[str, torch.Tensor]] = {}
    shared_buf: dict[int, dict[str, torch.Tensor]] = {}
    dense_buf: dict[int, dict[str, torch.Tensor]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            if t.ggml_type not in (GGML_Q8_0, GGML_Q4_K):
                raise ValueError(f"{name} expected Q8_0 or Q4_K, got {t.ggml_type}")
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            if t.ggml_type != GGML_Q6_K:
                raise ValueError(f"{name} expected Q6_K, got {t.ggml_type}")
            yield "lm_head.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            # Unlike Gemma GGUF checkpoints, Qwen stores the final RMSNorm scale
            # directly.  Adding one here would apply the Gemma delta convention
            # to an already complete Qwen weight and corrupt every output logit.
            yield "model.norm.weight", _to_bf16(t)
            continue
        if not name.startswith("blk."):
            continue

        parts = name.split(".")
        layer = int(parts[1])
        suffix = ".".join(parts[2:])
        base = f"model.layers.{layer}"
        if suffix in _EXPERT_SUFFIXES:
            continue
        if dense_model and suffix in ("ffn_gate.weight", "ffn_up.weight"):
            dense_buf.setdefault(layer, {})[suffix.removeprefix("ffn_").removesuffix(".weight")] = t.packed()
        elif dense_model and suffix == "ffn_down.weight":
            yield f"{base}.mlp.down_proj.qweight", t.packed()
        elif suffix in _GDN_BA_SUFFIXES:
            # The split GGUF path keeps qkv|z packed Q8_0, while recurrence b|a
            # remains a conventional dense fused projection.  The runtime order is
            # explicitly b then a, matching Qwen3_5GatedDeltaNet._in_proj_split.
            gdn_buf.setdefault(layer, {})[_GDN_BA_SUFFIXES[suffix]] = _restore_gdn_value_head_order(
                _to_bf16(t), gdn_num_key_heads
            )
            slots = gdn_buf[layer]
            if all(key in slots for key in ("b", "a")):
                yield f"{base}.linear_attn.in_proj_ba.weight", torch.cat(
                    [slots.pop("b"), slots.pop("a")], dim=0
                )
                if not slots:
                    del gdn_buf[layer]
            continue
        if suffix == "ssm_a":
            # llama.cpp serializes the already-exponentiated negative coefficient;
            # FreeToken stores A_log and evaluates -exp(A_log) at runtime.
            yield f"{base}.linear_attn.A_log", _ssm_a_to_a_log(
                _restore_gdn_value_head_order(_to_bf16(t), gdn_num_key_heads)
            )
            continue
        if suffix in _SCALAR_MAP:
            tensor = _to_bf16(t)
            rel = _SCALAR_MAP[suffix]
            if suffix == "ssm_dt.bias":
                tensor = _restore_gdn_value_head_order(tensor, gdn_num_key_heads)
            if suffix == "ssm_conv1d.weight":
                # GGUF stores depthwise filters as [channels, kernel]; FreeToken's
                # causal-convolution holder uses the PyTorch depthwise layout
                # [channels, 1, kernel].
                # Its Q|K prefix retains key-head order, while its V suffix uses
                # llama.cpp's grouped value-head order and must be made consistent
                # with the restored scalar recurrence terms.
                gdn_key_dim = gdn_num_key_heads * gdn_value_head_dim
                gdn_value_dim = gdn_num_value_heads * gdn_value_head_dim
                qk_prefix = tensor[: 2 * gdn_key_dim]
                value_rows = _restore_gdn_value_head_rows(
                    tensor[2 * gdn_key_dim : 2 * gdn_key_dim + gdn_value_dim],
                    gdn_num_key_heads,
                    gdn_value_head_dim,
                )
                tensor = torch.cat((qk_prefix, value_rows), dim=0).unsqueeze(1)
            elif suffix == "ffn_gate_inp_shexp.weight":
                # The single shared-expert gate is stored as a vector in GGUF but
                # executes as a one-row replicated linear projection.
                tensor = tensor.unsqueeze(0)
            # Qwen GGUF stores all of these RMSNorm vectors as direct scales.
            # ``GemmaRMSNorm`` in this runtime applies the raw stored tensor; the
            # safetensors loader performs a separate +1 bake only because HF Qwen
            # checkpoints carry delta-from-unity weights.  GGUF must not adjust it.
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
            # The Q|K prefix is keyed by the 16 GDN key heads.  The V suffix is
            # keyed by the 32 value heads and is grouped by llama.cpp in GGUF.
            packed = t.packed()
            gdn_key_dim = gdn_num_key_heads * gdn_value_head_dim
            qk_rows = packed[: 2 * gdn_key_dim]
            value_rows = _restore_gdn_value_head_rows(
                packed[2 * gdn_key_dim :], gdn_num_key_heads, gdn_value_head_dim
            )
            gdn_buf.setdefault(layer, {})["qkv"] = torch.cat((qk_rows, value_rows), dim=0)
        elif suffix == "attn_gate.weight":
            gdn_buf.setdefault(layer, {})["z"] = _restore_gdn_value_head_rows(
                t.packed(), gdn_num_key_heads, gdn_value_head_dim
            )
        elif suffix == "ssm_out.weight":
            yield f"{base}.linear_attn.out_proj.qweight", _restore_gdn_value_head_input_blocks(
                t.packed(), gdn_num_key_heads, gdn_value_head_dim
            )
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
        slots = dense_buf.get(layer)
        if slots is not None and all(key in slots for key in ("gate", "up")):
            yield f"{base}.mlp.gate_up_proj.qweight", torch.cat(
                [slots["gate"], slots["up"]], dim=0
            )
            del dense_buf[layer]

    assert not qkv_buf, f"incomplete Qwen attention QKV groups: {sorted(qkv_buf)}"
    assert not gdn_buf, f"incomplete Qwen GDN qkv/z groups: {sorted(gdn_buf)}"
    assert not shared_buf, f"incomplete Qwen shared gate/up groups: {sorted(shared_buf)}"
    assert not dense_buf, f"incomplete Qwen dense gate/up groups: {sorted(dense_buf)}"


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
    return getattr(config, "moe_weight_format", None) in {"q4_k_q5_k", "qwen35_dense"}


def convert_qwen3_5_to_gguf(model, config: ModelConfig) -> None:
    """Replace Qwen dense projections with packed GGUF HIP operators in place."""
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    dense_model = not config.moe_enabled
    embed_quant = GGML_Q4_K if dense_model else GGML_Q8_0
    full_output_quant = GGML_Q6_K if dense_model else GGML_Q8_0

    def swap_linear(owner, attr: str, quant_type: int, in_features: int, out_features: int):
        old = getattr(owner, attr)
        setattr(owner, attr, GGUFLinear(in_features, out_features, quant_type, old.bias is not None))

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(
        config.vocab_size, config.hidden_size, embed_quant, embed_scale=None
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
                layer.self_attn, "o_proj", full_output_quant,
                layer.self_attn.qo_attn_dim, config.hidden_size,
            )
        if config.moe_enabled:
            owner = layer.mlp.shared_expert
            intermediate = config.shared_expert_intermediate_size
            mlp_quant = GGML_Q8_0
        else:
            owner = layer.mlp
            intermediate = config.intermediate_size
            mlp_quant = GGML_Q4_K
        swap_linear(owner, "gate_up_proj", mlp_quant, config.hidden_size, 2 * intermediate)
        swap_linear(owner, "down_proj", mlp_quant, intermediate, config.hidden_size)
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
