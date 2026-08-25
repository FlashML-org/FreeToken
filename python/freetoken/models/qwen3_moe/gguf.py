"""Qwen3-MoE GGUF adapter: build the FreeToken ``ModelConfig`` and stream weights
from a llama.cpp ``qwen3moe`` checkpoint.

Qwen3-MoE is simpler than Qwen3.5-MoE: all layers use standard full attention (no GDN),
there is no shared expert, and norms are plain (no Gemma-style (1+w) shift). The tensors
map directly from llama.cpp's gguf-py/gguf/tensor_mapping.py without complex fusion layers.

Verified against unsloth/Qwen3-235B-A22B-GGUF (Q4_K_M) and unsloth/Qwen3-30B-A3B-GGUF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_UNQUANTIZED as GGML_UNQUANTIZED_SET,
    GGML_NAME,
    dequantize,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _kv(shim: "GgufConfigShim", key: str, default: Any = None) -> Any:
    """Read ``<arch>.<key>`` from the GGUF metadata.

    The prefix is the checkpoint's own ``general.architecture``: "qwen3moe" for the MoE
    variant. Metadata keys follow the llama.cpp convention with this prefix.
    """
    val = shim.metadata.get(f"{shim.model_type}.{key}", default)
    if val is None and default is None:
        raise ValueError(
            f"GGUF {shim.model_path}: missing required key {shim.model_type}.{key}"
        )
    return val


def _uniform_expert_types(model_path: str, num_layers: int) -> tuple[int, int] | None:
    """``(gate_up, down)`` ggml types of the routed-expert banks, or None if not uniform.

    The offload slot pool is one allocation per bank shared by every layer, and
    ``moe_vec.cuh`` addresses it as ``expert * nrows * (ncols / qk)`` with no padding
    allowance -- so a bank whose type varies by layer cannot be served. We return None
    rather than raising here because ``parse_gguf_config`` also runs for metadata-only
    inspection; ``expert_banks._gguf_banks`` is where the load actually fails, with the
    offending layers named. (llama.cpp's *_M mixes hit this.)
    """
    from .gguf_experts import gguf_expert_types

    try:
        types = gguf_expert_types(model_path, num_layers)
    except Exception:
        return None
    gate_up, down = set(types["gate_up"]), set(types["down"])
    if len(gate_up) != 1 or len(down) != 1:
        return None
    return (next(iter(gate_up)), next(iter(down)))


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    """Build ModelConfig from qwen3moe GGUF metadata.

    All layers are standard full attention. Attention groups contain a single
    FullAttentionGroupConfig spanning all layers.
    """
    num_layers = int(_kv(shim, "block_count"))

    hidden_size = int(_kv(shim, "embedding_length"))
    num_qo_heads = int(_kv(shim, "attention.head_count"))
    num_kv_heads = int(_kv(shim, "attention.head_count_kv"))
    head_dim = int(_kv(shim, "attention.key_length"))
    rms_eps = float(_kv(shim, "attention.layer_norm_rms_epsilon"))
    rope_base = float(_kv(shim, "rope.freq_base"))
    max_pos = int(_kv(shim, "context_length"))

    # Routed expert configuration.
    num_experts = int(_kv(shim, "expert_count"))
    experts_per_tok = int(_kv(shim, "expert_used_count"))
    moe_inter = int(_kv(shim, "expert_feed_forward_length"))
    dense_inter = int(_kv(shim, "feed_forward_length"))
    moe_enabled = num_experts > 0

    # Rotary embedding configuration: use rope.dimension_count if present, else head_dim.
    rotary_dim = int(_kv(shim, "rope.dimension_count", head_dim))

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=rope_base,
        scaling=None,
    )

    # All layers are full attention: single FullAttentionGroupConfig.
    groups = (
        FullAttentionGroupConfig(
            name="full",
            layer_ids=tuple(range(num_layers)),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_config=rotary,
        ),
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=shim.vocab_size,
        intermediate_size=0 if moe_enabled else dense_inter,
        hidden_act="silu",
        rms_norm_eps=rms_eps,
        tie_word_embeddings=shim.tie_word_embeddings,
        rotary_config=rotary,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_tok,
        moe_intermediate_size=moe_inter,
        norm_topk_prob=True,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=shim.model_type,
        architectures=list(shim.architectures),
        vision_config=None,
        image_token_id=None,
        attention_groups=groups,
        expert_quant="gguf" if moe_enabled else "none",
        gguf_expert_types=(
            _uniform_expert_types(shim.model_path, num_layers) if moe_enabled else None
        ),
        gguf_model_path=shim.model_path,
        weight_block_size=None,
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
    )


# --------------------------------------------------------------------------------------
# Tensor-name mapping (inverse of llama.cpp gguf-py/gguf/tensor_mapping.py for qwen3moe)
# --------------------------------------------------------------------------------------

# Per-layer 1:1 renames that need no reshaping or fusing.
_LAYER_MAP: dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
}

# Suffixes that are PARTS of a merged projection: never renamed 1:1, always combined by
# iter_gguf_weights into the merged buffer the model actually declares.
_MERGED_PARTS: frozenset[str] = frozenset({
    "attn_q.weight", "attn_k.weight", "attn_v.weight",
})

# Routed-expert stacks: [num_experts, out, in] packed blocks, handled by the offload
# expert-bank loader rather than yielded as ordinary parameters.
_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)

_GLOBAL_MAP: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def gguf_name_to_freetoken(name: str, num_layers: int) -> str | None:
    """Map one llama.cpp tensor name to its FreeToken parameter name.

    Returns ``None`` for the routed-expert stacks (read directly by the expert-bank
    loader) and the parts of a merged projection (combined by :func:`iter_gguf_weights`).
    """
    if name in _GLOBAL_MAP:
        return _GLOBAL_MAP[name]
    if not name.startswith("blk."):
        return None
    _, idx, suffix = name.split(".", 2)
    layer = int(idx)
    if layer >= num_layers:
        return None  # out-of-bounds (should not happen in qwen3moe)
    if suffix in _EXPERT_SUFFIXES:
        return None
    if suffix in _MERGED_PARTS:
        return None  # fused by iter_gguf_weights into the merged buffer
    mapped = _LAYER_MAP.get(suffix)
    if mapped is None:
        return None
    return f"model.layers.{layer}.{mapped}"


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken qwen3moe module params.
# --------------------------------------------------------------------------------------


def _scan_quant_types(model_path: str) -> dict[tuple[int, str], int]:
    """Scan GGUF tensor table once and return {(layer, suffix): ggml_type}.

    This allows us to detect which groups are mixed-quant without hardcoding.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    quant_types = {}
    for t in iter_gguf_tensors(model_path):
        if not t.name.startswith("blk."):
            # Globals (token_embd.weight, output.weight, output_norm.weight) keyed under
            # layer -1 so the swap can size the embedding and lm_head from the file too.
            quant_types[(-1, t.name)] = t.ggml_type
            continue
        _, idx, suffix = t.name.split(".", 2)
        layer = int(idx)
        quant_types[(layer, suffix)] = t.ggml_type
    return quant_types


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense bf16 tensor of its torch shape.

    Unlike Qwen3.5-MoE's Gemma-style norms which apply a (1+w) shift at load time, qwen3moe
    norms are plain. Dequantize as-is without adding 1.0.
    """
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _to_f32(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense float32 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
    return flat.reshape(t.shape)


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 with a clear error."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"qwen3moe GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert qwen3moe parameter.

    Quantized projections (attention qkv/o) stay in their native packed block layout and
    are yielded as ``.qweight`` or ``.qweight_<i>`` for mixed-quant groups; norms and the
    router gate dequantize to bf16. q/k/v are fused by concatenating packed rows.

    Routed experts are served from the offload cache (asserts the offload contract).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    # Only the MoE variant keeps its routed experts out of this iterator; they come from
    # the offload cache instead.
    config_moe = int(_kv(cached_load_hf_config(model_path), "expert_count", 0)) > 0
    assert not (config_moe and include_moe_experts), (
        "qwen3moe GGUF keeps its routed experts in the offload cache; they are loaded by "
        "the expert-bank loader, not by iter_gguf_weights."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    # Parse config to get the number of layers and other geometry.
    config = parse_gguf_config(cached_load_hf_config(model_path))

    # Scan quant types once to determine which fusion groups are mixed-quant.
    quant_map = _scan_quant_types(model_path)

    # Per-layer fusion buffer for qkv: layer -> {slot: packed[out, row_bytes]}.
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])

    for t in iter_gguf_tensors(model_path):
        name = t.name
        layer = layer_of(name) if name.startswith("blk.") else None

        # Global tensors
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            if not config.tie_word_embeddings:
                yield "lm_head.qweight", t.packed()
            continue
        if not name.startswith("blk."):
            continue

        # Skip out-of-bounds layers (should not happen in qwen3moe).
        if layer >= config.num_layers:
            continue

        # Skip routed-expert stacks (offload banks).
        if any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            continue

        suffix = name.split(".", 2)[2]  # after "blk.N."
        base = f"model.layers.{layer}"

        # Scalar/norm tensors: dequant to bf16.
        if suffix == "attn_norm.weight":
            yield f"{base}.input_layernorm.weight", _to_bf16(t)
            continue
        if suffix == "ffn_norm.weight":
            yield f"{base}.post_attention_layernorm.weight", _to_bf16(t)
            continue
        if suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _to_bf16(t)
            continue
        if suffix == "attn_q_norm.weight":
            yield f"{base}.self_attn.q_norm.weight", _to_bf16(t)
            continue
        if suffix == "attn_k_norm.weight":
            yield f"{base}.self_attn.k_norm.weight", _to_bf16(t)
            continue

        # Quantized projections: keep packed; fuse per layer.
        # All layers are full-attention: fuse q, k, v into qkv_proj.
        if suffix == "attn_q.weight":
            qkv_buf.setdefault(layer, {})["q"] = t.packed()
        elif suffix == "attn_k.weight":
            qkv_buf.setdefault(layer, {})["k"] = t.packed()
        elif suffix == "attn_v.weight":
            qkv_buf.setdefault(layer, {})["v"] = t.packed()
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
        else:
            continue  # unmapped suffix

        # Emit fused qkv once all three parts are present.
        slots = qkv_buf.get(layer)
        if slots is not None and "q" in slots and "k" in slots and "v" in slots:
            # Determine if this is a mixed-quant group.
            types = [
                quant_map.get((layer, "attn_q.weight")),
                quant_map.get((layer, "attn_k.weight")),
                quant_map.get((layer, "attn_v.weight")),
            ]
            if len(set(types)) == 1:
                # Uniform quant: fuse via torch.cat along dim 0.
                yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                    [slots["q"], slots["k"], slots["v"]], dim=0
                )
            else:
                # Mixed quant: emit GGUFMergedLinear format.
                yield f"{base}.self_attn.qkv_proj.qweight_0", slots["q"]
                yield f"{base}.self_attn.qkv_proj.qweight_1", slots["k"]
                yield f"{base}.self_attn.qkv_proj.qweight_2", slots["v"]
            del qkv_buf[layer]

    # Verify no fusion buffers are incomplete.
    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when this config came from a GGUF checkpoint (native block-quant path).

    Keys on gguf_model_path rather than expert_quant: ensures the op swap runs.
    """
    return getattr(config, "gguf_model_path", None) is not None


def convert_qwen3moe_to_gguf(model, config: ModelConfig, *, model_path: str) -> None:
    """In place: replace qwen3moe's dense projections + embedding with native GGUF ops.

    Quantized in the checkpoint -> swapped: attention qkv/o (mixed-quant). Left as dense
    bf16 (F32 in the GGUF): all RMSNorms, the router gate, and the routed experts
    (served from the offload cache).

    The per-layer quant types are read from the GGUF file, not hardcoded, to support
    different quant levels.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, gguf_merged_or_plain

    # Scan quant types to drive layer swaps.
    quant_map = _scan_quant_types(model_path)

    # Split widths for qkv come from the config.
    _qkv_split = [
        config.num_qo_heads * config.head_dim,
        config.num_kv_heads * config.head_dim,
        config.num_kv_heads * config.head_dim,
    ]

    def qt(layer: int, suffix: str) -> int:
        """The ggml type of one tensor, straight from the file.

        No default: a guessed type silently allocates a wrong-sized packed buffer.
        """
        key = (layer, suffix)
        if key not in quant_map:
            raise ValueError(
                f"GGUF {model_path}: expected tensor "
                f"{suffix if layer < 0 else f'blk.{layer}.{suffix}'} is absent, so its quant "
                f"type cannot be read; this checkpoint does not match the qwen3moe layout "
                f"this adapter expects"
            )
        return quant_map[key]

    def swap_linear(owner, attr, quant_type: int):
        """Replace a dense Linear with the GGUFLinear its packed weight will land in."""
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(in_features, out_features, quant_type, has_bias=lin.bias is not None),
        )

    inner = model.model
    embed = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=qt(-1, "token_embd.weight"),
    )
    inner.embed_tokens = embed

    for layer_idx, layer in enumerate(inner.layers.op_list):
        # qkv_proj: q | k | v.
        layer.self_attn.qkv_proj = gguf_merged_or_plain(
            config.hidden_size,
            _qkv_split,
            [
                qt(layer_idx, "attn_q.weight"),
                qt(layer_idx, "attn_k.weight"),
                qt(layer_idx, "attn_v.weight"),
            ],
            has_bias=False,
        )
        swap_linear(layer.self_attn, "o_proj", qt(layer_idx, "attn_output.weight"))

    if config.tie_word_embeddings:
        from freetoken.models.gemma4.gguf import GGUFTiedLMHead

        model.lm_head = GGUFTiedLMHead(embed, qt(-1, "token_embd.weight"))
    else:
        # NOT swap_linear: a plain GGUFLinear would compute logits for every prefill
        # position, and [tokens, vocab] is the largest tensor in the model. See GGUFLMHead.
        from freetoken.layers.gguf import GGUFLMHead

        head = model.lm_head
        out_features, in_features = head.weight.shape
        model.lm_head = GGUFLMHead(
            in_features, out_features, qt(-1, "output.weight"),
            has_bias=head.bias is not None,
        )


__all__ = [
    "parse_gguf_config",
    "gguf_name_to_freetoken",
    "iter_gguf_weights",
    "convert_qwen3moe_to_gguf",
    "is_gguf_model",
    "_MERGED_PARTS",
    "_EXPERT_SUFFIXES",
]
