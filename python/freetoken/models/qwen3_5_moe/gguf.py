"""Qwen3.5-MoE GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata and
map GGUF tensors to the model's state dict.

The qwen35moe GGUF arch is a hybrid GatedDeltaNet (linear-attention SSM) + full-attention
MoE (40 layers, every 4th full; 256 routed experts + shared expert). The GGUF geometry
matches the HF qwen3_5_moe model, so ``parse_gguf_config`` produces the *same*
``ModelConfig`` as ``qwen3_5_moe.config.parse_config`` -- only the source is GGUF KV
metadata. ``expert_quant``/``attn_quant``/``dense_quant`` are set to ``"gguf"`` so
``convert_qwen35moe_to_gguf`` can detect the native-quant checkpoint and swap the dense
projections for native GGUF-quant ops.

Quantized projections stay in their source-native packed block layout and are yielded as
``.qweight`` (uint8); tiny F32 tensors (norms, router, GDN b/a) dequantize to bf16; GDN
conv/A_log/dt_bias stay fp32. Source tensor types may vary by projection and layer, so
merged projections retain per-slice packed buffers. Routed experts retain Q4_K/Q5_K/Q6_K/
Q8_0 rows. Resident mode uses native per-layer buffers; GPU offload uses
``load_gguf_expert_sources_native`` with padded strides and explicit type dispatch; CPU
decode converts one source tensor at a time to the executor's Q4_0 contract. Legacy
converter/offload banks remain available with uniform Q8_0 down rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_F32,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    quantize_q4_0,
    quantize_q8_0,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"qwen3.5-moe GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    down_quant_types = _gguf_down_quant_types(shim.model_path)
    gate_up_quant_types = _gguf_gate_up_quant_types(shim.model_path)

    def g(key: str):
        val = m.get(f"qwen35moe.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key qwen35moe.{key}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    num_kv_heads = int(g("attention.head_count_kv"))
    full_head_dim = int(g("attention.key_length"))
    max_pos = int(g("context_length"))
    interval = int(g("full_attention_interval"))  # every Nth (1-indexed) layer is full

    full_ids = set(i for i in range(num_layers) if (i + 1) % interval == 0)
    # Qwen3.5 GGUF exports carry one terminal full-attention block beyond the
    # interval schedule. Source tensor presence is authoritative for this exception.
    full_ids.update(_gguf_full_attention_layers(shim.model_path))
    full_ids = tuple(sorted(full_ids))
    full_id_set = set(full_ids)
    linear_ids = tuple(i for i in range(num_layers) if i not in full_id_set)

    full_rotary = RotaryConfig(
        head_dim=full_head_dim,
        rotary_dim=int(g("rope.dimension_count")),
        max_position=max_pos,
        base=float(g("rope.freq_base")),
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=full_head_dim,
        rotary_config=full_rotary,
    )
    # GDN (SSM) dims. conv_dim = 2*key_dim + value_dim (attn_qkv); value_dim = nv*vhead_dim.
    key_head_dim = int(g("ssm.state_size"))
    value_head_dim = int(g("ssm.state_size"))
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=int(g("ssm.group_count")),
        num_value_heads=int(g("ssm.time_step_rank")),
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_dim=int(g("ssm.conv_kernel")),
        output_gate=True,
    )
    groups = tuple(sorted((full_group, linear_group), key=lambda grp: grp.layer_ids[0] or 1 << 30))

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=full_head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,  # routed MoE; no dense MLP
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=full_rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        model_type="qwen3_5_moe",
        architectures=list(shim.architectures),
        moe_enabled=True,
        use_qk_norm=True,
        attention_groups=groups,
        expert_quant="gguf",
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
        moe_weight_format="gguf",
        gguf_down_quant_types=down_quant_types,
        gguf_gate_up_quant_types=gate_up_quant_types,
        gguf_tensor_types=_gguf_tensor_types(shim.model_path),
    )


def is_gguf_model(config: ModelConfig) -> bool:
    return getattr(config, "moe_weight_format", None) == "gguf"


def _gguf_down_quant_types(model_path: str) -> tuple[int, ...]:
    """Read routed down types from tensor headers without touching tensor payloads."""
    from freetoken.models.gguf.reader import _reader, gguf_shard_paths

    try:
        shards = gguf_shard_paths(model_path)
    except (OSError, ValueError, ImportError):
        return ()
    types: dict[int, int] = {}
    for shard in shards:
        for tensor in _reader(shard).tensors:
            if not tensor.name.startswith("blk.") or not tensor.name.endswith("ffn_down_exps.weight"):
                continue
            layer = int(tensor.name.split(".")[1])
            quant_type = int(tensor.tensor_type)
            if quant_type not in (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0):
                raise ValueError(
                    f"{tensor.name}: resident GGUF supports Q4_K/Q5_K/Q6_K/Q8_0 down experts, "
                    f"got ggml type {quant_type}"
                )
            prior = types.setdefault(layer, quant_type)
            if prior != quant_type:
                raise ValueError(f"{tensor.name}: expert down quant type changed within layer")
    if not types:
        return ()
    last = max(types)
    if set(types) != set(range(last + 1)):
        raise ValueError(f"GGUF expert down types missing layers: expected 0..{last}, got {sorted(types)}")
    return tuple(types[i] for i in range(last + 1))


def _gguf_gate_up_quant_types(model_path: str) -> tuple[int, ...]:
    """Read routed gate/up types; GGUF exports may change format at terminal blocks."""
    from freetoken.models.gguf.reader import _reader, gguf_shard_paths

    gate: dict[int, int] = {}
    up: dict[int, int] = {}
    allowed = (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0)
    for shard in gguf_shard_paths(model_path):
        for tensor in _reader(shard).tensors:
            if not tensor.name.startswith("blk."):
                continue
            if tensor.name.endswith("ffn_gate_exps.weight"):
                target = gate
            elif tensor.name.endswith("ffn_up_exps.weight"):
                target = up
            else:
                continue
            layer = int(tensor.name.split(".")[1])
            quant_type = int(tensor.tensor_type)
            if quant_type not in allowed:
                raise ValueError(
                    f"{tensor.name}: unsupported GGUF gate/up type {quant_type}; "
                    "expected Q4_K/Q5_K/Q6_K/Q8_0"
                )
            prior = target.setdefault(layer, quant_type)
            if prior != quant_type:
                raise ValueError(f"{tensor.name}: expert gate/up type changed within layer")
    if gate != up:
        raise ValueError(
            f"GGUF expert gate/up quant types disagree: gate={sorted(gate.items())}, "
            f"up={sorted(up.items())}"
        )
    if not gate:
        return ()
    last = max(gate)
    if set(gate) != set(range(last + 1)):
        raise ValueError(f"GGUF expert gate/up types missing layers: expected 0..{last}")
    return tuple(gate[i] for i in range(last + 1))


def _gguf_tensor_types(model_path: str) -> tuple[tuple[str, int], ...]:
    """Capture native tensor types from GGUF headers without reading payload bytes."""
    from freetoken.models.gguf.reader import _reader, gguf_shard_paths

    result: list[tuple[str, int]] = []
    for shard in gguf_shard_paths(model_path):
        result.extend((t.name, int(t.tensor_type)) for t in _reader(shard).tensors)
    return tuple(result)


def _gguf_full_attention_layers(model_path: str) -> tuple[int, ...]:
    """Find GGUF blocks carrying standalone full-attention q/k/v tensors."""
    from freetoken.models.gguf.reader import _reader, gguf_shard_paths

    layers: set[int] = set()
    for shard in gguf_shard_paths(model_path):
        layers.update(
            int(t.name.split(".")[1])
            for t in _reader(shard).tensors
            if t.name.startswith("blk.") and t.name.endswith("attn_q.weight")
        )
    return tuple(sorted(layers))


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 Linear -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def convert_qwen35moe_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace the dense projections + embedding with native GGUF ops.

    Quantized in the checkpoint -> swapped to ``GGUFLinear``/``GGUFEmbedding``: the token
    embedding (Q8_0) and the (untied) lm_head (Q6_K), full-attention qkv/o (Q8_0), GDN
    in_proj_qkvz + out_proj (Q8_0; in_proj_ba stays dense bf16), and the shared-expert
    gate_up/down (Q8_0). Left dense bf16/fp32 (F32 in the GGUF): the norms, the two
    routers, and the GDN conv1d/A_log/dt_bias. Routed experts are native-resident when
    selected, or remain in host banks for offload.
    """
    from freetoken.layers.gguf import GGUFMergedLinear, GGUFEmbedding, GGUFLinear
    from freetoken.models.gguf.dequant import GGML_Q6_K, GGML_Q8_0

    inner = model.model
    source_types = dict(getattr(config, "gguf_tensor_types", ()))

    def qtype(name: str, fallback: int) -> int:
        return int(source_types.get(name, fallback))

    inner.embed_tokens = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=qtype("token_embd.weight", GGML_Q8_0),
    )
    model.lm_head = GGUFLinear(
        config.hidden_size, config.vocab_size,
        qtype("output.weight", GGML_Q6_K), has_bias=False,
    )
    shared_I = config.shared_expert_intermediate_size

    for layer in inner.layers.op_list:
        if layer._is_linear:
            g = layer.linear_attn
            g.in_proj_qkvz = GGUFMergedLinear(
                config.hidden_size, [g.conv_dim, g.value_dim],
                [
                    qtype(f"blk.{layer._layer_id}.attn_qkv.weight", GGML_Q8_0),
                    qtype(f"blk.{layer._layer_id}.attn_gate.weight", GGML_Q8_0),
                ], has_bias=False,
            )
            g.out_proj = GGUFLinear(
                g.value_dim, config.hidden_size,
                # K-quant blocks span two 128-wide value heads, so packed column
                # de-interleaving is impossible without changing block boundaries.
                # The loader normalizes non-Q8 GDN output rows after de-interleaving.
                GGML_Q8_0,
                has_bias=False,
            )
        else:
            attn = layer.self_attn
            attn.qkv_proj = GGUFMergedLinear(
                config.hidden_size,
                attn._qkv_split,
                [
                    qtype(f"blk.{layer._layer_id}.attn_q.weight", GGML_Q8_0),
                    qtype(f"blk.{layer._layer_id}.attn_k.weight", GGML_Q8_0),
                    qtype(f"blk.{layer._layer_id}.attn_v.weight", GGML_Q8_0),
                ], has_bias=False,
            )
            attn.o_proj = GGUFLinear(
                attn.qo_attn_dim, config.hidden_size,
                qtype(f"blk.{layer._layer_id}.attn_output.weight", GGML_Q8_0), has_bias=False,
            )
        m = layer.mlp
        m.shared_expert.gate_up_proj = GGUFMergedLinear(
            config.hidden_size, [shared_I, shared_I],
            [
                qtype(f"blk.{layer._layer_id}.ffn_gate_shexp.weight", GGML_Q8_0),
                qtype(f"blk.{layer._layer_id}.ffn_up_shexp.weight", GGML_Q8_0),
            ], has_bias=False,
        )
        m.shared_expert.down_proj = GGUFLinear(
            shared_I, config.hidden_size,
            qtype(f"blk.{layer._layer_id}.ffn_down_shexp.weight", GGML_Q8_0), has_bias=False,
        )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken qwen3_5_moe module params.
# --------------------------------------------------------------------------------------

# Gemma-style (1+w) norms get +1 baked in; the GDN gated norm / router / shared-gate are
# standard (no +1).
_GEMMA_NORMS = {
    "attn_norm.weight",
    "post_attention_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
}

_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def _to_bf16(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _to_fp32(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
    return flat.reshape(t.shape)


def _name_to_key(suffix: str) -> tuple[str, bool]:
    """gguf layer suffix -> (module-relative key, is_gemma_norm)."""
    if suffix == "attn_norm.weight":
        return "input_layernorm.weight", True
    if suffix == "post_attention_norm.weight":
        return "post_attention_layernorm.weight", True
    if suffix == "attn_q_norm.weight":
        return "self_attn.q_norm.weight", True
    if suffix == "attn_k_norm.weight":
        return "self_attn.k_norm.weight", True
    if suffix == "ssm_norm.weight":
        return "linear_attn.norm.weight", False
    if suffix == "ffn_gate_inp.weight":
        return "mlp.gate.weight", False
    if suffix == "ffn_gate_inp_shexp.weight":
        return "mlp.shared_expert_gate.weight", False
    if suffix == "ssm_conv1d.weight":
        return "linear_attn.conv1d.weight", False  # fp32; reshaped below
    if suffix == "ssm_a":
        return "linear_attn.A_log", False  # fp32
    if suffix == "ssm_dt.bias":
        return "linear_attn.dt_bias", False  # fp32
    return None, False


# --------------------------------------------------------------------------------------
# GDN value-head de-interleaving.
#
# llama.cpp stores the GDN *value* projections with the ``mrope_interleaved`` head order:
# the 32 value heads are split into [even heads, odd heads] (head h lives at GGUF position
# ``(h // 2) + (h % 2) * (num_vheads // 2)``). Full-attention heads are NOT interleaved.
# FreeToken uses the HF contiguous head order, so the value-dim projection weights must be
# de-interleaved when loading. Affected: GDN ``in_proj_qkvz`` (the v and z rows), ``out_proj``
# (the value input columns), and ``in_proj_ba`` (the per-head b/a rows).
# --------------------------------------------------------------------------------------


def _gdn_head_perm(num_vheads: int) -> list[int]:
    """GGUF value-head index of each HF head h (``result[h] = old[perm[h]]``)."""
    half = num_vheads // 2
    return [(h // 2) + (h % 2) * half for h in range(num_vheads)]


def _deint_q8_rows(
    packed: torch.Tensor, num_vheads: int, rows_per_head: int
) -> torch.Tensor:
    """De-interleave value heads along the packed rows (output dim)."""
    m = packed.reshape(num_vheads, rows_per_head, -1)
    return m[_gdn_head_perm(num_vheads)].reshape(packed.shape)


def _deint_q8_cols(
    packed: torch.Tensor, num_vheads: int, blocks_per_head: int, block_bytes: int = 34
) -> torch.Tensor:
    """De-interleave value heads along the packed columns (input dim, per Q8_0 row)."""
    m = packed.reshape(packed.shape[0], num_vheads, blocks_per_head * block_bytes)
    return m[:, _gdn_head_perm(num_vheads), :].reshape(packed.shape)


def _deint_dense_cols(
    dense: torch.Tensor, num_vheads: int, head_dim: int
) -> torch.Tensor:
    """De-interleave value heads along columns after dequantization."""
    m = dense.reshape(dense.shape[0], num_vheads, head_dim)
    return m[:, _gdn_head_perm(num_vheads), :].reshape(dense.shape)


def _deint_dense_rows(w: torch.Tensor, num_vheads: int) -> torch.Tensor:
    """De-interleave value heads along the leading (head) dim of a dense tensor."""
    m = w.reshape(num_vheads, -1)
    return m[_gdn_head_perm(num_vheads)].reshape(w.shape)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield packed state tensors for Qwen3.5-MoE.

    Quantized projections stay packed and are yielded as ``.qweight`` (uint8), or as
    ``.qweight_N`` for mixed-type merged projections; the F32 norms/router/GDN b,a
    dequantize to bf16; conv1d/A_log/dt_bias stay fp32. Routed experts are skipped for
    offload and yielded as native packed banks for resident mode.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(cached_load_hf_config(model_path))
    full_layers = set(
        next(g.layer_ids for g in config.attention_groups
             if isinstance(g, FullAttentionGroupConfig))
    )
    # GDN value-head geometry (for mrope_interleaved de-interleave).
    gdn = next(g for g in config.attention_groups
               if isinstance(g, LinearGatedDeltaGroupConfig))
    n_vheads = gdn.num_value_heads
    vhead_dim = gdn.value_head_dim
    key_dim = gdn.num_key_heads * gdn.key_head_dim
    q8_blocks_per_head = vhead_dim // 32  # Q8_0 block = 32

    qkv_buf: dict[int, dict[str, tuple[torch.Tensor, int]]] = {}
    qkvz_buf: dict[int, dict[str, tuple[torch.Tensor, int]]] = {}
    ba_buf: dict[int, dict[str, torch.Tensor]] = {}
    shexp_buf: dict[int, dict[str, tuple[torch.Tensor, int]]] = {}
    expert_buf: dict[int, dict[str, tuple[torch.Tensor, int]]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            # GGUF stores Gemma norms as the full (1+w) scale; GemmaRMSNorm multiplies by
            # it directly (no +1, unlike the HF safetensors form which stores scale-1).
            yield "model.norm.weight", _to_bf16(t)
            continue
        if not name.startswith("blk."):
            continue
        expert_suffix = next((sfx for sfx in _EXPERT_SUFFIXES if name.endswith(sfx)), None)
        if expert_suffix is not None:
            if not include_moe_experts:
                continue  # routed experts -> offload banks
            layer = int(name.split(".")[1])
            kind = {
                "ffn_gate_exps.weight": "gate",
                "ffn_up_exps.weight": "up",
                "ffn_down_exps.weight": "down",
            }[expert_suffix]
            gate_types = getattr(config, "gguf_gate_up_quant_types", ())
            expected_type = (gate_types[layer],) if kind in ("gate", "up") and gate_types else (
                GGML_Q4_K,
            ) if kind in ("gate", "up") else (
                GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0
            )
            if t.ggml_type not in expected_type:
                raise ValueError(
                    f"{name}: expected {expected_type}, got GGML type {t.ggml_type}"
                )
            E = config.num_experts
            I = config.moe_intermediate_size
            packed = t.packed().reshape(E, I, t.row_bytes)
            layer_buf = expert_buf.setdefault(layer, {})
            layer_buf[kind] = (packed, t.ggml_type)
            if {"gate", "up", "down"} <= set(layer_buf):
                gate, gate_type = layer_buf["gate"]
                up, up_type = layer_buf["up"]
                down, down_type = layer_buf["down"]
                if gate_type != up_type:
                    raise AssertionError(f"layer {layer}: gate/up expert types disagree")
                yield f"model.layers.{layer}.mlp.experts.gate_up_proj", torch.cat(
                    [gate, up], dim=1
                )
                yield f"model.layers.{layer}.mlp.experts.down_proj", down
                del expert_buf[layer]
            continue

        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"

        # Served path is base-model-only; terminal nextn/MTP tensors are optional
        # speculative heads and have no destination in Qwen3_5MoEForCausalLM.
        if suffix.startswith("nextn."):
            continue

        if suffix == "ssm_conv1d.weight":
            # conv channels span [q|k|v]; the v channels are value-head interleaved too.
            c = _to_fp32(t)
            c = c.clone()
            c[key_dim * 2:] = _deint_dense_rows(c[key_dim * 2:], n_vheads)
            yield f"{base}.linear_attn.conv1d.weight", c.unsqueeze(1)
            continue
        if suffix == "ssm_a":
            # GGUF stores the GDN decay directly as ``A = -exp(A_log)`` (mamba convention,
            # value-head interleaved); recover the log-decay the model consumes.
            a = _deint_dense_rows(_to_fp32(t), n_vheads)
            yield f"{base}.linear_attn.A_log", torch.log(-a)
            continue
        if suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", _deint_dense_rows(_to_fp32(t), n_vheads)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).unsqueeze(0)
            continue

        # The GDN b and a projections fuse into a dense in_proj_ba.
        if suffix == "ssm_beta.weight":
            ba_buf.setdefault(layer, {})["b"] = _to_bf16(t)
        elif suffix == "ssm_alpha.weight":
            ba_buf.setdefault(layer, {})["a"] = _to_bf16(t)
        else:
            key, _gemma = _name_to_key(suffix)
            if key is not None:
                yield f"{base}.{key}", _to_bf16(t)
                continue

            is_full = layer in full_layers
            if is_full and suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
                qkv_buf.setdefault(layer, {})[suffix[5]] = (t.packed(), t.ggml_type)
            elif suffix == "attn_qkv.weight":
                qkvz_buf.setdefault(layer, {})["qkv"] = (t.packed(), t.ggml_type)
            elif suffix == "attn_gate.weight":
                qkvz_buf.setdefault(layer, {})["z"] = (t.packed(), t.ggml_type)
            elif suffix == "attn_output.weight":
                yield f"{base}.self_attn.o_proj.qweight", t.packed()
            elif suffix == "ssm_out.weight":
                if t.ggml_type == GGML_Q8_0:
                    packed = _deint_q8_cols(t.packed(), n_vheads, q8_blocks_per_head)
                else:
                    # K-quant blocks are wider than one GDN value head. Dequantize
                    # only this projection, de-interleave columns, then repack Q8_0
                    # so the runtime sees correct HF head order with fixed row blocks.
                    packed = quantize_q8_0(
                        _deint_dense_cols(_to_fp32(t), n_vheads, vhead_dim)
                    )
                yield f"{base}.linear_attn.out_proj.qweight", packed
            elif suffix == "ffn_gate_shexp.weight":
                shexp_buf.setdefault(layer, {})["gate"] = (t.packed(), t.ggml_type)
            elif suffix == "ffn_up_shexp.weight":
                shexp_buf.setdefault(layer, {})["up"] = (t.packed(), t.ggml_type)
            elif suffix == "ffn_down_shexp.weight":
                yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
            else:
                raise ValueError(f"unmapped qwen3.5-moe GGUF tensor: {name}")

        slots = qkv_buf.get(layer)
        if slots is not None and {"q", "k", "v"} <= set(slots):
            parts = [slots[key] for key in ("q", "k", "v")]
            if len({item[1] for item in parts}) == 1:
                yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                    [item[0] for item in parts], dim=0)
            else:
                for index, (packed, _qtype) in enumerate(parts):
                    yield f"{base}.self_attn.qkv_proj.qweight_{index}", packed
            del qkv_buf[layer]
        qz = qkvz_buf.get(layer)
        if qz is not None and "qkv" in qz and "z" in qz:
            qkv, qkv_type = qz["qkv"]  # [2*key_dim + value_dim, cols]
            z, z_type = qz["z"]
            qkv = qkv.clone()
            # de-interleave the value rows (last value_dim rows of the qkv projection)
            qkv[key_dim * 2:] = _deint_q8_rows(
                qkv[key_dim * 2:], n_vheads, vhead_dim)
            z = _deint_q8_rows(z, n_vheads, vhead_dim)
            if qkv_type == z_type:
                yield f"{base}.linear_attn.in_proj_qkvz.qweight", torch.cat([qkv, z], dim=0)
            else:
                yield f"{base}.linear_attn.in_proj_qkvz.qweight_0", qkv
                yield f"{base}.linear_attn.in_proj_qkvz.qweight_1", z
            del qkvz_buf[layer]
        ba = ba_buf.get(layer)
        if ba is not None and "b" in ba and "a" in ba:
            b = _deint_dense_rows(ba["b"], n_vheads)
            a = _deint_dense_rows(ba["a"], n_vheads)
            yield f"{base}.linear_attn.in_proj_ba.weight", torch.cat([b, a], dim=0)
            del ba_buf[layer]
        gu = shexp_buf.get(layer)
        if gu is not None and "gate" in gu and "up" in gu:
            parts = [gu[key] for key in ("gate", "up")]
            if len({item[1] for item in parts}) == 1:
                yield f"{base}.mlp.shared_expert.gate_up_proj.qweight", torch.cat(
                    [item[0] for item in parts], dim=0)
            else:
                for index, (packed, _qtype) in enumerate(parts):
                    yield f"{base}.mlp.shared_expert.gate_up_proj.qweight_{index}", packed
            del shexp_buf[layer]

    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not qkvz_buf, f"incomplete GDN qkvz groups: {sorted(qkvz_buf)}"
    assert not ba_buf, f"incomplete GDN ba groups: {sorted(ba_buf)}"
    assert not shexp_buf, f"incomplete shared-expert gate/up: {sorted(shexp_buf)}"
    if include_moe_experts:
        assert not expert_buf, f"incomplete routed expert groups: {sorted(expert_buf)}"


# --------------------------------------------------------------------------------------
# Routed-expert host banks for the offload cache.
#
# The GGUF stores mostly Q4_K gate/up rows, but terminal full-attention blocks may use Q8_0;
# down rows also vary by layer. The offload cache requires uniform bank shapes, so each bank
# uses maximum native row width and dispatch carries actual per-layer type. Legacy CPU/hybrid
# down banks still re-quantize to uniform Q8_0; native GPU banks retain source down types.
# --------------------------------------------------------------------------------------


def load_gguf_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of routed experts with padded native gate/up rows and Q8_0 down.

    ``ffn_{gate,up}_exps`` are each ``[E, I, row_bytes(H, gate_type)]`` packed and are fused
    along the intermediate dim into ``gate_up``; ``ffn_down_exps`` is
    dequantized and re-quantized to Q8_0. Whole layers complete in two writes
    (gate_up + down). ``layer_sink=None`` (serving): pin each layer's banks as they
    complete via an internal :class:`PinPipeline`.
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks
    from freetoken.models.gguf.reader import iter_gguf_tensors

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    gate_types = getattr(config, "gguf_gate_up_quant_types", ()) or (GGML_Q4_K,) * L
    gate_rb = max(row_bytes(H, q) for q in set(gate_types))
    i_rb = row_bytes(I, GGML_Q8_0)
    specs = {
        "gate_up": ((E, 2 * I, gate_rb), torch.uint8),
        "down": ((E, H, i_rb), torch.uint8),
    }
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if t.name.endswith("ffn_gate_exps.weight"):
                packed = t.packed().reshape(E, I, t.row_bytes)
                banks["gate_up"][layer][:, :I, :t.row_bytes].copy_(packed)
            elif t.name.endswith("ffn_up_exps.weight"):
                packed = t.packed().reshape(E, I, t.row_bytes)
                banks["gate_up"][layer][:, I:, :t.row_bytes].copy_(packed)
            elif t.name.endswith("ffn_down_exps.weight"):
                flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
                banks["down"][layer].copy_(quantize_q8_0(flat.reshape(E, H, I)))
            else:
                continue
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)
    return banks


def load_gguf_expert_sources_native(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Load Qwen routed experts without Q8 conversion for GPU offload.

    Q5_K and Q6_K have different packed row widths. The cache needs one shape,
    so every down row uses a Q6_K-sized stride; Q5_K retains its 176-byte block
    prefix and zero tail. ``ggml_moe_a8_vec_strided`` consumes the exact source
    type and skips that tail. Converter callers stay on ``load_gguf_expert_sources``
    because its FTW schema intentionally remains the legacy uniform Q8 layout.
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks
    from freetoken.models.gguf.reader import iter_gguf_tensors

    _require_tp1("native expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    gate_types = getattr(config, "gguf_gate_up_quant_types", ()) or (GGML_Q4_K,) * L
    gate_rb = max(row_bytes(H, qtype) for qtype in set(gate_types))
    q5_rb = row_bytes(I, GGML_Q5_K)
    q6_rb = row_bytes(I, GGML_Q6_K)
    q8_rb = row_bytes(I, GGML_Q8_0)
    specs = {
        "gate_up": ((E, 2 * I, gate_rb), torch.uint8),
        "down": ((E, H, max(q6_rb, q8_rb)), torch.uint8),
    }
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if t.name.endswith("ffn_gate_exps.weight"):
                packed = t.packed().reshape(E, I, t.row_bytes)
                banks["gate_up"][layer][:, :I, :t.row_bytes].copy_(packed)
            elif t.name.endswith("ffn_up_exps.weight"):
                packed = t.packed().reshape(E, I, t.row_bytes)
                banks["gate_up"][layer][:, I:, :t.row_bytes].copy_(packed)
            elif t.name.endswith("ffn_down_exps.weight"):
                if t.ggml_type == GGML_Q4_K:
                    packed = t.packed().reshape(E, H, row_bytes(I, GGML_Q4_K))
                    banks["down"][layer][:, :, :packed.shape[-1]].copy_(packed)
                elif t.ggml_type == GGML_Q5_K:
                    packed = t.packed().reshape(E, H, q5_rb)
                    banks["down"][layer][:, :, :q5_rb].copy_(packed)
                elif t.ggml_type == GGML_Q6_K:
                    banks["down"][layer].copy_(
                        t.packed().reshape(E, H, q6_rb)
                    )
                elif t.ggml_type == GGML_Q8_0:
                    packed = t.packed().reshape(E, H, q8_rb)
                    banks["down"][layer][:, :, :q8_rb].copy_(packed)
                else:
                    raise ValueError(
                        f"native GGUF down layer {layer} has unsupported type {t.ggml_type}"
                    )
            else:
                continue
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)
    return banks


def load_gguf_expert_sources_cpu(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Convert mixed-native Qwen GGUF experts to the CPU executor's Q4_0 schema.

    Qwen3.5 GGUF uses Q4_K/Q5_K/Q6_K/Q8_0 rows, while the portable CPU MoE
    executor intentionally accepts one packed layout: Q4_0 gate/up and down.
    Convert one source tensor at a time so CPU decode works without the
    CUDA-only ``flashlib`` slot-cache dependency or a second dense expert bank.
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks
    from freetoken.models.gguf.reader import iter_gguf_tensors

    _require_tp1("CPU expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    specs = {
        "gate_up": ((E, 2 * I, row_bytes(H, GGML_Q4_0)), torch.uint8),
        "down": ((E, H, row_bytes(I, GGML_Q4_0)), torch.uint8),
    }
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _convert(t, shape):
        dense = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
        return quantize_q4_0(dense.reshape(*shape))

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if t.name.endswith("ffn_gate_exps.weight"):
                banks["gate_up"][layer][:, :I].copy_(_convert(t, (E, I, H)))
            elif t.name.endswith("ffn_up_exps.weight"):
                banks["gate_up"][layer][:, I:].copy_(_convert(t, (E, I, H)))
            elif t.name.endswith("ffn_down_exps.weight"):
                banks["down"][layer].copy_(_convert(t, (E, H, I)))
            else:
                continue
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)
    return banks


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_qwen35moe_to_gguf",
    "is_gguf_model",
    "load_gguf_expert_sources",
    "load_gguf_expert_sources_native",
    "load_gguf_expert_sources_cpu",
]
