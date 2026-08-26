"""Laguna GGUF adapter: ModelConfig from GGUF metadata + native-quant conversion.

Laguna (poolside) is hybrid full/SWA attention (full at ``il % 4 == 0`` with 48
query heads, SWA elsewhere with 72), sigmoid-routed MoE with a selection-only
score-correction bias, one shared expert, a per-head softplus attention output
gate, and QK-norm. Reference: llama.cpp ``src/models/laguna.cpp``.

Unsloth's "Dynamic" GGUFs mix quant types per tensor (Q4_K embed/head, Q5_K/Q6_K
attention + dense, IQ1_S/IQ2_XXS/IQ3_XXS/IQ4_XS expert banks), so conversion
swaps dense projections for :class:`DeferredGGUFLinear`, whose packed buffer is
materialized at load time when each tensor's ggml type is known.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _embed_quant(shim: "GgufConfigShim") -> int | None:
    from freetoken.models.gguf.reader import gguf_tensor_type

    return gguf_tensor_type(shim.model_path, "token_embd.weight")


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        full_key = f"laguna.{key}"
        val = m.get(full_key)
        if val is None:
            raise KeyError(f"missing GGUF metadata key {full_key}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    intermediate = int(g("feed_forward_length"))
    context = int(g("context_length"))

    head_counts = tuple(int(h) for h in g("attention.head_count"))
    assert len(head_counts) == num_layers, "attention.head_count length != block_count"
    num_qo_heads = max(head_counts)

    full_count = min(head_counts)
    full_layer_ids = tuple(i for i, c in enumerate(head_counts) if c == full_count)
    swa_layer_ids = tuple(i for i in range(num_layers) if i not in full_layer_ids)
    assert full_layer_ids == tuple(i for i in range(0, num_layers, 4)), (
        "Laguna full attention layers must be exactly i % 4 == 0"
    )

    num_kv_heads = int(g("attention.head_count_kv"))
    head_dim = int(g("attention.key_length"))
    assert head_dim == int(g("attention.value_length")), "key_length != value_length"

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(g("rope.dimension_count")),
        max_position=context,
        base=float(g("rope.freq_base")),
        scaling={
            "rope_type": "yarn",
            "factor": float(g("rope.scaling.factor")),
            # ggml applies yarn_attn_factor verbatim (1.0 here); without it
            # freetoken's yarn would default to 1 + 0.1*ln(factor).
            "attention_factor": float(g("rope.scaling.yarn_attn_factor")),
            "original_max_position_embeddings": int(g("rope.scaling.original_context_length")),
            "beta_fast": float(g("rope.scaling.yarn_beta_fast")),
            "beta_slow": float(g("rope.scaling.yarn_beta_slow")),
        },
    )
    swa_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(g("rope.dimension_count_swa")),
        max_position=context,
        base=float(g("rope.freq_base_swa")),
        scaling=None,
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_qo_heads_per_layer=head_counts,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=intermediate,
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        rotary_config=full_rotary,
        hidden_act="silu",
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        n_shared_experts=1,
        norm_topk_prob=bool(g("expert_weights_norm")),
        routed_scaling_factor=float(g("expert_weights_scale")),
        # The selection-only e_score_correction bias (blk.N.exp_probs_b) is part of
        # the laguna arch itself, not flagged in metadata; has_router_bias (a bias
        # on the router *linear*) stays False.
        first_k_dense_replace=int(g("leading_dense_block_count")),
        use_qk_norm=True,
        model_type="laguna",
        architectures=list(shim.architectures),
        moe_enabled=True,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layer_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=full_rotary,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_layer_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=swa_rotary,
                sliding_window=int(g("attention.sliding_window")),
            ),
        ),
        gguf_embed_quant=_embed_quant(shim),
        gguf_model_path=shim.model_path,
        # Routed experts: mixed per-layer ggml types (Unsloth Dynamic), served by the
        # "gguf" offload bank format; types read from the tensor table when present
        # (None on a metadata-only FTW source).
        expert_quant="gguf",
        moe_weight_format="gguf",
        gguf_expert_types=_expert_types(shim),
    )


def _expert_types(shim: "GgufConfigShim") -> tuple[tuple[int, int], ...] | None:
    """(gate_up, down) ggml type per MoE layer, from the file's tensor table.

    gate and up always share a type in the published files (asserted); a
    metadata-only GGUF has no tensor table -> None.
    """
    from freetoken.models.gguf.reader import gguf_tensor_names

    if not gguf_tensor_names(shim.model_path):
        return None
    types = gguf_tensor_types(shim.model_path)
    num_layers = int(shim.metadata["laguna.block_count"])
    dense = int(shim.metadata["laguna.leading_dense_block_count"])
    out = []
    for i in range(dense, num_layers):
        gu = types[f"blk.{i}.ffn_gate_exps.weight"]
        up = types[f"blk.{i}.ffn_up_exps.weight"]
        dn = types[f"blk.{i}.ffn_down_exps.weight"]
        if gu != up:
            raise ValueError(f"blk.{i}: gate/up expert banks have different ggml types")
        out.append((gu, dn))
    return tuple(out)


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken laguna module params.
# --------------------------------------------------------------------------------------

# Routed expert banks are consumed by the MoE offload cache, not the state dict.
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")

# Per-layer 1:1 tensors dequantized to a dense dtype (suffix -> (rel name, dtype)).
# The router gate and selection bias are consumed fp32 (top-k boundary fidelity,
# minimax_m3 precedent); norms load bf16.
_DENSE_MAP = {
    "attn_norm.weight": ("input_layernorm.weight", "bf16"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "bf16"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "bf16"),
    "ffn_norm.weight": ("ffn_norm.weight", "bf16"),
    "ffn_gate_inp.weight": ("mlp.gate.weight", "f32"),
    "exp_probs_b.bias": ("mlp.e_score_correction_bias", "f32"),
}

# Packed (native-quant) projections that map 1:1 (suffix -> rel name). q/k/v stay
# separate modules: their ggml types differ within a layer in some laguna files
# (XS Q4_K_M quantizes attn_v as Q6_K on half the layers), so packed rows cannot fuse.
_PACKED_MAP = {
    "attn_q.weight": "self_attn.q_proj.qweight",
    "attn_k.weight": "self_attn.k_proj.qweight",
    "attn_v.weight": "self_attn.v_proj.qweight",
    "attn_output.weight": "self_attn.o_proj.qweight",
    "attn_gate.weight": "self_attn.gate_proj.qweight",
    "ffn_down.weight": "mlp.down_proj.qweight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.qweight",
}
_GATE_UP_SLOTS = {
    "ffn_gate.weight": ("mlp.gate_up_proj", "gate"),
    "ffn_up.weight": ("mlp.gate_up_proj", "up"),
    "ffn_gate_shexp.weight": ("mlp.shared_experts.gate_up_proj", "gate"),
    "ffn_up_shexp.weight": ("mlp.shared_experts.gate_up_proj", "up"),
}


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(f"laguna GGUF {what} currently supports TP=1 only")


def gguf_tensor_types(model_path: str) -> dict[str, int]:
    """One pass over the tensor table: name -> ggml type (no tensor data touched)."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    return {t.name: t.ggml_type for t in iter_gguf_tensors(model_path)}


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
):
    """Yield (param_name, tensor) for every non-expert laguna param.

    Quantized projections stay packed and are yielded as ``.qweight`` (uint8);
    q/k/v and gate/up fuse by concatenating packed rows along the output dim
    (valid only when the components share one ggml type -- Unsloth Dynamic keeps
    fused groups uniform, enforced here). Norms load bf16; the router gate and
    exp_probs_b load fp32. Routed experts go to the offload cache (4b), not here.
    """
    import torch

    from freetoken.models.gguf.dequant import dequantize
    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert not include_moe_experts, (
        "laguna GGUF experts are loaded into the MoE offload cache, not the state dict"
    )
    assert include_non_moe
    _require_tp1("weight loading")

    def dense(t, kind):
        dtype = torch.float32 if kind == "f32" else torch.bfloat16
        return dequantize(t.packed().reshape(-1), t.ggml_type, dtype).reshape(t.shape)

    gate_up_buf: dict[tuple[int, str], dict[str, tuple]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", dense(t, "bf16")
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if not name.startswith("blk."):
            raise ValueError(f"unmapped laguna GGUF tensor: {name}")
        if name.endswith(_EXPERT_SUFFIXES):
            continue  # routed experts -> offload banks

        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"

        if suffix in _DENSE_MAP:
            rel, kind = _DENSE_MAP[suffix]
            yield f"{base}.{rel}", dense(t, kind)
        elif suffix in _PACKED_MAP:
            yield f"{base}.{_PACKED_MAP[suffix]}", t.packed()
        elif suffix in _GATE_UP_SLOTS:
            rel, slot = _GATE_UP_SLOTS[suffix]
            gate_up_buf.setdefault((layer, rel), {})[slot] = (t.packed(), t.ggml_type)
        else:
            raise ValueError(f"unmapped laguna GGUF tensor: {name}")

        for key in [k for k in gate_up_buf if k[0] == layer]:
            gu = gate_up_buf[key]
            if len(gu) == 2:
                if gu["gate"][1] != gu["up"][1]:
                    raise ValueError(f"blk.{layer}: mixed ggml types across fused gate/up")
                yield f"{base}.{key[1]}.qweight", torch.cat(
                    [gu["gate"][0], gu["up"][0]], dim=0
                )
                del gate_up_buf[key]

    assert not gate_up_buf, f"incomplete gate_up groups: {sorted(gate_up_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return config.gguf_embed_quant is not None


class DeferredGGUFLinear(BaseOP):
    """GGUF linear whose quant type is only known at weight-load time.

    Unsloth Dynamic checkpoints choose the ggml type per tensor, so conversion
    cannot size the packed buffer up front; the loader calls :meth:`materialize`
    with the tensor's recorded type before copying rows in.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type: int | None = None
        self.qweight: torch.Tensor | None = None
        self.bias = torch.empty(out_features) if has_bias else None

    def materialize(self, quant_type: int) -> None:
        from freetoken.models.gguf.dequant import row_bytes

        self._quant_type = quant_type
        self.qweight = torch.empty(
            self.out_features, row_bytes(self.in_features, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.layers.gguf import fused_mul_mat_gguf

        assert self.qweight is not None and self._quant_type is not None, (
            "DeferredGGUFLinear used before materialize() -- weight was never loaded"
        )
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out


class LagunaGGUFLMHead(DeferredGGUFLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf
        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        assert self.qweight is not None and self._quant_type is not None
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


def _swap(owner, attr: str) -> None:
    # _LinearTPImpl exposes local_*_size (== full sizes at TP=1, which the GGUF
    # path requires); the weight shape is the ground truth either way.
    old = getattr(owner, attr)
    out_features, in_features = old.weight.shape
    setattr(
        owner,
        attr,
        DeferredGGUFLinear(in_features, out_features, getattr(old, "bias", None) is not None),
    )


def convert_laguna_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace laguna's dense projections + embedding + head with GGUF ops.

    Swapped: attention qkv/gate/o, the dense layer's MLP, every shared expert, the
    embedding, and the (untied) lm_head. Kept dense: the fp32 MoE router gate and
    e_score_correction_bias, all RMSNorms (F32 in the GGUF), and the routed expert
    banks (they live on the MoE offload cache).

    When the source ``.gguf`` is known (``config.gguf_model_path``), every swapped
    module is materialized here from the file's per-tensor ggml types, so the
    packed buffers exist before the engine collects ``model.state_dict()``.
    """
    from freetoken.layers.gguf import GGUFEmbedding

    types = gguf_tensor_types(config.gguf_model_path) if config.gguf_model_path else None

    def qt(name: str) -> int | None:
        return None if types is None else types.get(name)

    def mat(module: DeferredGGUFLinear, tensor_name: str) -> None:
        t = qt(tensor_name)
        if t is not None:
            module.materialize(t)

    model.model.embed_tokens = GGUFEmbedding(
        config.vocab_size, config.hidden_size, config.gguf_embed_quant
    )
    for i, layer in enumerate(model.model.layers.op_list):
        for attr, tname in (
            ("q_proj", "attn_q.weight"),
            ("k_proj", "attn_k.weight"),
            ("v_proj", "attn_v.weight"),
            ("gate_proj", "attn_gate.weight"),
            ("o_proj", "attn_output.weight"),
        ):
            _swap(layer.self_attn, attr)
            mat(getattr(layer.self_attn, attr), f"blk.{i}.{tname}")
        mlp = layer.mlp
        if hasattr(mlp, "gate_up_proj"):  # dense leading layer (LagunaMLP)
            _swap(mlp, "gate_up_proj")
            _swap(mlp, "down_proj")
            mat(mlp.gate_up_proj, f"blk.{i}.ffn_gate.weight")
            mat(mlp.down_proj, f"blk.{i}.ffn_down.weight")
        else:  # LagunaSparseMoeBlock: shared expert only (router stays fp32)
            _swap(mlp.shared_experts, "gate_up_proj")
            _swap(mlp.shared_experts, "down_proj")
            mat(mlp.shared_experts.gate_up_proj, f"blk.{i}.ffn_gate_shexp.weight")
            mat(mlp.shared_experts.down_proj, f"blk.{i}.ffn_down_shexp.weight")
    # Untied output head (output.weight, Q4_K in the target file).
    model.lm_head = LagunaGGUFLMHead(config.hidden_size, config.vocab_size)
    mat(model.lm_head, "output.weight")



# --------------------------------------------------------------------------------------
# Routed expert banks (mixed per-layer ggml types) for the MoE offload cache.
# --------------------------------------------------------------------------------------


def _expert_bank_geometry(config: ModelConfig) -> list[tuple[int, int]]:
    """Compact flat-slot strides per MoE layer, each 64-byte aligned."""
    from freetoken.models.gguf.dequant import row_bytes

    assert config.gguf_expert_types, "laguna expert banks need gguf_expert_types"
    H, I = config.hidden_size, config.moe_intermediate_size
    def align(n: int) -> int:
        return (n + 63) // 64 * 64

    return [
        (align(2 * I * row_bytes(H, gu)), align(H * row_bytes(I, dn)))
        for gu, dn in config.gguf_expert_types
    ]


def load_gguf_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-MoE-layer host banks of the routed experts' native packed bytes.

    Each bank is one compact flat ``[E, stride]`` uint8 tensor per MoE layer (bank
    index = layer_id - first_k_dense_replace). The GPU slot cache chooses the maximum
    per-bank stride, while host RAM stores only each layer's real packed payload.
    """
    from freetoken.models.gguf.dequant import row_bytes
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import HostBank, LayerCompletionTracker, PinPipeline

    _require_tp1("expert banks")
    types = config.gguf_expert_types
    assert types, "laguna expert banks need gguf_expert_types (tensor table missing?)"
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    L = len(types)  # MoE layers only
    geometry = _expert_bank_geometry(config)
    hb = {
        "gate_up": [HostBank((E, gu_stride), torch.uint8) for gu_stride, _ in geometry],
        "down": [HostBank((E, dn_stride), torch.uint8) for _, dn_stride in geometry],
    }
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen_gu, seen_dn = set(), set()

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        gu_parts: dict[int, dict[str, torch.Tensor]] = {}
        for t in iter_gguf_tensors(model_path):
            name = t.name
            if not name.startswith("blk.") or not name.endswith(tuple(_EXPERT_SUFFIXES)):
                continue
            layer = int(name.split(".")[1])
            bank_id = layer - config.first_k_dense_replace
            gu_t, dn_t = types[bank_id]
            if name.endswith("ffn_down_exps.weight"):
                pay = H * row_bytes(I, dn_t)
                banks["down"][bank_id][:, :pay].copy_(t.packed().reshape(E, pay))
                seen_dn.add(bank_id)
                if tracker is not None:
                    tracker.note(bank_id)
            else:
                # gate and up arrive as separate tensors; each expert's slot holds
                # gate rows then up rows (the fused gate_up layout the kernel expects).
                half = I * row_bytes(H, gu_t)
                part = gu_parts.setdefault(bank_id, {})
                part["gate" if "gate" in name else "up"] = t.packed().reshape(E, half)
                if len(part) < 2:
                    continue
                dst = banks["gate_up"][bank_id]
                dst[:, :half].copy_(part["gate"])
                dst[:, half : 2 * half].copy_(part["up"])
                del gu_parts[bank_id]
                seen_gu.add(bank_id)
                if tracker is not None:
                    tracker.note(bank_id)
        assert not gu_parts, f"incomplete expert gate/up layers: {sorted(gu_parts)}"

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    want = set(range(L))
    assert seen_gu == want and seen_dn == want, (
        f"missing expert layers: gate_up {sorted(want - seen_gu)}, down {sorted(want - seen_dn)}"
    )
    return banks


def dummy_gguf_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    """Random banks shaped like ``load_gguf_expert_sources`` output."""
    from freetoken.moe.host_banks import HostBank, pin_banks

    E = config.num_experts
    L = len(config.gguf_expert_types or ())
    assert L, "laguna dummy expert banks need gguf_expert_types"
    geometry = _expert_bank_geometry(config)
    hb = {
        "gate_up": [HostBank((E, gu_stride), torch.uint8) for gu_stride, _ in geometry],
        "down": [HostBank((E, dn_stride), torch.uint8) for _, dn_stride in geometry],
    }
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks

__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "gguf_tensor_types",
    "is_gguf_model",
    "DeferredGGUFLinear",
    "LagunaGGUFLMHead",
    "convert_laguna_to_gguf",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
]
