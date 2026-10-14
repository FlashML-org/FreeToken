"""Gemma 4 GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata.

The GGUF checkpoint's geometry is identical to the HF gemma4 model (hybrid
SWA/full attention with ``k_eq_v`` full layers, 128 routed experts + shared MLP,
final-logit softcap), so this produces the *same* ``ModelConfig`` as
``gemma4.config.parse_config`` -- only the source is GGUF KV metadata instead of a
HF config object. transformers' own GGUF->config conversion is rejected by the
gemma4 strict dataclass (per-layer ``num_key_value_heads`` array), so we read the
metadata directly. ``expert_quant`` is set to ``"q4_0"`` to route the routed experts
through the native-Q4_0 offload-cache path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch
from freetoken.layers import BaseOP

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q8_0, GGML_Q6_K, dequantize, row_bytes

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _full_rotary_dim(shim: "GgufConfigShim", full_head_dim: int) -> int:
    """Rotated width of the full-attention layers, recovered from ``rope_freqs.weight``.

    llama.cpp writes ``rope.dimension_count = full head_dim`` and carries gemma4's
    partial_rotary_factor as a divisor tensor instead: ``[1.0] * n_rot//2 + [1e30] * rest``,
    which drives the unrotated tail's frequencies to zero (ggml ops.cpp ``theta / ff``).
    Count the untouched entries to get the rotated width back.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    for t in iter_gguf_tensors(shim.model_path):
        if t.name == "rope_freqs.weight":  # F32, so the packed bytes ARE the values
            freqs = t.packed().reshape(-1).view(torch.float32)
            return int((freqs == 1.0).sum().item()) * 2
    # A metadata-only GGUF (an FTW dir's source_metadata.gguf) has no tensor table. Every
    # released gemma4 sets partial_rotary_factor 0.25 on the full layers, and the converter
    # asserts rope_type == "proportional" (llama.cpp conversion/gemma.py).
    return full_head_dim // 4


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        val = m.get(f"gemma4.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key gemma4.{key}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    kv_per_layer = g("attention.head_count_kv")  # per-layer list
    # True -> sliding-window (SWA) layer, False -> full attention.
    swa_pattern = [bool(x) for x in g("attention.sliding_window_pattern")]
    assert len(swa_pattern) == num_layers, "sliding_window_pattern length != block_count"

    swa_layer_ids = tuple(i for i, is_swa in enumerate(swa_pattern) if is_swa)
    full_layer_ids = tuple(i for i, is_swa in enumerate(swa_pattern) if not is_swa)

    swa_head_dim = int(g("attention.key_length_swa"))
    full_head_dim = int(g("attention.key_length"))
    swa_kv = int(kv_per_layer[swa_layer_ids[0]]) if swa_layer_ids else int(kv_per_layer[0])
    full_kv = int(kv_per_layer[full_layer_ids[0]]) if full_layer_ids else int(kv_per_layer[0])

    max_pos = int(g("context_length"))
    full_rotary = RotaryConfig(
        head_dim=full_head_dim,
        rotary_dim=_full_rotary_dim(shim, full_head_dim),
        max_position=max_pos,
        base=float(g("rope.freq_base")),
        # gemma4's full layers are partial-rotary ("proportional"): the tail dims must come
        # out unrotated. llama.cpp expresses that with rope.dimension_count == full head_dim
        # plus a rope_freqs divisor tensor, so a plain "default" rope here would rotate dims
        # that have to be identity (conversion/gemma.py generate_extra_tensors).
        scaling={"rope_type": "proportional"},
    )
    swa_rotary = RotaryConfig(
        head_dim=swa_head_dim,
        rotary_dim=int(g("rope.dimension_count_swa")),
        max_position=max_pos,
        base=float(g("rope.freq_base_swa")),
        scaling=None,
    )

    hf_nvfp4 = shim.is_hf_nvfp4
    from freetoken.models.gguf.reader import iter_gguf_tensors

    embedding_bf16 = any(
        t.name in {"model.language_model.embed_tokens.weight", "token_embd.weight"}
        and t.ggml_type != 40
        for t in iter_gguf_tensors(shim.model_path)
    )
    embedding_type = next(
        (
            t.ggml_type
            for t in iter_gguf_tensors(shim.model_path)
            if t.name in {"model.language_model.embed_tokens.weight", "token_embd.weight"}
        ),
        None,
    )
    router_bf16 = any(
        (
            t.name.endswith(".router.proj.weight")
            or t.name.endswith(".ffn_gate_inp.weight")
        )
        and t.ggml_type != 40
        for t in iter_gguf_tensors(shim.model_path)
    )
    dense_bf16 = any(
        (
            t.name.endswith(".self_attn.q_proj.weight")
            or t.name.endswith(".attn_q.weight")
        )
        and t.ggml_type != 40
        for t in iter_gguf_tensors(shim.model_path)
    )
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=full_kv,
        head_dim=full_head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(g("feed_forward_length")),
        hidden_act="gelu_tanh",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=full_rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        norm_topk_prob=True,
        model_type="gemma4",
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant="nvfp4" if hf_nvfp4 else _gguf_expert_quant(shim.model_path),
        moe_weight_format="nvfp4" if hf_nvfp4 else _gguf_expert_quant(shim.model_path),
        gguf_embedding_bf16=embedding_bf16,
        gguf_embedding_type=embedding_type,
        gguf_model_path=shim.model_path,
        gguf_router_bf16=router_bf16,
        gguf_dense_bf16=dense_bf16,
        use_qk_norm=True,
        attn_sm_scale=1.0,
        final_logit_softcapping=float(g("final_logit_softcapping")),
        embedding_scale=float(hidden) ** 0.5,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layer_ids,
                num_kv_heads=full_kv,
                head_dim=full_head_dim,
                rotary_config=full_rotary,
                # Full layers ship no v_proj in the GGUF (k reused as v).
                k_eq_v=True,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_layer_ids,
                num_kv_heads=swa_kv,
                head_dim=swa_head_dim,
                rotary_config=swa_rotary,
                sliding_window=int(g("attention.sliding_window")),
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken gemma4 module params.
# --------------------------------------------------------------------------------------

# Per-layer 1:1 norm/scale tensors (gguf suffix -> freetoken module-relative name). All
# are tiny F32 tensors dequantized to bf16.
_LAYER_SCALAR_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "pre_ffw_norm_2.weight": "feed_forward.pre_feedforward_layernorm_2.weight",
    "post_ffw_norm.weight": "feed_forward.post_feedforward_layernorm.weight",
    "post_ffw_norm_1.weight": "feed_forward.post_feedforward_layernorm_1.weight",
    "post_ffw_norm_2.weight": "feed_forward.post_feedforward_layernorm_2.weight",
    "layer_output_scale.weight": "feed_forward.layer_scalar",
    "ffn_gate_inp.weight": "feed_forward.router.proj.weight",
    "ffn_gate_inp.scale": "feed_forward.router.scale",
    "ffn_down_exps.scale": "feed_forward.router.per_expert_scale",
}
# Expert weight tensors (handled by the offload bank loader, not iter_gguf_weights).
_EXPERT_SUFFIXES = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight")


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense bf16 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 with a clear
    error instead of failing later on a confusing shape mismatch (mirrors the HF
    gemma4 loader's TP=1 restriction)."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"gemma4 GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def _gguf_expert_quant(model_path: str) -> str:
    """Return the native GGML format used by the routed expert tensors."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    expert_types = {
        t.ggml_type
        for t in iter_gguf_tensors(model_path)
        if "exps.weight" in t.name
    }
    if not expert_types:
        return "q4_0"
    if expert_types == {GGML_Q8_0}:
        return "q8_0"
    if expert_types == {GGML_Q4_0}:
        return "q4_0"
    raise ValueError(f"unsupported mixed GGUF expert quantization types: {sorted(expert_types)}")


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert gemma4 param.

    Quantized projections (attention q/k/v/o, shared-MLP gate/up/down) and the
    embedding stay in their native packed block layout and are yielded as ``.qweight``
    (uint8); norms / router / per-layer scalars dequantize to bf16. q/k/v and gate/up
    are fused by concatenating packed rows along the output dim (same input dim ->
    same ``row_bytes``). Routed experts are served from the offload cache, so they are
    skipped here (asserts the offload contract like the other MoE models).
    """
    from freetoken.models.gguf.reader import GGML_NVFP4, iter_gguf_tensors, load_gguf_metadata
    from freetoken.utils import cached_load_hf_config

    from freetoken.models.gguf.reader import gguf_tensor_names

    tensor_names = gguf_tensor_names(model_path)
    if (
        "model.language_model.embed_tokens.weight" in tensor_names
        or (
            "token_embd.weight" in tensor_names
            and any(t.ggml_type == GGML_NVFP4 for t in iter_gguf_tensors(model_path))
        )
    ):
        yield from _iter_hf_nvfp4_weights(model_path, device, include_moe_experts, include_non_moe)
        return

    assert not include_moe_experts, (
        "gemma4 GGUF stores experts as Q4_0 and only supports the offload backend; "
        "experts are loaded into the offload cache via load_q4_0_expert_sources()."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    # Full-attention layers ship no attn_v (k reused as v); SWA layers do. Knowing which
    # is which lets us emit the fused qkv as soon as its parts are present.
    config = parse_gguf_config(cached_load_hf_config(model_path))
    k_eq_v_layers = {
        lid
        for lid in range(config.num_layers)
        if isinstance(config.attention_group_for_layer(lid), FullAttentionGroupConfig)
        and config.attention_group_for_layer(lid).k_eq_v
    }

    # Per-layer fusion buffers: layer -> {slot: packed[out, row_bytes]}.
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()  # Q6_K packed table
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if name == "rope_freqs.weight":
            continue  # rope frequencies recomputed in-engine
        if not name.startswith("blk."):
            continue
        if any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            continue  # routed experts -> offload banks

        layer = layer_of(name)
        suffix = name.split(".", 2)[2]  # after "blk.N."
        base = f"model.layers.{layer}"

        if suffix in _LAYER_SCALAR_MAP:
            rel = _LAYER_SCALAR_MAP[suffix]
            tensor = _to_bf16(t)
            if rel.endswith("layer_scalar"):
                tensor = tensor.reshape(1)
            yield f"{base}.{rel}", tensor
            continue

        # Quantized projections: keep packed; fuse q/k/v and gate/up.
        if suffix == "attn_q.weight":
            qkv_buf.setdefault(layer, {})["q"] = t.packed()
        elif suffix == "attn_k.weight":
            qkv_buf.setdefault(layer, {})["k"] = t.packed()
        elif suffix == "attn_v.weight":
            qkv_buf.setdefault(layer, {})["v"] = t.packed()
        elif suffix == "ffn_gate.weight":
            gate_up_buf.setdefault(layer, {})["gate"] = t.packed()
        elif suffix == "ffn_up.weight":
            gate_up_buf.setdefault(layer, {})["up"] = t.packed()
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
        elif suffix == "ffn_down.weight":
            yield f"{base}.feed_forward.shared_mlp.down_proj.qweight", t.packed()
        else:
            raise ValueError(f"unmapped gemma4 GGUF tensor: {name}")

        # Emit fused qkv once all parts are present: k_eq_v full layers reuse k as v
        # (no attn_v tensor), SWA layers wait for their separate attn_v.
        slots = qkv_buf.get(layer)
        if slots is not None and "q" in slots and "k" in slots:
            if layer in k_eq_v_layers:
                v = slots["k"]
            elif "v" in slots:
                v = slots["v"]
            else:
                v = None
            if v is not None:
                yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                    [slots["q"], slots["k"], v], dim=0
                )
                del qkv_buf[layer]
        gu = gate_up_buf.get(layer)
        if gu is not None and "gate" in gu and "up" in gu:
            yield f"{base}.feed_forward.shared_mlp.gate_up_proj.qweight", torch.cat(
                [gu["gate"], gu["up"]], dim=0
            )
            del gate_up_buf[layer]

    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not gate_up_buf, f"incomplete gate_up groups: {sorted(gate_up_buf)}"


from freetoken.models.gguf.nvfp4 import nvfp4_parts

_nvfp4_parts = nvfp4_parts


def _iter_hf_nvfp4_weights(model_path, device, include_moe_experts, include_non_moe):
    from freetoken.models.gguf.reader import GGML_NVFP4, iter_gguf_tensors, load_gguf_metadata

    tensors = list(iter_gguf_tensors(model_path))
    by_name = {t.name: t for t in tensors}
    merge: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    metadata = load_gguf_metadata(model_path)
    swa_pattern = [bool(x) for x in metadata["gemma4.attention.sliding_window_pattern"]]

    def rename(name: str) -> str:
        if name == "token_embd.weight":
            return "model.embed_tokens.weight"
        if name == "output_norm.weight":
            return "model.norm.weight"
        if name.startswith("blk."):
            _, layer, suffix = name.split(".", 2)
            layer_prefix = f"model.layers.{layer}"
            legacy_map = {
                "attn_q.weight": "self_attn.q_proj.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "ffn_gate.weight": "feed_forward.shared_mlp.gate_proj.weight",
                "ffn_up.weight": "feed_forward.shared_mlp.up_proj.weight",
                "ffn_down.weight": "feed_forward.shared_mlp.down_proj.weight",
                "ffn_gate_inp.weight": "feed_forward.router.proj.weight",
            }
            if suffix in legacy_map:
                return f"{layer_prefix}.{legacy_map[suffix]}"
            if suffix in _LAYER_SCALAR_MAP:
                return f"{layer_prefix}.{_LAYER_SCALAR_MAP[suffix]}"
            if suffix.endswith("_exps.weight"):
                return f"{layer_prefix}.feed_forward.experts.{suffix}"
            return f"{layer_prefix}.{suffix}"
        name = name.removeprefix("model.language_model.")
        name = "model." + name
        marker = ".layers."
        if marker not in name:
            return name
        layer_prefix, layer_key = name.split(marker, 1)
        layer_id, _, layer_key = layer_key.partition(".")
        prefix = f"{layer_prefix}{marker}{layer_id}"
        if layer_key.startswith("mlp."):
            return f"{prefix}.feed_forward.shared_mlp.{layer_key[4:]}"
        if layer_key.startswith((
            "experts.", "router.", "layer_scalar",
            "post_feedforward_layernorm.", "post_feedforward_layernorm_1.",
            "post_feedforward_layernorm_2.", "pre_feedforward_layernorm_2.",
        )):
            return f"{prefix}.feed_forward.{layer_key}"
        return name

    def emit(name: str, packed: torch.Tensor, scales: torch.Tensor):
        global_name = name.removesuffix(".weight") + ".weight_scale"
        if global_name in by_name:
            global_scale = _to_bf16(by_name[global_name]).reshape(-1)
        else:
            global_scale = torch.ones(packed.shape[0], dtype=torch.float16)
        # FreeToken's E4M3 decoder uses the full-scale representation, so the
        # standard E2M1 LUT and GGML's doubled-code convention already cancel out.
        return packed.to(device), scales.to(device), global_scale.to(device)

    def state(name: str, values):
        packed, scales, global_scale = values
        yield name, packed
        yield name.removesuffix(".weight") + ".weight_scale", scales
        yield name.removesuffix(".weight") + ".weight_global", global_scale

    for t in tensors:
        name = t.name
        if not (
            name.startswith("model.language_model.")
            or name.startswith("blk.")
            or name in {"token_embd.weight", "output_norm.weight"}
        ):
            continue
        if name.endswith(".weight_scale") or name.endswith(".weight_scale_2"):
            continue
        if t.ggml_type != GGML_NVFP4:
            if include_non_moe and ".experts." not in name:
                key = rename(name)
                tensor = _to_bf16(t).to(device)
                scale_name = name.removesuffix(".weight") + ".weight_scale"
                if name.endswith(".weight") and scale_name in by_name:
                    scale = _to_bf16(by_name[scale_name]).to(device=device, dtype=tensor.dtype)
                    tensor = tensor * scale.reshape(-1, 1)
                if key.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
                    slot = key.rsplit(".", 2)[-2][0]
                    fused = key.rsplit(".", 2)[0] + ".qkv_proj.weight"
                    merge.setdefault(fused, {})[slot] = (tensor, None, None)
                    if all(x in merge[fused] for x in "qkv"):
                        parts = merge.pop(fused)
                        yield fused, torch.cat([parts[x][0] for x in "qkv"])
                elif key.endswith((".gate_proj.weight", ".up_proj.weight")):
                    slot = "gate" if key.endswith(".gate_proj.weight") else "up"
                    fused = key.rsplit(".", 2)[0] + ".gate_up_proj.weight"
                    merge.setdefault(fused, {})[slot] = (tensor, None, None)
                    if all(x in merge[fused] for x in ("gate", "up")):
                        parts = merge.pop(fused)
                        yield fused, torch.cat([parts[x][0] for x in ("gate", "up")])
                else:
                    yield key, tensor
            continue
        if not name.endswith(".weight"):
            continue
        packed, scales = _nvfp4_parts(t)
        key = rename(name)
        if ".experts." in key:
            if include_moe_experts:
                yield from state(key, emit(name, packed, scales))
            continue
        if not include_non_moe:
            continue
        if key.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
            slot = key.rsplit(".", 2)[-2][0]
            fused = key.rsplit(".", 2)[0] + ".qkv_proj.weight"
            merge.setdefault(fused, {})[slot] = emit(name, packed, scales)
            if all(slot in merge[fused] for slot in "qkv"):
                parts = merge.pop(fused)
                if parts["q"][1] is None:
                    yield fused, torch.cat([parts[x][0] for x in "qkv"])
                else:
                    yield from state(fused, (torch.cat([parts[x][0] for x in "qkv"]), torch.cat([parts[x][1] for x in "qkv"]), torch.cat([parts[x][2] for x in "qkv"])))
        elif key.endswith((".gate_proj.weight", ".up_proj.weight")):
            slot = "gate" if key.endswith(".gate_proj.weight") else "up"
            fused = key.rsplit(".", 2)[0] + ".gate_up_proj.weight"
            merge.setdefault(fused, {})[slot] = emit(name, packed, scales)
            if all(slot in merge[fused] for slot in ("gate", "up")):
                parts = merge.pop(fused)
                if parts["gate"][1] is None:
                    yield fused, torch.cat([parts[x][0] for x in ("gate", "up")])
                else:
                    yield from state(fused, (torch.cat([parts[x][0] for x in ("gate", "up")]), torch.cat([parts[x][1] for x in ("gate", "up")]), torch.cat([parts[x][2] for x in ("gate", "up")])))
        elif key.endswith(".down_proj.weight") or key.endswith(".weight"):
            yield from state(key, emit(name, packed, scales))
    for fused, parts in list(merge.items()):
        if fused.endswith(".self_attn.qkv_proj.weight"):
            layer = int(fused.split(".layers.", 1)[1].split(".", 1)[0])
            if not swa_pattern[layer] and set(parts) == {"q", "k"}:
                parts["v"] = parts["k"]
                del merge[fused]
                if parts["q"][1] is None:
                    yield fused, torch.cat([parts[x][0] for x in "qkv"])
                else:
                    yield from state(fused, (torch.cat([parts[x][0] for x in "qkv"]), torch.cat([parts[x][1] for x in "qkv"]), torch.cat([parts[x][2] for x in "qkv"])))
    assert not merge, f"incomplete HF mixed groups: {sorted(merge)}"


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 Linear/Embedding -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "moe_weight_format", None) in ("q4_0", "q8_0", "nvfp4")


class GGUFTiedLMHead:
    """Tied LM head over a native Q6_K embedding table (logits via ggml matmul).

    Holds only a reference to the GGUF embedding (no params of its own -> empty
    state_dict), mirroring ``ParallelLMHead`` with ``tie_word_embeddings``. TP=1 only.
    """

    def __init__(self, embedding, quant_type: int):
        self._embedding = embedding
        self._quant_type = quant_type

    def state_dict(self, *, prefix: str = "", result=None):
        return result if result is not None else {}

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        state_dict.pop(f"{prefix}.weight", None)
        state_dict.pop(f"{prefix}.bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self._embedding.qweight, self._quant_type)


class Nvfp4Embedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float | None = None):
        self.weight = torch.empty(num_embeddings, embedding_dim // 2, dtype=torch.uint8)
        self.weight_scale = torch.empty(num_embeddings, embedding_dim // 16, dtype=torch.float8_e4m3fn)
        self.weight_global = torch.empty(num_embeddings, dtype=torch.float16)
        self.embed_scale = embed_scale

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        self.weight = state_dict.pop(f"{prefix}.weight")
        self.weight_scale = state_dict.pop(f"{prefix}.weight_scale")
        self.weight_global = state_dict.pop(f"{prefix}.weight_global")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

        flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
        # Dequantize only the vocabulary rows used by this batch. Passing the full
        # [vocab, hidden] table as one expert-like bank makes the dequantizer allocate
        # [tokens, vocab, hidden], which can be hundreds of GiB during warmup.
        packed = self.weight.index_select(0, flat_ids).unsqueeze(1).contiguous()
        scales = self.weight_scale.index_select(0, flat_ids).unsqueeze(1).contiguous()
        globals_ = self.weight_global.index_select(0, flat_ids).unsqueeze(1).contiguous()
        rows = dequant_nvfp4(
            packed, scales, globals_, torch.arange(flat_ids.numel(), device=flat_ids.device),
            dtype=torch.bfloat16,
        ).squeeze(1)
        if self.embed_scale is not None:
            rows = rows * self.embed_scale
        return rows.reshape(*input_ids.shape, -1)


class Nvfp4TiedLMHead(BaseOP):
    def __init__(self, embedding):
        self._embedding = embedding

    def state_dict(self, *, prefix: str = "", result=None):
        return result if result is not None else {}

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        state_dict.pop(f"{prefix}.weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return nvfp4_dense_linear(
            x, self._embedding.weight, self._embedding.weight_scale,
            self._embedding.weight_global,
        )


def convert_gemma4_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace gemma4's dense projections + embedding with native GGUF ops.

    Quantized in the checkpoint -> swapped: attention qkv/o, shared-MLP gate_up/down
    (all Q4_0) and the token embedding (native GGUF type, also the tied LM head). Left as dense
    bf16 (F32 in the GGUF): the router gate, all RMSNorms, the per-layer scalars, and
    the routed experts (served from the offload cache).
    """
    if config.moe_weight_format == "nvfp4":
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear

        inner = model.model
        if config.gguf_embedding_bf16:
            from freetoken.layers import ParallelLMHead, VocabParallelEmbedding

            embed = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, config.embedding_scale
            )
            inner.embed_tokens = embed
            model.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                tie_word_embeddings=True,
                tied_embedding=embed,
            )
        else:
            embed = Nvfp4Embedding(config.vocab_size, config.hidden_size, config.embedding_scale)
            inner.embed_tokens = embed
            model.lm_head = Nvfp4TiedLMHead(embed)
        if not config.gguf_dense_bf16:
            for layer in inner.layers.op_list:
                layer.self_attn.qkv_proj = Nvfp4DenseColMerged(
                    config.hidden_size,
                    [layer.self_attn.q_dim, layer.self_attn.kv_dim, layer.self_attn.kv_dim],
                )
                layer.self_attn.o_proj = Nvfp4DenseLinear(
                    layer.self_attn.q_dim, config.hidden_size
                )
        return

    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    from freetoken.models.gguf.reader import iter_gguf_tensors

    tensor_types = {t.name: t.ggml_type for t in iter_gguf_tensors(config.gguf_model_path)}

    def tensor_type(name: str) -> int:
        try:
            return tensor_types[name]
        except KeyError as exc:
            raise ValueError(f"GGUF tensor type is missing for {name}") from exc

    def swap_linear(owner, attr, quant_type):
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(in_features, out_features, quant_type, has_bias=lin.bias is not None),
        )

    inner = model.model
    embedding_type = config.gguf_embedding_type
    if embedding_type is None:
        raise ValueError("GGUF embedding quantization type is missing from the model config")
    embed = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=embedding_type,
        embed_scale=config.embedding_scale,
    )
    inner.embed_tokens = embed

    for layer_id, layer in enumerate(inner.layers.op_list):
        prefix = f"blk.{layer_id}."
        swap_linear(layer.self_attn, "qkv_proj", tensor_type(prefix + "attn_q.weight"))
        swap_linear(layer.self_attn, "o_proj", tensor_type(prefix + "attn_output.weight"))
        swap_linear(
            layer.feed_forward.shared_mlp,
            "gate_up_proj",
            tensor_type(prefix + "ffn_gate.weight"),
        )
        swap_linear(
            layer.feed_forward.shared_mlp,
            "down_proj",
            tensor_type(prefix + "ffn_down.weight"),
        )

    if config.tie_word_embeddings:
        model.lm_head = GGUFTiedLMHead(embed, embedding_type)


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native Q4_0) for the offload cache.
# --------------------------------------------------------------------------------------

def _q4_0_expert_specs(config: ModelConfig, gate_up_type=GGML_Q4_0, down_type=GGML_Q4_0) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, gate_up_type)), torch.uint8),
        "down": ((E, H, row_bytes(I, down_type)), torch.uint8),
    }


def load_q4_0_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None, quant_type=GGML_Q4_0,
    down_quant_type=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native Q4_0 block bytes.

    ``gate_up`` is one ``[E, 2I, H//32*18]`` tensor per layer and ``down`` one
    ``[E, H, I//32*18]`` per layer (independent :class:`HostBank` allocations) -- each
    expert's packed rows verbatim from the GGUF (no dequant), whole layers arriving in
    one shot (gate_up + down = 2 writes/layer), so the offload cache streams whole
    experts to the ggml MoE kernels.

    ``layer_sink=None`` (serving): pin each layer's two banks as they complete via an
    internally-owned :class:`PinPipeline` (or, on a CUDA-less host, allocate the mmap
    banks but never pin -- the CPU executor reads them pageable). ``layer_sink`` given
    (converter): the completion tracker fires into it instead -- nothing here is pinned,
    and the sink may release banks it has written out, so the returned tensors are only
    valid until then (the caller owns that tradeoff).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import HostBank, LayerCompletionTracker, PinPipeline

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    down_quant_type = quant_type if down_quant_type is None else down_quant_type
    from freetoken.models.gguf.reader import iter_gguf_tensors

    tensors = {t.name: t for t in iter_gguf_tensors(model_path)}
    gate_up_names = [
        f"blk.{layer}.ffn_gate_up_exps.weight"
        if f"blk.{layer}.ffn_gate_up_exps.weight" in tensors
        else f"blk.{layer}.ffn_gate_exps.weight"
        for layer in range(L)
    ]
    hb = {
        "gate_up": [HostBank((E, 2 * I, tensors[name].row_bytes), torch.uint8) for name in gate_up_names],
        "down": [HostBank((E, H, tensors[f"blk.{layer}.ffn_down_exps.weight"].row_bytes), torch.uint8) for layer in range(L)],
    }
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen_gu, seen_dn = set(), set()
    pending_gu: dict[int, dict[str, torch.Tensor]] = {}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None  # gate_up + down
        for t in tensors.values():
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if t.name.endswith("ffn_gate_up_exps.weight"):
                banks["gate_up"][layer].copy_(t.packed().reshape(E, 2 * I, t.row_bytes))
                seen_gu.add(layer)
            elif t.name.endswith("ffn_gate_exps.weight") or t.name.endswith("ffn_up_exps.weight"):
                slot = "gate" if t.name.endswith("ffn_gate_exps.weight") else "up"
                parts = pending_gu.setdefault(layer, {})
                parts[slot] = t.packed().reshape(E, I, t.row_bytes)
                if len(parts) == 2:
                    banks["gate_up"][layer].copy_(torch.cat([parts["gate"], parts["up"]], dim=1))
                    seen_gu.add(layer)
                    del pending_gu[layer]
            elif t.name.endswith("ffn_down_exps.weight"):
                banks["down"][layer].copy_(t.packed().reshape(E, H, t.row_bytes))
                seen_dn.add(layer)
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
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    want = set(range(L))
    assert seen_gu == want and seen_dn == want, (
        f"missing Q4_0 expert layers: gate_up {sorted(want - seen_gu)}, "
        f"down {sorted(want - seen_dn)}"
    )
    return banks


def load_q8_0_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    return load_q4_0_expert_sources(
        model_path, config, layer_sink=layer_sink, quant_type=GGML_Q8_0
    )


def load_nvfp4_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Convert HF-style GGML NVFP4 expert blocks into native NVFP4 host banks."""
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    E, H, I, L = config.num_experts, config.hidden_size, config.moe_intermediate_size, config.num_layers
    hb = alloc_layer_banks(
        {
            "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
            "gate_up_scale": ((E, 2 * I, H // 16), torch.float8_e4m3fn),
            "gate_up_global": ((E, 2 * I), torch.float16),
            "down_packed": ((E, H, I // 2), torch.uint8),
            "down_scale": ((E, H, I // 16), torch.float8_e4m3fn),
            "down_global": ((E, H), torch.float16),
        },
        L,
    )
    banks = {name: [bank.tensor for bank in layers] for name, layers in hb.items()}
    tensors = {t.name: t for t in iter_gguf_tensors(model_path)}
    legacy = "blk.0.ffn_gate_up_exps.weight" in tensors
    for layer in range(L):
        if legacy:
            gate_up, gate_up_scales = _nvfp4_parts(tensors[f"blk.{layer}.ffn_gate_up_exps.weight"])
            down, down_scales = _nvfp4_parts(tensors[f"blk.{layer}.ffn_down_exps.weight"])
            gate_up_global = torch.ones(2 * I, dtype=torch.float16)
            down_global = torch.ones(H, dtype=torch.float16)
            banks["gate_up_packed"][layer].copy_(gate_up.reshape(E, 2 * I, H // 2))
            banks["gate_up_scale"][layer].copy_(gate_up_scales.reshape(E, 2 * I, H // 16))
            banks["gate_up_global"][layer].copy_(gate_up_global.expand(E, -1))
            banks["down_packed"][layer].copy_(down.reshape(E, H, I // 2))
            banks["down_scale"][layer].copy_(down_scales.reshape(E, H, I // 16))
            banks["down_global"][layer].copy_(down_global.expand(E, -1))
            continue
        for expert in range(E):
            parts = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.language_model.layers.{layer}.experts.{expert}.{projection}"
                packed, scales = _nvfp4_parts(tensors[prefix + ".weight"])
                global_scale = _to_bf16(tensors[prefix + ".weight_scale"]).reshape(-1)
                parts[projection] = (packed, scales, global_scale)
            gate, up, down = parts["gate_proj"], parts["up_proj"], parts["down_proj"]
            banks["gate_up_packed"][layer][expert].copy_(torch.cat((gate[0], up[0])))
            banks["gate_up_scale"][layer][expert].copy_(torch.cat((gate[1], up[1])))
            banks["gate_up_global"][layer][expert].copy_(torch.cat((gate[2], up[2])))
            banks["down_packed"][layer][expert].copy_(down[0])
            banks["down_scale"][layer][expert].copy_(down[1])
            banks["down_global"][layer][expert].copy_(down[2])
    if layer_sink is None and torch.cuda.is_available():
        pin_banks(hb)
    return banks


def dummy_q4_0_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    """Random Q4_0 expert banks shaped like ``load_q4_0_expert_sources`` output."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = config.num_layers
    hb = alloc_layer_banks(_q4_0_expert_specs(config), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)  # match the other dummies: pin-after-fill (no-op mmap fill on CPU-only)
    return banks


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_gemma4_to_gguf",
    "is_gguf_model",
    "load_q4_0_expert_sources",
    "load_q8_0_expert_sources",
    "load_nvfp4_expert_sources",
    "dummy_q4_0_expert_sources",
]
