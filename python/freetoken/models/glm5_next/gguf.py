"""GLM-5.3-Flash from a llama.cpp ``glm5-next`` GGUF (unsloth ``GLM-5.3-Flash-GGUF``): config from the GGUF metadata, tensor names, and the converter's transforms undone.

The GGUF geometry is identical to the HF checkpoint (34 KDA linear-attention layers,
11 NoPE-MLA/DSA layers with a kpool-compressed indexer on a 3+1 hybrid block layout,
mHC x4 residual streams, 288-expert sigmoid/noaux_tc MoE with a shared expert and
clamped SwiGLU), so ``parse_gguf_config`` derives a text-config-shaped namespace from
the GGUF KV metadata and reuses ``glm5_next.config.parse_config`` wholesale.

The llama.cpp converter (unsloth's glm5-next writer; tensor names identical across
the ``glm5-next`` and the older ``glm5next`` arch spellings) stores KDA's ``A_log``
pre-transformed as ``-exp(A_log)`` -- inverted at load -- and splits the MLA
``kv_b_proj`` into per-head 3-D ``attn_k_b`` / ``attn_v_b`` tensors (k stored
transposed, [heads, kv_lora, nope] next to [heads, v, kv_lora]); the halves may be
quantized (dynamic quants store them Q8_0), so they dequantize and fuse into the
dense layout the module's MLA absorption reads. The trailing MTP block (``blk.N.nextn.*``, ``nextn_predict_layers == 1``) is
dropped exactly like the HF loader drops the never-read MTP layer; routed experts
serve from the GGUF expert banks (``freetoken.models.gguf.experts``).
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, gguf_tensor_names
from freetoken.models.gguf.tensors import check_scales, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

# HF config fields the GGUF metadata does not carry (GLM-5.3-Flash's config.json values).
_FAMILY_DEFAULTS = dict(
    hidden_act="silu",
    attention_bias=False,
    # NoPE means no rope on the main attention; the indexer's interleaved rope is a
    # served-model fact the GGUF KV predates, and getting it wrong scores the indexer
    # keys incorrectly (DSA retrieval degrades to garbage).
    indexer_rope_interleave=True,
)

# checkpoint module leaf (glm5_next HF naming, language-model rooted) -> GGUF tensors
_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    # KDA input projections (the six leaves of the fused in_proj)
    "self_attn.q_proj": ("attn_q",),
    "self_attn.k_proj": ("attn_k",),
    "self_attn.v_proj": ("attn_v",),
    "self_attn.b_proj": ("ssm_beta",),
    "self_attn.f_a_proj": ("ssm_f_a",),
    "self_attn.g_a_proj": ("ssm_g_a",),
    "self_attn.f_b_proj": ("ssm_f_b",),
    "self_attn.g_b_proj": ("ssm_g_b",),
    # both layer kinds name their output projection the same
    "self_attn.o_proj": ("attn_output",),
    # DSA (MLA) projections
    "self_attn.q_a_proj": ("attn_q_a",),
    "self_attn.q_b_proj": ("attn_q_b",),
    "self_attn.kv_a_proj_with_mqa": ("attn_kv_a_mqa",),
    # two tensors, dequantized and fused at load: the module keeps its dense bf16 kv_b
    # (exempted in the registry's unquantized_modules, so quantized halves are fine too)
    "self_attn.kv_b_proj": ("attn_k_b", "attn_v_b"),
    # kpool indexer
    "self_attn.indexer.wq_b": ("indexer.attn_q_b",),
    "self_attn.indexer.wk": ("indexer.attn_k",),
    "self_attn.indexer.weights_proj": ("indexer.proj",),
    # dense MLP of the leading blocks
    "mlp.gate_proj": ("ffn_gate",),
    "mlp.up_proj": ("ffn_up",),
    "mlp.down_proj": ("ffn_down",),
    # shared expert of the MoE blocks
    "mlp.shared_experts.gate_proj": ("ffn_gate_shexp",),
    "mlp.shared_experts.up_proj": ("ffn_up_shexp",),
    "mlp.shared_experts.down_proj": ("ffn_down_shexp",),
    # routed experts (GGUF expert banks)
    "mlp.experts.0.gate_proj": ("ffn_gate_exps",),
    "mlp.experts.0.up_proj": ("ffn_up_exps",),
    "mlp.experts.0.down_proj": ("ffn_down_exps",),
}
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    # checkpoint-space names: the language-model root is already applied when to_gguf runs
    "model.language_model.embed_tokens": ("token_embd",),
    "lm_head": ("output",),
}
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return tuple(f"blk.{m.group(1)}.{leaf}" for leaf in _LAYER_LEAVES.get(m.group(2), ()))


def _layer_types(shim: "GgufConfigShim", num_layers: int) -> tuple[str, ...]:
    """Per-decoder-layer kinds: DSA where the MLA projections sit, KDA elsewhere, from the
    per-block ``attention.head_count_kv`` list (0 on the KDA blocks, 1 on the MLA ones).

    Every public glm5-next GGUF carries the list (the generic scalar spelling cannot express
    the hybrid; llama.cpp broadcasts it and then fails tensor creation), so anything else is
    refused rather than guessed at."""
    from .args import DSA_LAYER, KDA_LAYER

    per_block = shim.metadata.get(f"{shim.model_type}.attention.head_count_kv")
    if not isinstance(per_block, list) or len(per_block) < num_layers:
        raise ValueError(
            f"{shim.model_path}: glm5-next GGUF lacks the per-block attention.head_count_kv list "
            "the KDA/MLA hybrid layout needs (reconvert with a current llama.cpp)"
        )
    return tuple(DSA_LAYER if int(kv) else KDA_LAYER for kv in per_block[:num_layers])


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    md = shim.metadata
    prefix = shim.model_type

    check_scales(shim.model_path)

    def kv(key: str, default=None):
        return md.get(f"{prefix}.{key}", default)

    def need(key: str):
        val = kv(key)
        if val is None:
            raise KeyError(f"missing GGUF metadata key {prefix}.{key}")
        return val

    # block_count also counts the MTP block a GGUF carries after the decoder
    n_mtp = int(kv("nextn_predict_layers", 0) or 0)
    num_layers = int(need("block_count")) - n_mtp
    layer_types = _layer_types(shim, num_layers)
    first_dense = int(need("leading_dense_block_count"))

    heads = int(need("attention.head_count"))
    qk_nope, v_head = int(need("attention.key_length_mla")), int(need("attention.value_length_mla"))
    # NoPE: the main attention carries no rotary dims (rope survives only in the indexer)
    if int(kv("rope.dimension_count", 0) or 0) != 0:
        raise NotImplementedError("glm5-next GGUF: rotary dims on the main attention are not served")

    # KDA head count: one q/k/v head set of kda.head_dim each; the conv channels or the
    # beta rows count them, falling back to the shared head_count (the two coincide on
    # every release) when the tensor table is gone (metadata-only GGUF).
    kda_head_dim = int(need("kda.head_dim"))
    index = gguf_tensor_index(shim.model_path)
    conv_q = next((i for i in index.values() if i.name.endswith("ssm_conv1d_q.weight")), None)
    kda_heads = conv_q.ne[2] // kda_head_dim if conv_q is not None else heads

    kpool = int(kv("attention.indexer.kpool", 1) or 1)
    # the compressor tensors exist exactly when the indexer pools; a metadata-only GGUF
    # keeps the fact only through kpool > 1
    names = gguf_tensor_names(shim.model_path)
    kpool_compress = any(n.endswith("indexer_compressor_gate.weight") for n in names) if names else kpool > 1
    indexer_types_kv = kv("attention.indexer.types")
    if indexer_types_kv is not None and not all(indexer_types_kv):
        # llama.cpp reads False as "no full indexer on this layer", which the model cannot express
        raise NotImplementedError(f"{shim.model_path}: glm5-next GGUF with layers that lack a full indexer is not served")
    indexer_types = ("full",) * num_layers

    # the clamp is per-layer in the GGUF (and the shared expert carries its own array); the model
    # takes one value, so anything non-uniform is refused instead of silently taking the first
    clamp = need("swiglu_clamp_exp")
    clamp_vals = list(clamp) if isinstance(clamp, list) else [clamp]
    clamp_shexp = kv("swiglu_clamp_shexp")
    shexp_vals = list(clamp_shexp) if isinstance(clamp_shexp, list) else [clamp_shexp] if clamp_shexp is not None else []
    if len(set(map(float, clamp_vals + shexp_vals))) != 1:
        raise NotImplementedError(f"{shim.model_path}: non-uniform swiglu_clamp_exp / swiglu_clamp_shexp is not served")
    if int(kv("expert_gating_func", 2) or 2) != 2:
        raise NotImplementedError(f"{shim.model_path}: expert_gating_func other than sigmoid (2) is not served")
    text = SimpleNamespace(
        **_FAMILY_DEFAULTS,
        num_hidden_layers=num_layers,
        layer_types=tuple(layer_types),
        mlp_layer_types=("dense",) * first_dense + ("sparse",) * (num_layers - first_dense),
        first_k_dense_replace=first_dense,
        hidden_size=int(need("embedding_length")),
        num_attention_heads=heads,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(need("feed_forward_length")),
        rms_norm_eps=float(need("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(need("context_length")),
        q_lora_rank=int(need("attention.q_lora_rank")),
        kv_lora_rank=int(need("attention.kv_lora_rank")),
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=0,
        v_head_dim=v_head,
        mla_use_nope=True,
        index_n_heads=int(need("attention.indexer.head_count")),
        index_head_dim=int(need("attention.indexer.key_length")),
        index_topk=int(need("attention.indexer.top_k")),
        indexer_types=indexer_types,
        index_kpool=kpool,
        index_kpool_compress=kpool_compress,
        # old converters predate the key; both released checkpoints set it true
        index_kpool_always_select_tail=bool(kv("attention.indexer.kpool_select_tail", True)),
        linear_attn_config={
            "num_heads": kda_heads,
            "head_dim": kda_head_dim,
            "short_conv_kernel_size": int(need("ssm.conv_kernel")),
            "gate_lower_bound": float(need("kda.gate_lower_bound")),
        },
        mhc=True,
        hc_mult=int(need("hyper_connection.count")),
        hc_sinkhorn_iters=int(need("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(need("hyper_connection.epsilon")),
        n_routed_experts=int(need("expert_count")),
        n_shared_experts=int(kv("expert_shared_count", 1) or 1),
        num_experts_per_tok=int(need("expert_used_count")),
        moe_intermediate_size=int(need("expert_feed_forward_length")),
        norm_topk_prob=bool(need("expert_weights_norm")),
        routed_scaling_factor=float(need("expert_weights_scale")),
        n_group=int(kv("expert_group_count", 1) or 1),
        topk_group=int(kv("expert_group_used_count", 1) or 1),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        swiglu_limit=float(clamp_vals[0]),
    )
    hf_like = SimpleNamespace(
        text_config=text,
        model_type="glm5_next",
        architectures=list(shim.architectures),
        vision_config=None,
        image_token_id=None,
    )
    config = parse_config(hf_like)
    return replace(config, expert_quant="gguf")


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------

def _float(info: GgufTensorInfo) -> torch.Tensor:
    """A float tensor in torch order, dequantizing packed ggml rows: dynamic quants quantize
    tensors the module consumes as floats (the kv_b halves, the mHC mixers, the compressor gate).
    Trailing unit dims in ``ne`` are dropped -- some writers ship the conv kernels 4-D
    (llama.cpp's glm5-next loader tries that shape first) and a trailing 1 carries no elements."""
    ne = info.ne
    while len(ne) > 1 and ne[-1] == 1:
        ne = ne[:-1]
    if not is_quantized(info.ggml_type):
        from freetoken.models.gguf.tensors import float_tensor

        return float_tensor(info).reshape(tuple(reversed(ne)))
    from freetoken.third_party.gguf import quants
    from freetoken.third_party.gguf.constants import GGMLQuantizationType

    flat = quants.dequantize(np.array(info.data()), GGMLQuantizationType(info.ggml_type))
    return torch.from_numpy(flat).view(*reversed(ne))


def _kv_b(kb: GgufTensorInfo, vb: GgufTensorInfo, num_heads: int, kv_lora: int) -> torch.Tensor:
    """Per-head k [kv_lora, nope] (stored transposed) + v [v, kv_lora] -> the module's fused dense [heads * (nope + v), kv_lora]."""
    k = _float(kb).transpose(1, 2)  # [heads, nope, kv_lora]
    fused_w = torch.cat([k, _float(vb)], dim=1)
    assert fused_w.shape[0] == num_heads, (fused_w.shape, num_heads)
    return fused_w.reshape(num_heads * fused_w.shape[1], kv_lora).to(torch.bfloat16).contiguous()


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF; the routed experts load as offload banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    args = config.glm5_args
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def bf16(name: str) -> torch.Tensor:
        return _float(t(name)).to(torch.bfloat16)

    yield from fused("model.embed_tokens", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    yield from fused("lm_head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", bf16("output_norm")

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", bf16(f"{blk}.attn_norm")
        yield f"{pre}.post_attention_layernorm.weight", bf16(f"{blk}.ffn_norm")
        # mHC mixing tensors, fp32 on every layer
        for sub in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                yield f"{pre}.hc_{sub}_{part}", _float(t(f"{blk}.hc_{sub}_{part}"))

        if args.is_kda_layer(layer):
            sa = f"{pre}.self_attn"
            p, h, d = args.linear_num_heads * args.linear_head_dim, args.linear_num_heads, args.linear_head_dim
            group = [t(f"{blk}.{n}") for n in ("attn_q", "attn_k", "attn_v", "ssm_beta", "ssm_f_a", "ssm_g_a")]
            yield from fused(f"{sa}.in_proj", group, [matrix(i) for i in group], hidden)
            # the GGUF conv kernels already carry the [channels, 1, kernel] singleton dim
            conv = torch.cat([_float(t(f"{blk}.ssm_conv1d_{q}")) for q in ("q", "k", "v")], dim=0)
            yield f"{sa}.conv1d.weight", conv.to(torch.bfloat16)
            for proj, src, k in (("f_b_proj", "ssm_f_b", d), ("g_b_proj", "ssm_g_b", d), ("o_proj", "attn_output", p)):
                info = t(f"{blk}.{src}")
                yield from fused(f"{sa}.{proj}", [info], [matrix(info)], k)
            # the converter stores A_log pre-transformed as -exp(A_log); the kernels want the raw log
            yield f"{sa}.A_log", torch.log(-_float(t(f"{blk}.ssm_a")))
            yield f"{sa}.dt_bias", _float(t(f"{blk}.ssm_dt.bias"))
            yield f"{sa}.o_norm.weight", bf16(f"{blk}.ssm_norm")
        else:
            sa = f"{pre}.self_attn"
            for proj, src, k in (
                ("q_a_proj", "attn_q_a", hidden),
                ("q_b_proj", "attn_q_b", args.q_lora_rank),
                ("kv_a_proj_with_mqa", "attn_kv_a_mqa", hidden),
                ("o_proj", "attn_output", args.num_heads * args.v_head_dim),
            ):
                info = t(f"{blk}.{src}")
                yield from fused(f"{sa}.{proj}", [info], [matrix(info)], k)
            yield f"{sa}.kv_b_proj.weight", _kv_b(
                t(f"{blk}.attn_k_b"), t(f"{blk}.attn_v_b"), args.num_heads, args.kv_lora_rank
            )
            yield f"{sa}.q_a_layernorm.weight", bf16(f"{blk}.attn_q_a_norm")
            yield f"{sa}.kv_a_layernorm.weight", bf16(f"{blk}.attn_kv_a_norm")
            # wq_b serves its checkpoint quant (packed); wk / weights_proj are dense modules
            # (glm_moe_dsa's indexer reasoning), dequantized at load when the file quantized them
            yield from fused(f"{sa}.indexer.wq_b", [t(f"{blk}.indexer.attn_q_b")], [matrix(t(f"{blk}.indexer.attn_q_b"))], args.q_lora_rank)
            yield f"{sa}.indexer.wk.weight", _float(t(f"{blk}.indexer.attn_k")).to(torch.bfloat16)
            yield f"{sa}.indexer.weights_proj.weight", _float(t(f"{blk}.indexer.proj")).to(torch.bfloat16)
            yield f"{sa}.indexer.k_norm.weight", bf16(f"{blk}.indexer.k_norm")
            yield f"{sa}.indexer.k_norm.bias", bf16(f"{blk}.indexer.k_norm.bias")
            yield f"{sa}.indexer.index_kpool_compress_gate", bf16(f"{blk}.indexer_compressor_gate")
            yield f"{sa}.indexer.index_kpool_compress_ape", _float(t(f"{blk}.indexer_compressor_ape"))

        mlp = f"{pre}.mlp"
        if layer < config.first_k_dense_replace:
            for proj, src, k in (
                ("gate_proj", "ffn_gate", hidden),
                ("up_proj", "ffn_up", hidden),
                ("down_proj", "ffn_down", config.intermediate_size),
            ):
                info = t(f"{blk}.{src}")
                yield from fused(f"{mlp}.{proj}", [info], [matrix(info)], k)
        else:
            yield f"{mlp}.gate.weight", bf16(f"{blk}.ffn_gate_inp")
            yield f"{mlp}.e_score_correction_bias", _float(t(f"{blk}.exp_probs_b.bias"))
            shared_i = config.moe_intermediate_size * max(1, config.n_shared_experts)
            for proj, src, k in (
                ("gate_proj", "ffn_gate_shexp", hidden),
                ("up_proj", "ffn_up_shexp", hidden),
                ("down_proj", "ffn_down_shexp", shared_i),
            ):
                info = t(f"{blk}.{src}")
                yield from fused(f"{mlp}.shared_experts.{proj}", [info], [matrix(info)], k)


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
