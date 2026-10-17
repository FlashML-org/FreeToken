"""Gemma 4 GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata.

The GGUF checkpoint's geometry is identical to the HF gemma4 model (hybrid
SWA/full attention with ``k_eq_v`` full layers, 128 routed experts + shared MLP,
final-logit softcap), so this produces the *same* ``ModelConfig`` as
``gemma4.config.parse_config`` -- only the source is GGUF KV metadata instead of a
HF config object. transformers' own GGUF->config conversion is rejected by the
gemma4 strict dataclass (per-layer ``num_key_value_heads`` array), so we read the
metadata directly. Every tensor keeps its packed ggml type (``QuantKind.GGUF``); the
routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from freetoken.models.gguf.reader import gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix, weight_scale

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _full_rotary_dim(shim: "GgufConfigShim", full_head_dim: int) -> int:
    """Rotated width of the full-attention layers, recovered from ``rope_freqs.weight``.

    llama.cpp writes ``rope.dimension_count = full head_dim`` and carries gemma4's
    partial_rotary_factor as a divisor tensor instead: ``[1.0] * n_rot//2 + [1e30] * rest``,
    which drives the unrotated tail's frequencies to zero (ggml ops.cpp ``theta / ff``).
    Count the untouched entries to get the rotated width back.
    """
    info = gguf_tensor_index(shim.model_path).get("rope_freqs.weight")
    if info is not None:
        return int((float_tensor(info) == 1.0).sum().item()) * 2
    # A metadata-only GGUF (an FTW dir's source_metadata.gguf) has no tensor table. Every
    # released gemma4 sets partial_rotary_factor 0.25 on the full layers, and the converter
    # asserts rope_type == "proportional" (llama.cpp conversion/gemma.py).
    return full_head_dim // 4


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    check_scales(shim.model_path, known=("ffn_gate_inp.scale", "ffn_down_exps.scale"))
    m = shim.metadata

    def g(key: str):
        val = m.get(f"gemma4.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key gemma4.{key}")
        return val

    if int(m.get("gemma4.embedding_length_per_layer_input", 0)) or int(m.get("gemma4.attention.shared_kv_layers", 0)):
        raise NotImplementedError("gemma4 E2B / E4B GGUF (per-layer input embeddings, shared KV layers) is not supported")
    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_experts = int(m.get("gemma4.expert_count", 0))
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
        num_experts=num_experts,
        num_experts_per_tok=int(g("expert_used_count")) if num_experts else 0,
        moe_intermediate_size=int(g("expert_feed_forward_length")) if num_experts else 0,
        norm_topk_prob=True,
        model_type="gemma4",
        architectures=list(shim.architectures),
        moe_enabled=num_experts > 0,
        expert_quant="gguf" if num_experts else "none",
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

# Per-layer 1:1 norm/scale tensors (gguf suffix -> freetoken module-relative name), all F32.
_LAYER_SCALAR_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "feed_forward.post_feedforward_layernorm.weight",
    "layer_output_scale.weight": "feed_forward.layer_scalar",
}
# the MoE layers' second feed-forward branch and router
_MOE_LAYER_SCALAR_MAP = {
    "pre_ffw_norm_2.weight": "feed_forward.pre_feedforward_layernorm_2.weight",
    "post_ffw_norm_1.weight": "feed_forward.post_feedforward_layernorm_1.weight",
    "post_ffw_norm_2.weight": "feed_forward.post_feedforward_layernorm_2.weight",
    "ffn_gate_inp.weight": "feed_forward.router.proj.weight",
    "ffn_gate_inp.scale": "feed_forward.router.scale",
    "ffn_down_exps.scale": "feed_forward.router.per_expert_scale",
}

# checkpoint module leaf (gemma4 HF naming) -> GGUF tensor
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "experts.0.gate_proj": "ffn_gate_exps",
    "experts.0.up_proj": "ffn_up_exps",
    "experts.0.down_proj": "ffn_down_exps",
}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name == "model.embed_tokens":
        return ("token_embd",)
    if name == "lm_head":
        return ("output",) if "output.weight" in tensors else ()
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    blk, leaf = f"blk.{m.group(1)}", m.group(2)
    if leaf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj") and f"{blk}.attn_qkv.weight" in tensors:
        return (f"{blk}.attn_qkv",)
    if leaf == "self_attn.v_proj":
        # the full-attention layers store no attn_v: k is reused as v
        return (f"{blk}.attn_v",) if f"{blk}.attn_v.weight" in tensors else (f"{blk}.attn_k",)
    if leaf in ("experts.0.gate_proj", "experts.0.up_proj") and f"{blk}.ffn_gate_up_exps.weight" in tensors:
        return (f"{blk}.ffn_gate_up_exps",)
    tensor = _LAYER_LEAVES.get(leaf)
    return (f"{blk}.{tensor}",) if tensor is not None else ()


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


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    _require_tp1("weight loading")
    config = parse_gguf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size

    def t(name: str):
        return index[name if name in index else f"{name}.weight"]

    embed = t("token_embd")
    yield from fused("model.embed_tokens", [embed], [matrix(embed)], hidden)
    if "output.weight" in index:
        out = t("output")
        yield from fused("lm_head", [out], [matrix(out)], hidden)
    yield "model.norm.weight", float_tensor(t("output_norm"))

    scalar_map = {**_LAYER_SCALAR_MAP, **_MOE_LAYER_SCALAR_MAP} if config.moe_enabled else _LAYER_SCALAR_MAP
    for layer in range(config.num_layers):
        blk, base = f"blk.{layer}", f"model.layers.{layer}"
        for suffix, rel in scalar_map.items():
            tensor = float_tensor(index[f"{blk}.{suffix}"])
            if suffix == "ffn_down_exps.scale" and weight_scale(index[f"{blk}.ffn_down_exps.weight"]) is not None:
                # the converter folds per_expert_scale into the NVFP4 down weight_scale_2, which the expert kernel applies
                tensor = torch.ones_like(tensor)
            yield f"{base}.{rel}", tensor.reshape(1) if rel.endswith("layer_scalar") else tensor
        if f"{blk}.attn_qkv.weight" in index:
            # convert_hf_to_gguf.py --fuse-qkv stores q, k and v as one attn_qkv matrix
            qkv = [t(f"{blk}.attn_qkv")]
        else:
            q, k = t(f"{blk}.attn_q"), t(f"{blk}.attn_k")
            qkv = [q, k, index.get(f"{blk}.attn_v.weight", k)]
        yield from fused(f"{base}.self_attn.qkv_proj", qkv, [matrix(p) for p in qkv], hidden)
        o = t(f"{blk}.attn_output")
        yield from fused(f"{base}.self_attn.o_proj", [o], [matrix(o)], int(o.ne[0]))
        gate, up, down = (t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down"))
        yield from fused(f"{base}.feed_forward.shared_mlp.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], hidden)
        yield from fused(f"{base}.feed_forward.shared_mlp.down_proj", [down], [matrix(down)], int(down.ne[0]))


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "to_gguf",
]
