"""DFlash 2 draft from a llama.cpp ``dflash`` GGUF: config from the metadata and the state dict at the file's ggml types.

The file holds only the draft: the target's embedding and LM head stay in the target checkpoint, so ``token_embd`` / ``output`` are refused. The linear weights stay in the file's ggml types and the GGML kernels multiply them; the selector codebooks and the K/V context copy dequantize at load.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Iterator

import torch

from freetoken.models.gguf.reader import GgufTensorInfo, gguf_architecture, gguf_tensor_index, load_gguf_metadata
from freetoken.models.gguf.tensors import dequantized_tensor, float_tensor, fused, matrix

from .config import DFlashDraftConfig, parse_config

# checkpoint module leaf -> GGUF tensors (without ".weight") that store it
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    "fc": ("fc",),
    "candidate_selector.hidden_projection": ("selector_hidden",),
}
_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    "self_attn.q_proj": ("attn_q",),
    "self_attn.k_proj": ("attn_k",),
    "self_attn.v_proj": ("attn_v",),
    "self_attn.o_proj": ("attn_output",),
    "mlp.gate_proj": ("ffn_gate",),
    "mlp.up_proj": ("ffn_up",),
    "mlp.down_proj": ("ffn_down",),
    "attention_conv.kernel_projection": ("attn_conv_proj",),
    "mlp_conv.kernel_projection": ("ffn_conv_proj",),
}
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")

# the file's tensor names -> the model's state-dict names; the conv bases and norms carry no ".weight" in the file
_TOP_TENSORS = {
    "fc.weight": "fc.weight",
    "enc.output_norm.weight": "hidden_norm.weight",
    "output_norm.weight": "norm.weight",
    "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
    "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
    "selector_successor.weight": "candidate_selector.successor_codebook",
}
_LAYER_TENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
    "attn_conv_base": "attention_conv.base_kernel",
    "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight",
    "ffn_conv_base": "mlp_conv.base_kernel",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_CODEBOOKS = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
_PACKED = (("qkv_proj", ("q_proj", "k_proj", "v_proj")), ("gate_up_proj", ("gate_proj", "up_proj")))


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the file has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return tuple(f"blk.{m.group(1)}.{leaf}" for leaf in _LAYER_LEAVES.get(m.group(2), ()))


def parse_gguf_config(model_path: str) -> DFlashDraftConfig:
    """The draft's ``.gguf`` (architecture ``dflash``) through the checks of :func:`parse_config`; GGUF metadata does not record the target depth, so ``num_target_layers`` is None."""
    arch = gguf_architecture(model_path)
    if arch != "dflash":
        raise ValueError(f"{model_path}: expected a dflash GGUF draft, got architecture {arch!r}")
    md = load_gguf_metadata(model_path)
    index = gguf_tensor_index(model_path)

    def g(key: str):
        name = f"dflash.{key}"
        if name not in md:
            raise ValueError(f"{model_path}: GGUF metadata lacks {name}")
        return md[name]

    if int(g("attention.value_length")) != int(g("attention.key_length")):
        raise ValueError(f"{model_path}: DFlash drafts need attention.value_length == attention.key_length")
    # llama.cpp defaults a missing causal flag to true, so absence must not serve a causal draft
    if "dflash.attention.causal" not in md:
        raise ValueError(f"{model_path}: GGUF metadata lacks dflash.attention.causal; only non-causal DFlash drafts are supported")
    if md["dflash.attention.causal"]:
        raise ValueError(f"{model_path}: only non-causal (dflash.attention.causal false) DFlash drafts are supported")
    if "dflash.attention.sliding_window_pattern" not in md:
        raise ValueError(f"{model_path}: GGUF metadata lacks dflash.attention.sliding_window_pattern")
    layer_types = ["sliding_attention" if bool(x) else "full_attention" for x in md["dflash.attention.sliding_window_pattern"]]

    present = [key for key in ("attention.scale", "embedding_scale") if f"dflash.{key}" in md]
    if present:
        raise ValueError(f"{model_path}: DFlash drafts with {present} are not supported")
    if "dflash.sample_from_anchor" in md and not md["dflash.sample_from_anchor"]:
        raise ValueError(f"{model_path}: only sample_from_anchor true DFlash drafts are supported")
    if md.get("dflash.has_confidence_head"):
        raise ValueError(f"{model_path}: DFlash drafts with dflash.has_confidence_head are not supported")
    scaling = sorted(key for key in md if key.startswith("dflash.rope.scaling."))
    if scaling:
        raise ValueError(f"{model_path}: DFlash drafts with rope scaling ({scaling[0]}) are not supported")

    taps = [int(i) for i in g("target_layers")]
    if not taps:
        raise ValueError(f"{model_path}: dflash.target_layers is empty")
    # llama.cpp taps are 1-based; the metadata order is the feature column order, so never sort
    target_layer_ids = [i - 1 for i in taps]
    if min(target_layer_ids) < 0:
        raise ValueError(f"{model_path}: dflash.target_layers {taps} must be 1-based")
    if "tokenizer.ggml.mask_token_id" not in md:
        raise ValueError(f"{model_path}: GGUF metadata lacks tokenizer.ggml.mask_token_id")
    # the codebook is per token id, so its row count is the draft's vocabulary; the tokenizer table may pad
    codebook = index.get("selector_predecessor.weight")
    if codebook is None:
        raise ValueError(f"{model_path}: GGUF lacks the selector_predecessor codebook")
    head_dim = int(g("attention.key_length"))
    sections = [int(s) for s in md.get("dflash.rope.dimension_sections", ())]
    # llama.cpp writes [head_dim / 2, 0, 0, 0] for an M-RoPE target; for text that is plain NeoX rope
    if any(sections) and sections != [head_dim // 2] + [0] * (len(sections) - 1):
        raise ValueError(f"{model_path}: DFlash drafts with rope dimension sections {sections} are not supported")
    if "dflash.rope.dimension_count" in md:
        dimension_count = int(md["dflash.rope.dimension_count"])
        if dimension_count <= 0:
            raise ValueError(f"{model_path}: DFlash drafts need a positive dflash.rope.dimension_count, got {dimension_count}")
        if dimension_count != head_dim:
            raise ValueError(f"{model_path}: dflash.rope.dimension_count is {dimension_count}, expected head_dim {head_dim}")

    dc = {
        "block_size": int(g("block_size")),
        "target_layer_ids": target_layer_ids,
        "mask_token_id": int(md["tokenizer.ggml.mask_token_id"]),
        "selector_top_k": int(g("selector_top_k")),
        "selector_rank": int(g("selector_rank")),
        "conv_kernel_size": int(g("conv_kernel_size")),
        "conv_group_size": int(g("conv_group_size")),
    }
    # parse_config's identity guard rejects anything but the defaults these override
    for gguf_key, hf_key in (("logit_scale", "output_multiplier"), ("final_logit_softcapping", "final_logit_softcapping"),
                             ("attention.value_scale", "attention_value_scale")):
        if f"dflash.{gguf_key}" in md:
            dc[hf_key] = md[f"dflash.{gguf_key}"]
    hf = SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        is_causal=False,
        layer_types=layer_types,
        sliding_window=int(g("attention.sliding_window")),
        hidden_size=int(g("embedding_length")),
        head_dim=head_dim,
        rope_parameters={"rope_theta": float(g("rope.freq_base")), "rope_type": "default"},
        num_hidden_layers=int(g("block_count")),
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        intermediate_size=int(g("feed_forward_length")),
        hidden_act=str(md.get("dflash.hidden_activation", "silu")),
        vocab_size=int(codebook.ne[-1]),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        dflash_config=dc,
        num_target_layers=None,
    )
    return parse_config(hf, allow_unknown_depth=True)


def _expected(blocks: int) -> dict[str, str]:
    expected = dict(_TOP_TENSORS)
    for i in range(blocks):
        expected.update({f"blk.{i}.{leaf}": f"layers.{i}.{name}" for leaf, name in _LAYER_TENSORS.items()})
    return expected


def iter_gguf_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The draft's state dict from its GGUF: linears fused exactly as the model names them, at the file's ggml types; ``load`` moves the CPU copies to the device."""
    config = parse_gguf_config(model_path)
    index = gguf_tensor_index(model_path)
    expected = _expected(config.num_layers)
    if missing := sorted(set(expected) - set(index)):
        raise ValueError(f"{model_path}: DFlash tensors missing: {missing}")
    if extra := sorted(set(index) - set(expected)):
        head = [name for name in extra if name in ("token_embd.weight", "output.weight")]
        if head:
            raise ValueError(f"{model_path}: the draft uses the target's embedding and LM head, so {head} must not be in its GGUF")
        raise ValueError(f"{model_path}: unexpected DFlash tensors {extra}")

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def linear(prefix: str, names: tuple[str, ...]) -> Iterator[tuple[str, torch.Tensor]]:
        infos = [t(name) for name in names]
        # the row width of every fused part is the same; the tail padding is per ggml type and per segment
        yield from fused(prefix, infos, [matrix(info) for info in infos], infos[0].ne[0])

    yield from linear("fc", ("fc",))
    yield from linear("candidate_selector.hidden_projection", ("selector_hidden",))
    for hf_name, gguf_name in zip(_CODEBOOKS, ("selector_predecessor", "selector_successor")):
        # a plain parameter in the model, quantized like the matrices in every released variant
        yield hf_name, dequantized_tensor(t(gguf_name))
    yield "hidden_norm.weight", float_tensor(t("enc.output_norm"))
    yield "norm.weight", float_tensor(t("output_norm"))
    for i in range(config.num_layers):
        blk, pre = f"blk.{i}", f"layers.{i}"
        yield f"{pre}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{pre}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.ffn_norm"))
        yield f"{pre}.self_attn.q_norm.weight", float_tensor(t(f"{blk}.attn_q_norm"))
        yield f"{pre}.self_attn.k_norm.weight", float_tensor(t(f"{blk}.attn_k_norm"))
        yield f"{pre}.attention_conv.base_kernel", float_tensor(t(f"{blk}.attn_conv_base"))
        yield f"{pre}.mlp_conv.base_kernel", float_tensor(t(f"{blk}.ffn_conv_base"))
        yield from linear(f"{pre}.self_attn.qkv_proj", (f"{blk}.attn_q", f"{blk}.attn_k", f"{blk}.attn_v"))
        yield from linear(f"{pre}.self_attn.o_proj", (f"{blk}.attn_output",))
        yield from linear(f"{pre}.mlp.gate_up_proj", (f"{blk}.ffn_gate", f"{blk}.ffn_up"))
        yield from linear(f"{pre}.mlp.down_proj", (f"{blk}.ffn_down",))
        yield from linear(f"{pre}.attention_conv.kernel_projection", (f"{blk}.attn_conv_proj",))
        yield from linear(f"{pre}.mlp_conv.kernel_projection", (f"{blk}.ffn_conv_proj",))


def kv_context_weights(model_path: str, config: DFlashDraftConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Every layer's K and V rows ``[num_layers, k_rows + v_rows, hidden]`` for the context bmm; dequantized once, since the ring's projection reads raw weights."""
    index = gguf_tensor_index(model_path)
    rows = [
        torch.cat([dequantized_tensor(index[f"blk.{i}.attn_k.weight"]), dequantized_tensor(index[f"blk.{i}.attn_v.weight"])])
        for i in range(config.num_layers)
    ]
    return torch.stack(rows).to(device=device, dtype=dtype)


def gguf_quant(model_path: str) -> GGUFConfig:
    """The draft's quant over its own tensor table."""
    from freetoken.layers.quantization import GGUFConfig, NameMap

    types = {name: info.ggml_type for name, info in gguf_tensor_index(model_path).items()}
    return GGUFConfig(types, to_gguf, name_map=NameMap(packed=_PACKED))


__all__ = ["gguf_quant", "iter_gguf_weights", "kv_context_weights", "parse_gguf_config", "to_gguf"]
