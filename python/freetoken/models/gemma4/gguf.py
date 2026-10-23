"""Gemma 4 GGUF adapter, including dense and MoE variants, PLE, and shared KV."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch
from tqdm import tqdm

from freetoken.layers import BaseOP

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from freetoken.models.gguf.mmproj import Mmproj, find_mmproj, log_mmproj, mmproj_tensor
from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q8_0, GGML_Q6_K, dequantize, row_bytes
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, load_gguf_metadata
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix, weight_scale

from .config import _parse_vision_config

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _to_bf16(tensor: GgufTensorInfo) -> torch.Tensor:
    return dequantize(tensor.packed().reshape(-1), tensor.ggml_type, torch.bfloat16).reshape(tensor.shape)


def _kv_head_counts(value: Any, num_layers: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        if len(value) != num_layers:
            raise ValueError(
                f"gemma4.attention.head_count_kv has {len(value)} entries, expected {num_layers}"
            )
        counts = tuple(int(item) for item in value)
    else:
        counts = (int(value),) * num_layers
    if any(count <= 0 for count in counts):
        raise ValueError("gemma4.attention.head_count_kv entries must be positive")
    return counts


def _shared_kv_sources(
    swa_pattern: list[bool], shared_kv_layers: int
) -> tuple[int | None, ...]:
    num_layers = len(swa_pattern)
    if not 0 <= shared_kv_layers <= num_layers:
        raise ValueError(
            f"gemma4.attention.shared_kv_layers must be between 0 and {num_layers}"
        )
    if shared_kv_layers == 0:
        return ()
    first_shared = num_layers - shared_kv_layers
    sources: dict[bool, int] = {}
    for layer_id in range(first_shared):
        sources[swa_pattern[layer_id]] = layer_id
    result: list[int | None] = [None] * first_shared
    for layer_id in range(first_shared, num_layers):
        source = sources.get(swa_pattern[layer_id])
        if source is None:
            layer_type = "sliding" if swa_pattern[layer_id] else "full"
            raise ValueError(
                f"Gemma4 shared {layer_type}-attention layer {layer_id} has no "
                "preceding KV source layer of the same attention type"
            )
        result.append(source)
    return tuple(result)


def _native_qweight_rows(rows: torch.Tensor) -> torch.Tensor:
    if rows.dtype != torch.uint8:
        rows = rows.contiguous().view(torch.uint8).reshape(rows.shape[0], -1)
    return rows


def _native_qweight(info: GgufTensorInfo, device) -> torch.Tensor:
    return _native_qweight_rows(matrix(info)).to(device)


def _convert_gemma4_ple_to_gguf(model, config) -> None:
    if not config.per_layer_input_dim:
        return
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear
    from freetoken.models.gguf.reader import GGML_NVFP4, iter_gguf_tensors

    tensor_types = {t.name: t.ggml_type for t in iter_gguf_tensors(config.gguf_model_path)}

    def tensor_type(name: str) -> int:
        value = tensor_types.get(name)
        if value is None and name == "per_layer_token_embd.weight":
            value = config.gguf_per_layer_embedding_type
        if value is None:
            raise ValueError(f"GGUF PLE tensor type is missing for {name}")
        return value

    ple_names = [
        "per_layer_token_embd.weight",
        "per_layer_model_proj.weight",
        *(
            f"blk.{layer_id}.{suffix}.weight"
            for layer_id in range(config.num_layers)
            for suffix in ("inp_gate", "proj")
        ),
    ]
    if any(tensor_type(name) == GGML_NVFP4 for name in ple_names):
        raise NotImplementedError("Gemma4 PLE tensors in NVFP4 format are not supported")

    inner = model.model
    embedding_type = tensor_type("per_layer_token_embd.weight")
    ple_embedding = inner.embed_tokens_per_layer
    if not (
        isinstance(ple_embedding, GGUFEmbedding)
        and ple_embedding._quant_type == embedding_type
    ):
        inner.embed_tokens_per_layer = GGUFEmbedding(
            num_embeddings=config.per_layer_input_vocab_size or config.vocab_size,
            embedding_dim=config.num_layers * config.per_layer_input_dim,
            quant_type=embedding_type,
            embed_scale=config.per_layer_input_dim**0.5,
        )

    def swap_linear(owner, attr: str, name: str) -> None:
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(in_features, out_features, tensor_type(name), has_bias=False),
        )

    swap_linear(
        inner,
        "per_layer_model_projection",
        "per_layer_model_proj.weight",
    )
    for layer_id, layer in enumerate(inner.layers.op_list):
        swap_linear(
            layer,
            "per_layer_input_gate",
            f"blk.{layer_id}.inp_gate.weight",
        )
        swap_linear(
            layer,
            "per_layer_projection",
            f"blk.{layer_id}.proj.weight",
        )


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


# --------------------------------------------------------------------------------------
# Vision: the tower from a llama.cpp mmproj file. gemma4v is the ViT of the 26B-A4B and 31B releases, gemma4uv the linear embedder of the 12B "unified" release.
# --------------------------------------------------------------------------------------

_VIT, _UNIFIED = "gemma4v", "gemma4uv"
_TOKENS = {"image_token_id": "<|image|>", "boi_token_id": "<|image>", "eoi_token_id": "<image|>"}
# llama.cpp hardcodes these and the mmproj does not store them; the values match the HF configs
_POOLING_KERNEL = 3
_ROPE_THETA = 100.0
_MAX_SOFT_TOKENS = 280

_VIT_TOP = {
    "v.position_embd.weight": "vision_tower.patch_embedder.position_embedding_table",
    "v.std_bias": "vision_tower.std_bias",
    "v.std_scale": "vision_tower.std_scale",
    "mm.input_projection.weight": "embed_vision.embedding_projection.weight",
}
_VIT_BLOCK = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ln1": "input_layernorm",
    "attn_post_norm": "post_attention_layernorm",
    "ln2": "pre_feedforward_layernorm",
    "ffn_post_norm": "post_feedforward_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
_VIT_BLOCK_RE = re.compile(r"^v\.blk\.(\d+)\.(\w+)\.weight$")
# tensors over the patch features, stored channel-major
_UNIFIED_PATCH = {
    "v.patch_embd.weight": "vision_embedder.patch_dense.weight",
    "v.patch_norm.1.weight": "vision_embedder.patch_ln1.weight",
    "v.patch_norm.1.bias": "vision_embedder.patch_ln1.bias",
}
_UNIFIED_TOP = {
    "v.patch_embd.bias": "vision_embedder.patch_dense.bias",
    "v.patch_norm.2.weight": "vision_embedder.patch_ln2.weight",
    "v.patch_norm.2.bias": "vision_embedder.patch_ln2.bias",
    "v.patch_norm.3.weight": "vision_embedder.pos_norm.weight",
    "v.patch_norm.3.bias": "vision_embedder.pos_norm.bias",
    "mm.input_projection.weight": "embed_vision.embedding_projection.weight",
}


def _f32_value(v: float) -> float:
    """The shortest decimal of a float32 metadata value: 1e-06, as the HF config writes it, not 9.99999997e-07."""
    return float(str(np.float32(v)))


def gguf_mm_config(text_md: dict[str, Any], mmproj: Mmproj, vocab_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """The HF config.json keys and the preprocessor_config.json of the tower in ``mmproj``, for the text GGUF whose metadata is ``text_md``."""
    if mmproj.projector not in (_VIT, _UNIFIED):
        raise ValueError(f"mmproj {mmproj.path} is a {mmproj.projector!r} tower; Gemma 4 needs {_VIT!r} or {_UNIFIED!r}")
    md, index = load_gguf_metadata(mmproj.path), gguf_tensor_index(mmproj.path)

    def kv(key: str):
        return md[f"clip.vision.{key}"]

    text_hidden = int(text_md["gemma4.embedding_length"])
    if int(kv("projection_dim")) != text_hidden:
        raise ValueError(f"mmproj {mmproj.path} projects to {int(kv('projection_dim'))} but the text model is {text_hidden} wide")
    tokens = text_md["tokenizer.ggml.tokens"]
    missing = [t for t in _TOKENS.values() if t not in tokens]
    if missing:
        raise ValueError(f"the text GGUF tokenizer has no {missing} tokens")
    patch_size = int(kv("patch_size"))
    eps = _f32_value(kv("attention.layer_norm_epsilon"))
    if mmproj.projector == _VIT:
        head_dim = int(index["v.blk.0.attn_q_norm.weight"].ne[0])
        vision_config = {
            "hidden_size": int(kv("embedding_length")),
            "num_hidden_layers": int(kv("block_count")),
            "num_attention_heads": int(kv("attention.head_count")),
            "num_key_value_heads": int(index["v.blk.0.attn_k.weight"].ne[1]) // head_dim,
            "head_dim": head_dim,
            "intermediate_size": int(kv("feed_forward_length")),
            "patch_size": patch_size,
            # ggml order [hidden, table, axis]
            "position_embedding_size": int(index["v.position_embd.weight"].ne[1]),
            "pooling_kernel_size": _POOLING_KERNEL,
            "rms_norm_eps": eps,
            "rope_parameters": {"rope_theta": _ROPE_THETA},
            "hidden_activation": "gelu_pytorch_tanh",
            "standardize": "v.std_bias" in index,
        }
        processor_type = "Gemma4ImageProcessor"
    else:
        # ggml order [patch_dim, hidden]; patch_dim = 3 * model_patch_size**2
        patch_dim = int(index["v.patch_embd.weight"].ne[0])
        vision_config = {
            "mm_embed_dim": int(index["v.patch_embd.weight"].ne[1]),
            "model_patch_size": round((patch_dim // 3) ** 0.5),
            "mm_posemb_size": int(index["v.position_embd.weight"].ne[1]),
            "patch_size": patch_size,
            "pooling_kernel_size": _POOLING_KERNEL,
            "rms_norm_eps": eps,
        }
        processor_type = "Gemma4UnifiedImageProcessor"
    hf_fields = {"vision_config": vision_config, **{key: tokens.index(t) for key, t in _TOKENS.items()}}
    preprocessor = {
        "image_processor_type": processor_type,
        "patch_size": patch_size,
        "pooling_kernel_size": _POOLING_KERNEL,
        "max_soft_tokens": _MAX_SOFT_TOKENS,
        "image_mean": [float(v) for v in kv("image_mean")],
        "image_std": [float(v) for v in kv("image_std")],
        "resample": 3,
        "do_convert_rgb": True,
        "do_resize": True,
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": False,
    }
    return hf_fields, preprocessor


def _hwc(t: torch.Tensor, patch: int) -> torch.Tensor:
    """Patch features from llama.cpp's channel-major order back to the HF processor's (y, x, channel) order, along the last dim."""
    lead = t.shape[:-1]
    return t.reshape(*lead, 3, patch, patch).movedim(-3, -1).reshape(*lead, -1)


def iter_mmproj_weights(mmproj: Mmproj) -> Iterator[tuple[str, torch.Tensor]]:
    """The tower state dict, named as the HF checkpoints load; audio tensors of a unified mmproj are skipped."""
    index = gguf_tensor_index(mmproj.path)
    for name, info in index.items():
        if name.startswith(("a.", "mm.a.")):
            continue
        t = mmproj_tensor(info)
        if mmproj.projector == _VIT:
            if name == "v.patch_embd.weight":
                # [hidden, 3, p, p] -> [hidden, p * p * 3]
                yield "vision_tower.patch_embedder.input_proj.weight", _hwc(t.flatten(1), t.shape[-1])
            elif name in _VIT_TOP:
                yield _VIT_TOP[name], t
            elif (m := _VIT_BLOCK_RE.match(name)) and m[2] in _VIT_BLOCK:
                yield f"vision_tower.encoder.layers.{m[1]}.{_VIT_BLOCK[m[2]]}.weight", t
            else:
                raise ValueError(f"unexpected tensor {name!r} in a {_VIT} mmproj")
            continue
        if name in _UNIFIED_PATCH:
            yield _UNIFIED_PATCH[name], _hwc(t, round((t.shape[-1] // 3) ** 0.5))
        elif name == "v.position_embd.weight":
            # [axis, table, hidden] -> [table, axis, hidden]
            yield "vision_embedder.pos_embedding", t.transpose(0, 1)
        elif name in _UNIFIED_TOP:
            yield _UNIFIED_TOP[name], t
        else:
            raise ValueError(f"unexpected tensor {name!r} in a {_UNIFIED} mmproj")


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    check_scales(
        shim.model_path,
        known=("ffn_gate_inp.scale", "ffn_down_exps.scale", "router.scale"),
    )
    m = shim.metadata

    def g(key: str):
        val = m.get(f"gemma4.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key gemma4.{key}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_experts = int(m.get("gemma4.expert_count", 0))
    num_qo_heads = int(g("attention.head_count"))
    kv_per_layer = _kv_head_counts(g("attention.head_count_kv"), num_layers)
    # True -> sliding-window (SWA) layer, False -> full attention.
    swa_pattern = [bool(x) for x in g("attention.sliding_window_pattern")]
    assert len(swa_pattern) == num_layers, "sliding_window_pattern length != block_count"

    vision_config = _parse_vision_config(shim, hidden)
    swa_layer_ids = tuple(i for i, is_swa in enumerate(swa_pattern) if is_swa)
    full_layer_ids = tuple(i for i, is_swa in enumerate(swa_pattern) if not is_swa)

    swa_head_dim = int(g("attention.key_length_swa"))
    full_head_dim = int(g("attention.key_length"))
    swa_kv = kv_per_layer[swa_layer_ids[0]] if swa_layer_ids else kv_per_layer[0]
    full_kv = kv_per_layer[full_layer_ids[0]] if full_layer_ids else kv_per_layer[0]
    for layer_ids, expected, name in (
        (swa_layer_ids, swa_kv, "sliding"),
        (full_layer_ids, full_kv, "full"),
    ):
        if any(kv_per_layer[layer_id] != expected for layer_id in layer_ids):
            raise ValueError(f"Gemma4 {name}-attention layers have inconsistent KV head counts")

    max_pos = int(g("context_length"))
    shared_kv_layers = int(m.get("gemma4.attention.shared_kv_layers", 0))
    shared_kv_source_layers = _shared_kv_sources(swa_pattern, shared_kv_layers)
    per_layer_input_dim = int(m.get("gemma4.embedding_length_per_layer_input", 0))
    per_layer_vocab_size = int(
        m.get("gemma4.vocab_size_per_layer_input", shim.vocab_size)
    )
    tensor_index = gguf_tensor_index(shim.model_path)
    if per_layer_input_dim and tensor_index:
        required_ple_tensors = (
            "per_layer_token_embd.weight",
            "per_layer_model_proj.weight",
            "per_layer_proj_norm.weight",
            *(
                f"blk.{layer_id}.{suffix}.weight"
                for layer_id in range(num_layers)
                for suffix in ("inp_gate", "proj", "post_norm")
            ),
        )
        missing = [name for name in required_ple_tensors if name not in tensor_index]
        if missing:
            raise ValueError(
                "Gemma4 PLE metadata is present but GGUF tensors are missing: "
                + ", ".join(missing)
            )
        expected_shapes = {
            "per_layer_token_embd.weight": (
                per_layer_vocab_size,
                num_layers * per_layer_input_dim,
            ),
            "per_layer_model_proj.weight": (
                num_layers * per_layer_input_dim,
                hidden,
            ),
            "per_layer_proj_norm.weight": (per_layer_input_dim,),
        }
        for layer_id in range(num_layers):
            expected_shapes.update(
                {
                    f"blk.{layer_id}.inp_gate.weight": (
                        per_layer_input_dim,
                        hidden,
                    ),
                    f"blk.{layer_id}.proj.weight": (
                        hidden,
                        per_layer_input_dim,
                    ),
                    f"blk.{layer_id}.post_norm.weight": (hidden,),
                }
            )
        mismatched = [
            f"{name}: got {tensor_index[name].shape}, expected {shape}"
            for name, shape in expected_shapes.items()
            if tuple(tensor_index[name].shape) != shape
        ]
        if mismatched:
            raise ValueError("Gemma4 PLE tensor shapes are invalid: " + "; ".join(mismatched))
    full_v_presence = [
        f"blk.{layer_id}.attn_v.weight" in tensor_index for layer_id in full_layer_ids
    ]
    if tensor_index and full_v_presence and len(set(full_v_presence)) != 1:
        raise ValueError("Gemma4 full-attention layers disagree on whether V is tied to K")
    k_eq_v = (
        not full_v_presence[0]
        if full_v_presence
        else bool(m.get("gemma4.attention.k_eq_v", shared_kv_layers == 0))
    )
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
        num_experts=num_experts,
        num_experts_per_tok=int(g("expert_used_count")) if num_experts else 0,
        moe_intermediate_size=int(g("expert_feed_forward_length")) if num_experts else 0,
        norm_topk_prob=True,
        model_type="gemma4",
        architectures=list(shim.architectures),
        moe_enabled=num_experts > 0,
        expert_quant=(
            "nvfp4" if hf_nvfp4 else _gguf_expert_quant(shim.model_path)
        ) if num_experts else "none",
        moe_weight_format="nvfp4" if hf_nvfp4 else _gguf_expert_quant(shim.model_path),
        gguf_embedding_bf16=embedding_bf16,
        gguf_embedding_type=embedding_type,
        gguf_model_path=shim.model_path,
        per_layer_input_dim=per_layer_input_dim,
        per_layer_input_vocab_size=per_layer_vocab_size,
        gguf_per_layer_embedding_type=(
            tensor_index["per_layer_token_embd.weight"].ggml_type
            if "per_layer_token_embd.weight" in tensor_index
            else None
        ),
        shared_kv_source_layers=shared_kv_source_layers,
        gguf_router_bf16=router_bf16,
        gguf_dense_bf16=dense_bf16,
        use_qk_norm=True,
        attn_sm_scale=1.0,
        final_logit_softcapping=float(g("final_logit_softcapping")),
        embedding_scale=float(hidden) ** 0.5,
        vision_config=vision_config,
        image_token_id=getattr(shim, "image_token_id", None),
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layer_ids,
                num_kv_heads=full_kv,
                head_dim=full_head_dim,
                rotary_config=full_rotary,
                k_eq_v=k_eq_v,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_layer_ids,
                num_kv_heads=swa_kv,
                head_dim=swa_head_dim,
                rotary_config=swa_rotary,
                sliding_window=int(g("attention.sliding_window")),
                # every Gemma 4 vision release sets use_bidirectional_attention="vision"
                bidirectional_mm_blocks=vision_config is not None,
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
    "per_layer_input_gate": "inp_gate",
    "per_layer_projection": "proj",
    "post_per_layer_input_norm": "post_norm",
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
    if name == "model.embed_tokens_per_layer":
        return ("per_layer_token_embd",)
    if name == "model.per_layer_model_projection":
        return ("per_layer_model_proj",)
    if name == "model.per_layer_projection_norm":
        return ("per_layer_proj_norm",)
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
    include_vision: bool = True,
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
        if include_vision:
            mmproj = find_mmproj(model_path)
            if mmproj is not None:
                yield from iter_mmproj_weights(mmproj)
        return

    assert include_non_moe
    _require_tp1("weight loading")
    config = parse_gguf_config(cached_load_hf_config(model_path))
    assert not include_moe_experts or not config.moe_enabled, (
        "gemma4 GGUF stores experts as Q4_0 and only supports the offload backend; "
        "experts are loaded into the offload cache via load_q4_0_expert_sources()."
    )
    vision = include_vision and config.is_multimodal
    log_mmproj(model_path, vision)
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size

    def t(name: str):
        return index[name if name in index else f"{name}.weight"]

    embed = t("token_embd")
    yield "model.embed_tokens.qweight", _native_qweight(embed, device)
    if "output.weight" in index and not config.tie_word_embeddings:
        out = t("output")
        yield "lm_head.qweight", _native_qweight(out, device)
    yield "model.norm.weight", float_tensor(t("output_norm"))
    if config.per_layer_input_dim:
        ple_embed = t("per_layer_token_embd")
        yield "model.embed_tokens_per_layer.qweight", _native_qweight(ple_embed, device)
        yield "model.per_layer_model_projection.qweight", _native_qweight(
            t("per_layer_model_proj"), device
        )
        yield "model.per_layer_projection_norm.weight", float_tensor(
            t("per_layer_proj_norm")
        )

    def emit_linear(prefix: str, infos: list[GgufTensorInfo]) -> tuple[str, torch.Tensor]:
        types = {info.ggml_type for info in infos}
        row_widths = {matrix(info).shape[1] for info in infos}
        if len(types) != 1 or len(row_widths) != 1:
            raise NotImplementedError(
                f"{prefix}: GGUF fused parts have incompatible quantization layouts"
            )
        rows = torch.cat([_native_qweight_rows(matrix(info)) for info in infos], dim=0)
        return f"{prefix}.qweight", rows.to(device)

    scalar_map = {**_LAYER_SCALAR_MAP, **_MOE_LAYER_SCALAR_MAP} if config.moe_enabled else _LAYER_SCALAR_MAP
    for layer in range(config.num_layers):
        blk, base = f"blk.{layer}", f"model.layers.{layer}"
        for suffix, rel in scalar_map.items():
            tensor = float_tensor(index[f"{blk}.{suffix}"])
            if suffix == "ffn_down_exps.scale" and weight_scale(index[f"{blk}.ffn_down_exps.weight"]) is not None:
                # the converter folds per_expert_scale into the NVFP4 down weight_scale_2, which the expert kernel applies
                tensor = torch.ones_like(tensor)
            yield f"{base}.{rel}", tensor.reshape(1) if rel.endswith("layer_scalar") else tensor
        attn = f"{base}.self_attn"
        fused_qkv = f"{blk}.attn_qkv.weight"
        if fused_qkv in index:
            qkv_infos = [t(f"{blk}.attn_qkv")]
        else:
            qkv_infos = [t(f"{blk}.attn_{part}") for part in ("q", "k")]
            v_name = f"{blk}.attn_v.weight"
            if v_name in index:
                qkv_infos.append(index[v_name])
            elif (
                isinstance(config.attention_group_for_layer(layer), FullAttentionGroupConfig)
                and config.attention_group_for_layer(layer).k_eq_v
            ):
                qkv_infos.append(qkv_infos[-1])
            else:
                raise KeyError(f"missing {v_name} for Gemma4 layer {layer}")
        yield emit_linear(f"{attn}.qkv_proj", qkv_infos)

        out = t(f"{blk}.attn_output")
        yield emit_linear(f"{attn}.o_proj", [out])
        shared = f"{base}.feed_forward.shared_mlp"
        fused_gate_up = f"{blk}.ffn_gate_up.weight"
        if fused_gate_up in index:
            gate_up_infos = [t(f"{blk}.ffn_gate_up")]
        else:
            gate_up_infos = [t(f"{blk}.ffn_gate"), t(f"{blk}.ffn_up")]
        yield emit_linear(f"{shared}.gate_up_proj", gate_up_infos)
        yield emit_linear(f"{shared}.down_proj", [t(f"{blk}.ffn_down")])

        if config.per_layer_input_dim:
            ple_layer = f"{base}"
            yield emit_linear(
                f"{ple_layer}.per_layer_input_gate", [t(f"{blk}.inp_gate")]
            )
            yield emit_linear(
                f"{ple_layer}.per_layer_projection", [t(f"{blk}.proj")]
            )
            yield (
                f"{ple_layer}.post_per_layer_input_norm.weight",
                float_tensor(t(f"{blk}.post_norm")),
            )

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


from freetoken.models.gguf.nvfp4 import nvfp4_parts

_nvfp4_parts = nvfp4_parts


def _rename_hf_nvfp4_tensor_name(name: str) -> str:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "per_layer_token_embd.weight":
        return "model.embed_tokens_per_layer.weight"
    if name == "per_layer_model_proj.weight":
        return "model.per_layer_model_projection.weight"
    if name == "per_layer_proj_norm.weight":
        return "model.per_layer_projection_norm.weight"
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
            "inp_gate.weight": "per_layer_input_gate.weight",
            "proj.weight": "per_layer_projection.weight",
            "post_norm.weight": "post_per_layer_input_norm.weight",
            "ffn_gate.weight": "feed_forward.shared_mlp.gate_proj.weight",
            "ffn_up.weight": "feed_forward.shared_mlp.up_proj.weight",
            "ffn_down.weight": "feed_forward.shared_mlp.down_proj.weight",
            "ffn_gate_inp.weight": "feed_forward.router.proj.weight",
        }
        if suffix in legacy_map:
            return f"{layer_prefix}.{legacy_map[suffix]}"
        if suffix in _LAYER_SCALAR_MAP:
            return f"{layer_prefix}.{_LAYER_SCALAR_MAP[suffix]}"
        if suffix in _MOE_LAYER_SCALAR_MAP:
            return f"{layer_prefix}.{_MOE_LAYER_SCALAR_MAP[suffix]}"
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


def _iter_hf_nvfp4_weights(model_path, device, include_moe_experts, include_non_moe):
    from freetoken.models.gguf.reader import GGML_NVFP4, iter_gguf_tensors, load_gguf_metadata

    tensors = list(iter_gguf_tensors(model_path))
    by_name = {t.name: t for t in tensors}
    merge: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    metadata = load_gguf_metadata(model_path)
    swa_pattern = [bool(x) for x in metadata["gemma4.attention.sliding_window_pattern"]]

    rename = _rename_hf_nvfp4_tensor_name

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
            or name in {
                "token_embd.weight",
                "output_norm.weight",
                "per_layer_token_embd.weight",
                "per_layer_model_proj.weight",
                "per_layer_proj_norm.weight",
            }
        ):
            continue
        if name.endswith(".weight_scale") or name.endswith(".weight_scale_2"):
            continue
        if t.ggml_type != GGML_NVFP4:
            if include_non_moe and ".experts." not in name:
                if name == "per_layer_token_embd.weight":
                    yield "model.embed_tokens_per_layer.qweight", (
                        t.packed().reshape(t.shape[0], t.row_bytes).to(device)
                    )
                    continue
                if name in {"per_layer_model_proj.weight"} or (
                    name.startswith("blk.")
                    and name.endswith((".inp_gate.weight", ".proj.weight"))
                ):
                    target = rename(name).removesuffix(".weight") + ".qweight"
                    yield target, t.packed().reshape(t.shape[0], t.row_bytes).to(device)
                    continue
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
        _convert_gemma4_ple_to_gguf(model, config)
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
    _convert_gemma4_ple_to_gguf(model, config)


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
    from freetoken.distributed import get_tp_info
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
    completed = set()
    pending_gu: dict[int, dict[str, torch.Tensor]] = {}
    expert_tensors = [
        t
        for t in tensors.values()
        if t.name.startswith("blk.")
        and t.name.endswith(
            (
                "ffn_gate_up_exps.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            )
        )
    ]

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None  # gate_up + down
        with tqdm(
            total=L,
            desc="Loading Gemma GGUF expert banks",
            disable=not get_tp_info().is_primary(),
        ) as progress:
            for t in expert_tensors:
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
                if tracker is not None:
                    tracker.note(layer)
                if layer not in completed and layer in seen_gu and layer in seen_dn:
                    completed.add(layer)
                    progress.update()

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
    "gguf_mm_config",
    "iter_mmproj_weights",
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_gemma4_to_gguf",
    "is_gguf_model",
    "load_q4_0_expert_sources",
    "load_q8_0_expert_sources",
    "load_nvfp4_expert_sources",
    "dummy_q4_0_expert_sources",
]
