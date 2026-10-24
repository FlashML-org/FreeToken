"""DeepSeek-V4-Flash from a llama.cpp ``deepseek4`` GGUF (unsloth ``DeepSeek-V4-Flash-0731-GGUF``
and the Vision-Exp release): config from the GGUF metadata, tensor names, and the dense state dict.

The GGUF geometry matches the HF checkpoint (43 MLA blocks with sliding-window sparse
attention, per-layer KV compressors and a Lightning Indexer on the ratio-4 blocks, mHC x4
residual streams, a 256-expert sqrtsoftplus MoE with hash routing on the first blocks and
a shared expert), so ``parse_gguf_config`` builds the family's own :class:`DeepseekV4Args`
from the GGUF KV metadata and reuses the shared ModelConfig construction
(:func:`deepseek_v4.config.model_config_from_args`).

The converter drops the MTP blocks and writes ``attention.compress_ratios`` as a single
placeholder 0, so the per-layer compressor layout is recovered from the tensor table: an
``indexer.*`` tensor means ratio 4, a compressor without an indexer means 128, unless its
``wkv`` carries the CSA overlap's double rows (also ratio 4). Routed experts serve from
the GGUF expert banks (``freetoken.models.gguf.experts``) like every GGUF family.

A ``deepseek4v`` mmproj beside the Vision-Exp text file adds the vision tower under
``visual.``: the converter splits the fused SwiGLU ``w1`` into ``ffn_gate`` / ``ffn_up``
rows and reshapes the patch embedding into the conv layout (both undone at load); ggml-org
writes the attention block fused and unsloth split. A vision trunk also carries the
``exp_probs_b_vl`` router biases and drops the hash layers' never-applied text bias.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.models.gguf.mmproj import Mmproj, find_mmproj, log_mmproj, mmproj_tensor
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, gguf_tensor_names, load_gguf_metadata
from freetoken.models.gguf.tensors import check_scales, fused, matrix
from freetoken.utils import init_logger

from .args import DeepseekV4Args
from .config import IMAGE_TOKEN_ID, model_config_from_args, parse_vision_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

logger = init_logger(__name__)

# engine module leaf (after "model.layers.N.") -> GGUF tensor names (without ".weight")
_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    "attn.wq_a": ("attn_q_a",),
    "attn.wq_b": ("attn_q_b",),
    "attn.wkv": ("attn_kv",),
    "attn.wo_b": ("attn_output_b",),
    "attn.compressor.wkv": ("attn_compressor_kv",),
    "attn.compressor.wgate": ("attn_compressor_gate",),
    "attn.indexer.wq_b": ("indexer.attn_q_b",),
    "attn.indexer.weights_proj": ("indexer.proj",),
    "attn.indexer.compressor.wkv": ("indexer_compressor_kv",),
    "attn.indexer.compressor.wgate": ("indexer_compressor_gate",),
    "ffn.shared_experts.w1": ("ffn_gate_shexp",),
    "ffn.shared_experts.w3": ("ffn_up_shexp",),
    "ffn.shared_experts.w2": ("ffn_down_shexp",),
    # routed experts (GGUF expert banks), gate / up / down order
    "ffn.experts.0.w1": ("ffn_gate_exps",),
    "ffn.experts.0.w3": ("ffn_up_exps",),
    "ffn.experts.0.w2": ("ffn_down_exps",),
}
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    "model.embed": ("token_embd",),
    "model.head": ("output",),
}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Engine module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return tuple(f"blk.{m.group(1)}.{leaf}" for leaf in _LAYER_LEAVES.get(m.group(2), ()))


def _compress_ratios(index: dict[str, GgufTensorInfo], head_dim: int, num_layers: int) -> tuple[int, ...]:
    """Per-layer compressor ratios from the tensor table, which every released file's
    ``attention.compress_ratios`` array (43 trunk entries ahead of 3 MTP zeros, which
    llama.cpp requires) confirms element for element.

    An ``indexer.proj`` tensor exists exactly on the ratio-4 blocks; a compressor without an
    indexer is a 128 block, unless its ``wkv`` has the CSA overlap's double rows (ratio 4)."""
    ratios = []
    for layer in range(num_layers):
        blk = f"blk.{layer}"
        if f"{blk}.indexer.proj.weight" in index:
            ratios.append(4)
            continue
        info = index.get(f"{blk}.attn_compressor_kv.weight")
        ratios.append(0 if info is None else 4 if info.ne[-1] == 2 * head_dim else 128)
    return tuple(ratios)


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

    names = gguf_tensor_names(shim.model_path)
    if any(n.startswith(("vision.", "aligner.")) for n in names):
        raise NotImplementedError(
            f"{shim.model_path}: deepseek4 GGUF with an inline vision tower is not served "
            "(llama.cpp ships the tower as a separate mmproj file)"
        )

    # a deepseek4v mmproj beside the file fills the shim's flat vision fields; a text-only
    # process nulls vision_n_layers on its copy, and None here is the same answer
    vc = parse_vision_config(shim)

    # the clamp is per-layer in the GGUF (and the shared expert carries its own array); the model
    # takes one value, so anything non-uniform is refused instead of silently taking the first
    clamp = need("swiglu_clamp_exp")
    clamp_vals = list(clamp) if isinstance(clamp, list) else [clamp]
    clamp_shexp = kv("swiglu_clamp_shexp")
    shexp_vals = list(clamp_shexp) if isinstance(clamp_shexp, list) else [clamp_shexp] if clamp_shexp is not None else []
    if len(set(map(float, clamp_vals + shexp_vals))) != 1:
        raise NotImplementedError(f"{shim.model_path}: non-uniform swiglu_clamp_exp / swiglu_clamp_shexp is not served")

    from freetoken.third_party.gguf.constants import ExpertGatingFuncType

    if int(need("expert_gating_func")) != int(ExpertGatingFuncType.SQRTSOFTPLUS):
        raise NotImplementedError(f"{shim.model_path}: expert_gating_func other than sqrtsoftplus is not served")

    # every official DeepSeek-V4 config carries the same yarn block (V4-Flash, 0731, Pro,
    # Pro-0813: factor 16 over 65536, betas 32 / 1). The no-yarn files on HF lost the whole
    # block in conversion -- llama.cpp serves them unscaled, which is wrong for them -- so a
    # wholly missing block falls back to the family values and a partly present one refuses.
    scaling = {k: kv(f"rope.scaling.{k}") for k in
               ("type", "factor", "original_context_length", "yarn_beta_fast", "yarn_beta_slow")}
    if scaling["type"] not in (None, "yarn"):
        raise NotImplementedError(f"{shim.model_path}: rope scaling other than yarn is not served")
    missing = [k for k, v in scaling.items() if v is None]
    if missing:
        if len(missing) != len(scaling):
            raise ValueError(
                f"{shim.model_path}: rope.scaling is only partly present "
                f"(missing {', '.join(missing)}); the file is inconsistent"
            )
        logger.warning(
            f"{shim.model_path}: no rope.scaling block; using the family values "
            "(yarn, factor 16, original 65536, betas 32 / 1)"
        )
        scaling = {"type": "yarn", "factor": 16.0, "original_context_length": 65536,
                   "yarn_beta_fast": 32.0, "yarn_beta_slow": 1.0}

    head_dim = int(need("attention.key_length"))
    if int(need("attention.value_length")) != head_dim:
        raise NotImplementedError(f"{shim.model_path}: attention key and value lengths differ")
    if int(kv("attention.head_count_kv", 1) or 1) != 1:
        raise NotImplementedError(f"{shim.model_path}: more than one latent KV head is not served")

    num_layers = int(need("block_count"))
    index = gguf_tensor_index(shim.model_path)

    # REAP exports prune experts per layer: as a per-layer expert_count array, or with the
    # scalar kept but the expert tensors shrunk. llama.cpp builds every layer from one
    # n_expert and refuses both shapes; the model has no per-layer expert count to express.
    experts_kv = need("expert_count")
    if isinstance(experts_kv, list):
        raise NotImplementedError(
            f"{shim.model_path}: a per-layer expert_count array (REAP-style prune) is not served"
        )
    n_experts = int(experts_kv)
    for name, info in index.items():
        if name.endswith(".ffn_gate_exps.weight") and info.ne[-1] != n_experts:
            raise NotImplementedError(
                f"{shim.model_path}: {name} carries {info.ne[-1]} experts but expert_count says "
                f"{n_experts} (REAP-style per-layer prunes are not served)"
            )
    args = DeepseekV4Args(
        max_batch_size=1,
        vocab_size=int(shim.vocab_size),
        dim=int(need("embedding_length")),
        moe_inter_dim=int(need("expert_feed_forward_length")),
        n_layers=num_layers,
        n_hash_layers=int(kv("hash_layer_count", 0) or 0),
        n_heads=int(need("attention.head_count")),
        n_routed_experts=int(need("expert_count")),
        n_shared_experts=int(kv("expert_shared_count", 1) or 1),
        n_activated_experts=int(need("expert_used_count")),
        score_func="sqrtsoftplus",
        route_scale=float(need("expert_weights_scale")),
        swiglu_limit=float(clamp_vals[0]),
        q_lora_rank=int(need("attention.q_lora_rank")),
        head_dim=head_dim,
        rope_head_dim=int(need("rope.dimension_count")),
        norm_eps=float(need("attention.layer_norm_rms_epsilon")),
        o_groups=int(need("attention.output_group_count")),
        o_lora_rank=int(need("attention.output_lora_rank")),
        window_size=int(need("attention.sliding_window")),
        compress_ratios=_compress_ratios(index, head_dim, num_layers),
        compress_rope_theta=float(kv("attention.compress_rope_freq_base", 160000.0) or 160000.0),
        original_seq_len=int(scaling["original_context_length"]),
        rope_theta=float(need("rope.freq_base")),
        rope_factor=float(scaling["factor"]),
        beta_fast=float(scaling["yarn_beta_fast"]),
        beta_slow=float(scaling["yarn_beta_slow"]),
        index_n_heads=int(need("attention.indexer.head_count")),
        index_head_dim=int(need("attention.indexer.key_length")),
        index_topk=int(need("attention.indexer.top_k")),
        hc_mult=int(need("hyper_connection.count")),
        hc_sinkhorn_iters=int(need("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(need("hyper_connection.epsilon")),
        n_mtp_layers=0,  # the converter drops the MTP blocks
        # vision fields only when a mmproj filled them (the defaults are the text model's)
        **({} if vc is None else dict(
            vision_n_layers=vc.vision_n_layers,
            vision_dim=vc.vision_dim,
            vision_n_heads=vc.vision_n_heads,
            vision_inter_dim=vc.vision_inter_dim,
            vision_patch_size=vc.vision_patch_size,
            vision_rope_theta=vc.vision_rope_theta,
            vision_downsample_ratio=vc.vision_downsample_ratio,
            vision_max_n_token=vc.vision_max_n_token,
            vision_min_pixels=vc.vision_min_pixels,
            vision_max_wh_ratio=vc.vision_max_wh_ratio,
        )),
    )
    config = model_config_from_args(args, max_position=int(need("context_length")), vision_config=vc)
    return replace(config, expert_quant="gguf")


# ----------------------------------------------------------------------------------------------
# Vision tower (a deepseek4v mmproj)
# ----------------------------------------------------------------------------------------------

_PROJECTOR = "deepseek4v"
_IMAGE_TOKEN = "<｜deepseek_image｜>"
# no mmproj key carries these; llama.cpp hardcodes them and its converter asserts them
_ROPE_THETA = 10000.0
_MAX_N_TOKEN = 384
_MAX_WH_RATIO = 8

# top-level mmproj tensor -> the tower name under ``visual.``
_MM_TOP: dict[str, str] = {
    "v.patch_embd.bias": "vision.patch_embed.proj.bias",
    "v.post_ln.weight": "vision.norm.weight",
    "mm.1.weight": "aligner.w1.weight",
    "mm.1.bias": "aligner.w1.bias",
    "mm.2.weight": "aligner.w2.weight",
    "mm.2.bias": "aligner.w2.bias",
    "v.token_embd.img_start": "image_start",
    "v.token_embd.img_end": "image_end",
    "v.image_newline": "image_newline",
    "v.token_embd.img_pad": "image_pad",
}


def gguf_mm_config(text_md: dict, mmproj: Mmproj, vocab_size: int, *, model_path: str = "") -> tuple[dict, dict]:
    """The flat HF config keys (the vision dims sit at the top level on this family, like the
    checkpoint's inference config) and the preprocessor dict of the tower in ``mmproj``, for
    the text GGUF whose metadata is ``text_md``."""
    if mmproj.projector != _PROJECTOR:
        raise ValueError(f"mmproj {mmproj.path} is a {mmproj.projector!r} tower; DeepSeek-V4 needs {_PROJECTOR!r}")
    md, index = load_gguf_metadata(mmproj.path), gguf_tensor_index(mmproj.path)
    # the converter writes use_silu=True only for the SwiGLU tower; the module has no gelu MLP
    if md.get("clip.use_silu") is not True:
        raise NotImplementedError(f"mmproj {mmproj.path}: only the SwiGLU tower is served")

    def kv(key: str):
        return md[f"clip.vision.{key}"]

    arch = text_md["general.architecture"]
    text_hidden = int(text_md[f"{arch}.embedding_length"])
    if int(kv("projection_dim")) != text_hidden:
        raise ValueError(f"mmproj {mmproj.path} projects to {int(kv('projection_dim'))} but the text model is {text_hidden} wide")
    tokens = text_md["tokenizer.ggml.tokens"]
    if _IMAGE_TOKEN not in tokens:
        raise ValueError(
            f"the text GGUF tokenizer has no {_IMAGE_TOKEN} token (the 0731 text-only release "
            "uses another dialect; pair the mmproj with a Vision-Exp text GGUF)"
        )
    # the model config and the processor hardcode the family id; a file with the token at
    # another id would fail per request instead of here
    if tokens.index(_IMAGE_TOKEN) != IMAGE_TOKEN_ID:
        raise ValueError(
            f"the text GGUF carries {_IMAGE_TOKEN} at id {tokens.index(_IMAGE_TOKEN)}, not the family's {IMAGE_TOKEN_ID}"
        )
    # a ValueError so auto-discovery skips this mmproj (text only); only --mmproj fails
    if not any(n.endswith(".exp_probs_b_vl.bias") for n in gguf_tensor_names(model_path)):
        raise ValueError(
            f"the text trunk {os.path.basename(model_path)} predates vision routing "
            "(no exp_probs_b_vl; converted before llama.cpp #28154) -- pair a Vision-Exp "
            "text GGUF reconverted after 2026-09-02"
        )
    for sentinel in ("v.token_embd.img_start", "v.token_embd.img_end", "v.image_newline", "v.token_embd.img_pad"):
        if sentinel not in index:
            raise ValueError(f"mmproj {mmproj.path}: sentinel tensor {sentinel!r} is missing")
    # the tower hardcodes eps 1e-6 (vision.py); the converter warns it is NOT the text 1e-20
    if abs(float(kv("attention.layer_norm_epsilon")) - 1e-6) > 1e-9:
        raise NotImplementedError(f"mmproj {mmproj.path}: vision layer_norm_epsilon {kv('attention.layer_norm_epsilon')} is not the served 1e-6")
    # qkv dialect: ggml-org writes the fused attn_qkv, unsloth the split attn_q/k/v (clip.cpp loads both)
    split = all(f"v.blk.0.attn_{q}.{part}" in index for q in "qkv" for part in ("weight", "bias"))
    if split == ("v.blk.0.attn_qkv.weight" in index):
        raise ValueError(f"mmproj {mmproj.path}: the attn tensors are neither the fused attn_qkv nor the split attn_q/k/v layout (or carry both)")
    patch, dim = int(kv("patch_size")), int(kv("embedding_length"))
    # the converter reshapes the Linear's [dim, 3*p*p] kernel into the conv [p, p, 3, dim]
    if tuple(index["v.patch_embd.weight"].ne) != (patch, patch, 3, dim):
        raise ValueError(f"mmproj {mmproj.path}: v.patch_embd.weight {tuple(index['v.patch_embd.weight'].ne)} is not the [p, p, 3, dim] conv layout")
    # clip.vision.image_size (672) is deliberately unread: a converter compat/warmup value,
    # the tower is native-resolution

    hf_fields = {
        "vision_n_layers": int(kv("block_count")),
        "vision_dim": dim,
        "vision_n_heads": int(kv("attention.head_count")),
        "vision_inter_dim": int(kv("feed_forward_length")),
        "vision_patch_size": patch,
        "vision_rope_theta": _ROPE_THETA,
        "vision_downsample_ratio": int(kv("projector.scale_factor")),
        "vision_max_n_token": _MAX_N_TOKEN,
        "vision_min_pixels": int(kv("image_min_pixels")),
        "vision_max_wh_ratio": _MAX_WH_RATIO,
        # parse_vision_config reads the aligner's target width off hidden_size
        "hidden_size": text_hidden,
        "image_token_id": tokens.index(_IMAGE_TOKEN),
    }
    preprocessor = {
        # no transformers class exists; DSV4MMProcessor._image_processor returns None
        "image_processor_type": "DeepseekV4ImageProcessor",
        "patch_size": patch,
        "temporal_patch_size": 1,
        "merge_size": int(kv("projector.scale_factor")),
        "image_mean": [float(v) for v in kv("image_mean")],
        "image_std": [float(v) for v in kv("image_std")],
        "min_pixels": int(kv("image_min_pixels")),
        "max_n_token": _MAX_N_TOKEN,
    }
    return hf_fields, preprocessor


def iter_mmproj_weights(mmproj: Mmproj) -> Iterator[tuple[str, torch.Tensor]]:
    """The tower state dict under ``visual.``, named as the HF checkpoints load."""
    index = gguf_tensor_index(mmproj.path)
    fused = "v.blk.0.attn_qkv.weight" in index
    n_blocks = max(int(n.split(".")[2]) for n in index if n.startswith("v.blk.")) + 1

    pe = index["v.patch_embd.weight"]
    # ggml order [p, p, 3, out] -> torch [out, 3, p, p] -> the module's flat [out, 3*p*p]
    yield "visual.vision.patch_embed.proj.weight", mmproj_tensor(pe).reshape(pe.ne[3], 3 * pe.ne[0] * pe.ne[0])
    for gguf_name, ft_name in _MM_TOP.items():
        yield f"visual.{ft_name}", mmproj_tensor(index[gguf_name])

    consumed = {"v.patch_embd.weight", *_MM_TOP}
    for layer in range(n_blocks):
        blk, dst = f"v.blk.{layer}", f"visual.vision.blocks.{layer}"
        if fused:
            for part in ("weight", "bias"):
                yield f"{dst}.attn.wqkv.{part}", mmproj_tensor(index[f"{blk}.attn_qkv.{part}"])
            consumed.update((f"{blk}.attn_qkv.weight", f"{blk}.attn_qkv.bias"))
        else:
            for part in ("weight", "bias"):
                yield f"{dst}.attn.wqkv.{part}", torch.cat(
                    [mmproj_tensor(index[f"{blk}.attn_{q}.{part}"]) for q in "qkv"], dim=0
                )
            consumed.update(f"{blk}.attn_{q}.{part}" for q in "qkv" for part in ("weight", "bias"))
        for part in ("weight", "bias"):
            yield f"{dst}.attn.wo.{part}", mmproj_tensor(index[f"{blk}.attn_out.{part}"])
        # the converter chunks the fused SwiGLU w1 rows into gate / up; cat them back
        yield f"{dst}.mlp.w1.weight", torch.cat(
            [mmproj_tensor(index[f"{blk}.ffn_gate.weight"]), mmproj_tensor(index[f"{blk}.ffn_up.weight"])], dim=0
        )
        yield f"{dst}.mlp.w2.weight", mmproj_tensor(index[f"{blk}.ffn_down.weight"])
        yield f"{dst}.norm1.weight", mmproj_tensor(index[f"{blk}.ln1.weight"])
        yield f"{dst}.norm2.weight", mmproj_tensor(index[f"{blk}.ln2.weight"])
        consumed.update(
            f"{blk}.attn_out.{part}" for part in ("weight", "bias")
        )
        consumed.update(f"{blk}.ffn_{m}.weight" for m in ("gate", "up", "down"))
        consumed.update((f"{blk}.ln1.weight", f"{blk}.ln2.weight"))

    leftovers = set(index) - consumed
    if leftovers:
        raise ValueError(f"mmproj {mmproj.path}: unexpected tensors {sorted(leftovers)[:5]}")


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------

def _float(info: GgufTensorInfo) -> torch.Tensor:
    """A float tensor in torch order, dequantizing packed ggml rows: dynamic quants quantize
    tensors the module consumes as floats (the mHC mixers, the compressor gates, the router).
    Trailing unit dims in ``ne`` are dropped -- some writers ship tensors 4-D (llama.cpp's
    loaders try that shape first) and a trailing 1 carries no elements."""
    ne = info.ne
    while len(ne) > 1 and ne[-1] == 1:
        ne = ne[:-1]
    if not is_quantized(info.ggml_type):
        from freetoken.models.gguf.tensors import float_tensor

        return float_tensor(info).view(*reversed(ne))
    from freetoken.third_party.gguf import quants
    from freetoken.third_party.gguf.constants import GGMLQuantizationType

    flat = quants.dequantize(np.array(info.data()), GGMLQuantizationType(info.ggml_type))
    return torch.from_numpy(flat).view(*reversed(ne))


def _tid2eid(info: GgufTensorInfo) -> torch.Tensor:
    """The hash-route table as the module's int64 [vocab, top_k]."""
    ne = info.ne
    while len(ne) > 1 and ne[-1] == 1:
        ne = ne[:-1]
    raw = np.frombuffer(info.data(), dtype=np.int32).copy()
    return torch.from_numpy(raw).view(*reversed(ne)).to(torch.int64)


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF, in the names the HF loader yields; the routed experts
    load as offload banks (``freetoken.models.gguf.experts``)."""
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    vision = include_vision and config.is_multimodal
    log_mmproj(model_path, vision)
    args = config.dsv4_args
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def bf16(name: str) -> torch.Tensor:
        return _float(t(name)).to(torch.bfloat16)

    yield from fused("model.embed", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    yield from fused("model.head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", bf16("output_norm")
    for part in ("fn", "base", "scale"):
        yield f"model.hc_head_{part}", _float(t(f"output_hc_{part}"))

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        m = f"{pre}.attn"
        yield f"{pre}.attn_norm.weight", bf16(f"{blk}.attn_norm")
        yield f"{pre}.ffn_norm.weight", bf16(f"{blk}.ffn_norm")
        # mHC mixing tensors, fp32 on every layer
        for sub in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                yield f"{pre}.hc_{sub}_{part}", _float(t(f"{blk}.hc_{sub}_{part}"))

        for proj, src, k in (
            ("wq_a", "attn_q_a", hidden),
            ("wq_b", "attn_q_b", args.q_lora_rank),
            ("wkv", "attn_kv", hidden),
            ("wo_b", "attn_output_b", args.o_groups * args.o_lora_rank),
        ):
            info = t(f"{blk}.{src}")
            yield from fused(f"{m}.{proj}", [info], [matrix(info)], k)
        yield f"{m}.q_norm.weight", bf16(f"{blk}.attn_q_a_norm")
        yield f"{m}.kv_norm.weight", bf16(f"{blk}.attn_kv_a_norm")
        # wo_a feeds a bf16 einsum; the HF path dequantizes its fp8 the same way
        yield f"{m}.wo_a", _float(t(f"{blk}.attn_output_a")).to(torch.bfloat16)
        yield f"{m}.attn_sink", _float(t(f"{blk}.attn_sinks"))

        ratio = args.compress_ratios[layer]
        if ratio:
            compressors = ((f"{m}.compressor", f"{blk}.attn_compressor"),)
            if ratio == 4:
                compressors += ((f"{m}.indexer.compressor", f"{blk}.indexer_compressor"),)
            for c_dst, c_src in compressors:
                yield f"{c_dst}.ape", _float(t(f"{c_src}_ape"))
                # the compressors read wkv / wgate through a dense bf16 einsum, not the quant
                # method (the registry exempts them, like the HF spec), so a tensor the file
                # quantized dequantizes here
                for proj, suf in (("wkv", "kv"), ("wgate", "gate")):
                    yield f"{c_dst}.{proj}.weight", bf16(f"{c_src}_{suf}")
                yield f"{c_dst}.norm.weight", bf16(f"{c_src}_norm")
            if ratio == 4:
                for proj, src, k in (
                    ("indexer.wq_b", "indexer.attn_q_b", args.q_lora_rank),
                    ("indexer.weights_proj", "indexer.proj", hidden),
                ):
                    info = t(f"{blk}.{src}")
                    yield from fused(f"{m}.{proj}", [info], [matrix(info)], k)

        g = f"{pre}.ffn.gate"
        yield f"{g}.weight", bf16(f"{blk}.ffn_gate_inp")
        if layer < args.n_hash_layers:
            yield f"{g}.tid2eid", _tid2eid(t(f"{blk}.ffn_gate_tid2eid"))
            if vision:
                # the converter drops the hash layers' text bias (the reference never applies
                # it); the module builds the slot on a vision build, so it zero-fills
                yield f"{g}.bias", torch.zeros(args.n_routed_experts, dtype=torch.float32)
        else:
            yield f"{g}.bias", _float(t(f"{blk}.exp_probs_b.bias"))
        if vision:
            yield f"{g}.bias_vl", _float(t(f"{blk}.exp_probs_b_vl.bias"))
        for proj, src in (("w1", "ffn_gate_shexp"), ("w3", "ffn_up_shexp"), ("w2", "ffn_down_shexp")):
            info = t(f"{blk}.{src}")
            k = hidden if proj != "w2" else args.moe_inter_dim
            yield from fused(f"{pre}.ffn.shared_experts.{proj}", [info], [matrix(info)], k)

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


__all__ = ["gguf_mm_config", "iter_gguf_weights", "iter_mmproj_weights", "parse_gguf_config", "to_gguf"]
