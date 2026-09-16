"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Supported checkpoints, all in the multimodal-wrapper layout (``model.language_model.*``):
NVFP4 exports (ModelOpt tensor kinds from LibertAIDAI, compressed-tensors kinds from
RedHatAI, selected by ``quantization_config``) and the zai-org block-fp8 release. Not
supported: text-only key layouts, resident routed experts, block-fp8 experts or fp8
projections under TP > 1.

Routed experts go to the offload cache from their NVFP4 or block-fp8 pieces; every
other projection loads as stored (bf16, or fp8 codes with their block scales) with keys
renamed ``model.language_model.X`` -> ``model.X``. The vision tower loads as stored under
``model.visual.X`` -> ``visual.X`` when an encoder is built; the trailing MTP layer is
never read.

Load-time fusions (must mirror the module split orders):

* KDA ``in_proj``    = q|k|v|b projections concatenated on the output axis
* KDA ``in_proj_fg`` = f_a|g_a projections concatenated on the output axis
* KDA ``conv1d``     = q|k|v depthwise conv weights concatenated on the channel axis

Under TP each part is cut to this rank's heads BEFORE it is fused, so the fused tensor is
rank-major within each part, the order the module's splits read.

fp32-kept tensors: ``A_log`` / ``dt_bias``, the mHC ``hc_*`` tensors, the indexer
APE, and the router ``e_score_correction_bias``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.glm_moe_dsa.weight import _ShardReader
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.utils import cached_load_hf_config, div_ceil, div_even, download_hf_weight
from tqdm import tqdm

from .args import Glm5NextArgs
from .config import parse_config

# Checkpoint prefix (multimodal wrapper) -> model prefix.
_CKPT = "model.language_model"
_MODEL = "model"

# MTP-layer experts (layer == num_layers under the full checkpoint) map to None
# alongside the dense prefix; the bank loader skips them.
def _layer_to_bank(layer, config):
    return (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    )


# ModelOpt export (LibertAIDAI/GLM-5.3-Flash-NVFP4): weight | weight_scale |
# weight_scale_2 (dequant-side global).
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts",
)

# llm-compressor export (RedHatAI/GLM-5.3-Flash-NVFP4): weight_packed |
# weight_scale | weight_global_scale (quant-side global -> reciprocal at ingest).
# ``input_global_scale`` (the calibrated W4A4 activation scale) deliberately does
# not match: our routed-expert paths are W4A16 and never quantize activations.
_NVFP4_CT_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight_packed|weight_global_scale|weight_scale)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts (compressed-tensors)",
    kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
    global_reciprocal=True,
)


def _select_expert_source_spec(model_path: str) -> Nvfp4ExpertSourceSpec:
    quant = getattr(cached_load_hf_config(model_path), "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or "").lower()
    return _NVFP4_CT_SOURCE_SPEC if method == "compressed-tensors" else _NVFP4_SOURCE_SPEC

# KDA fusion orders; MUST match Glm5NextKDA's in_proj / in_proj_fg splits.
_KDA_IN_PROJ = ("q_proj", "k_proj", "v_proj", "b_proj")
_KDA_IN_PROJ_FG = ("f_a_proj", "g_a_proj")


def _shard(t: torch.Tensor, dim: int | None) -> torch.Tensor:
    """This rank's contiguous block of ``t`` along ``dim`` in its own storage; ``dim=None`` replicates."""
    tp = get_tp_info()
    if dim is None or tp.size == 1:
        return t
    step = div_even(t.shape[dim], tp.size)
    return t.narrow(dim, tp.rank * step, step).clone(memory_format=torch.contiguous_format)


def _shard_vocab(t: torch.Tensor) -> torch.Tensor:
    """Vocabulary rows split exactly like VocabParallelEmbedding / ParallelLMHead size them."""
    tp = get_tp_info()
    if tp.size == 1:
        return t
    per = div_ceil(t.shape[0], tp.size)
    lo = tp.rank * per
    return t[lo:min(lo + per, t.shape[0])].clone(memory_format=torch.contiguous_format)

# zai-org fp8 release: fp8 codes + fp32 128x128 block scales per expert projection
_FP8_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
)


def _proj(reader, src: str, dst: str, split: int | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """One projection as the checkpoint stores it: bf16, or fp8 codes with their block scales.
    ``split=0`` is column-parallel (output rows), ``split=1`` row-parallel (input columns)."""
    w = reader.get(f"{src}.weight")
    if w.dtype == torch.float8_e4m3fn:
        if split is not None and get_tp_info().size > 1:
            raise NotImplementedError(f"{src}: fp8 projections are not sharded for TP > 1")
        yield f"{dst}.weight", w
        yield f"{dst}.weight_scale_inv", reader.get(f"{src}.weight_scale_inv")
    else:
        yield f"{dst}.weight", _shard(w, split).to(torch.bfloat16)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    return _select_expert_source_spec(model_path)


def _iter_kda_layer(reader, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    # One fused input GEMM: q|k|v|b|f_a|g_a (output-axis concat).
    parts = [reader.get(f"{src}.{p}.weight") for p in _KDA_IN_PROJ + _KDA_IN_PROJ_FG]
    if any(p.dtype == torch.float8_e4m3fn for p in parts):
        raise NotImplementedError("fp8 KDA input projections are not fused by this reader")
    n = len(_KDA_IN_PROJ)
    yield f"{dst}.in_proj.weight", torch.cat([_shard(p, 0).to(torch.bfloat16) for p in parts[:n]], dim=0)
    yield f"{dst}.in_proj_fg.weight", torch.cat([p.to(torch.bfloat16) for p in parts[n:]], dim=0)
    del parts
    # One merged depthwise conv over the q|k|v stream (channel-axis concat).
    conv = torch.cat(
        [_shard(reader.get(f"{src}.{p}_conv1d.weight"), 0).to(torch.bfloat16) for p in ("q", "k", "v")],
        dim=0,
    )
    yield f"{dst}.conv1d.weight", conv
    for p, split in (("f_b_proj", 0), ("g_b_proj", 0), ("o_proj", 1)):
        yield from _proj(reader, f"{src}.{p}", f"{dst}.{p}", split)
    # Gate params stay fp32 (the recurrent kernels read them as fp32); both are per head.
    yield f"{dst}.A_log", _shard(reader.get(f"{src}.A_log"), 0).to(torch.float32)
    yield f"{dst}.dt_bias", _shard(reader.get(f"{src}.dt_bias"), 0).to(torch.float32)
    yield f"{dst}.o_norm.weight", reader.get(f"{src}.o_norm.weight").to(torch.bfloat16)


def _iter_dsa_layer(reader, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    for proj, split in (("q_a_proj", None), ("q_b_proj", 0), ("kv_a_proj_with_mqa", None), ("kv_b_proj", 0), ("o_proj", 1)):
        yield from _proj(reader, f"{src}.{proj}", f"{dst}.{proj}", split)
    for norm in ("q_a_layernorm", "kv_a_layernorm"):
        yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(torch.bfloat16)
    # kpool indexer (every DSA layer owns one). Kept bf16; the APE is fp32.
    for proj in ("wq_b", "wk", "weights_proj"):
        yield f"{dst}.indexer.{proj}.weight", reader.get(
            f"{src}.indexer.{proj}.weight"
        ).to(torch.bfloat16)
    for part, dtype in (
        ("k_norm.weight", torch.bfloat16),
        ("k_norm.bias", torch.bfloat16),
        ("index_kpool_compress_gate", torch.bfloat16),
        ("index_kpool_compress_ape", torch.float32),
    ):
        yield f"{dst}.indexer.{part}", reader.get(f"{src}.indexer.{part}").to(dtype)


def _iter_vision(reader, weight_map: dict) -> Iterator[tuple[str, torch.Tensor]]:
    for name in weight_map:
        if name.startswith("model.visual."):
            yield "visual." + name[len("model.visual.") :], reader.get(name).to(torch.bfloat16)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3 routed experts only serve from the offload cache; they are loaded from their expert pieces."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    args: Glm5NextArgs = config.glm5_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.3 dense weights",
            disable=not primary,
        ):
            src = f"{_CKPT}.layers.{layer}"
            dst = f"{_MODEL}.layers.{layer}"
            if args.is_kda_layer(layer):
                yield from _iter_kda_layer(reader, layer)
            else:
                yield from _iter_dsa_layer(reader, layer)

            # mHC mixing tensors, fp32 on every layer.
            for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                yield f"{dst}.{hc}", reader.get(f"{src}.{hc}").to(torch.float32)

            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(
                    torch.bfloat16
                )

            if layer < config.first_k_dense_replace:
                for proj, split in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                    yield from _proj(reader, f"{src}.mlp.{proj}", f"{dst}.mlp.{proj}", split)
            else:
                yield f"{dst}.mlp.gate.weight", reader.get(f"{src}.mlp.gate.weight").to(
                    torch.bfloat16
                )
                yield (
                    f"{dst}.mlp.e_score_correction_bias",
                    # fp32 like HF's router math (the module declares fp32; a bf16
                    # cast would perturb top-8 selection on fp32-bias checkpoints).
                    reader.get(f"{src}.mlp.gate.e_score_correction_bias").to(torch.float32),
                )
                for proj, split in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                    yield from _proj(reader, f"{src}.mlp.shared_experts.{proj}", f"{dst}.mlp.shared_experts.{proj}", split)

        yield f"{_MODEL}.embed_tokens.weight", _shard_vocab(reader.get(
            f"{_CKPT}.embed_tokens.weight"
        )).to(torch.bfloat16)
        yield f"{_MODEL}.norm.weight", reader.get(f"{_CKPT}.norm.weight").to(torch.bfloat16)
        yield "lm_head.weight", _shard_vocab(reader.get("lm_head.weight")).to(torch.bfloat16)
        if include_vision:
            yield from _iter_vision(reader, weight_map)
    finally:
        reader.close()


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    try:
        yield from _iter_vision(reader, weight_map)
    finally:
        reader.close()


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their ``_scale`` companions; other kinds use the generic readers."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("glm5_next fp8 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    suffix = {"weight": "", "weight_scale_inv": "_scale"}

    def locate(raw_name: str):
        m = _FP8_EXPERT_RE.match(raw_name)
        if m is None:
            return None
        bank = _layer_to_bank(int(m["layer"]), config)
        if bank is None:
            return None
        return bank, int(m["expert"]), m["proj"] + suffix[m["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        folder = download_hf_weight(model_path)
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        reader = _ShardReader(folder, weight_map, torch.device("cpu"))
        try:
            layers = range(config.first_k_dense_replace, config.num_layers)
            for layer in tqdm(layers, desc="Loading GLM-5.3 fp8 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(config.num_experts):
                    base = f"{_CKPT}.layers.{layer}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind_name in suffix:
                            name = f"{base}.{proj}_proj.{kind_name}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_expert_pieces", "nvfp4_expert_spec"]
