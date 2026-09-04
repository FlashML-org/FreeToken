"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Supported checkpoints: NVFP4 exports of GLM-5.3-Flash in the multimodal-wrapper
layout (``model.language_model.*``) -- ModelOpt tensor kinds (LibertAIDAI) or
compressed-tensors kinds (RedHatAI), selected by ``quantization_config``. Not
supported: bf16-expert originals (zai-org), text-only key layouts, TP > 1.

Routed experts go to the offload cache via ``load_nvfp4_expert_sources``;
everything else loads bf16 with keys renamed ``model.language_model.X`` ->
``model.X``. ``model.visual.*`` and the trailing MTP layer are never read.

Load-time fusions (must mirror the module split orders):

* KDA ``in_proj``  = q|k|v|b|f_a|g_a projections concatenated on the output axis
* KDA ``conv1d``   = q|k|v depthwise conv weights concatenated on the channel axis

fp32-kept tensors: ``A_log`` / ``dt_bias``, the mHC ``hc_*`` tensors, the indexer
APE, and the router ``e_score_correction_bias``. Optional W8A16 fp8-at-load
follows ``ModelConfig.attn_quant`` / ``dense_quant`` / ``lm_head_quant``
(defaults and env opt-ins: see config.py).
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.glm_moe_dsa.weight import _ShardReader, _quant_fp8_per_row
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
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

# KDA in_proj fusion order; MUST match Glm5NextKDA._in_proj_split.
_KDA_IN_PROJ = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")



def nvfp4_expert_source_spec(model_path: str, config):
    """The source spec the disk tier must index this checkpoint with.

    Same object the loader passes to ``load_nvfp4_expert_source_banks``, exposed so the
    shared provider can build the disk index without knowing the family: the index has to
    read the rows the loader placed, so one spec has to serve both."""
    return _select_expert_source_spec(model_path)

def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None, disk_tier=None):
    # disk_tier is threaded through the same way qwen3_5_moe does it: the caller builds the
    # NVMe tier and every NVFP4 family's loader has to pass it down, or the tail of the bank
    # is never registered and load fails with an unexpected-kwarg TypeError.
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _select_expert_source_spec(model_path),
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
        disk_tier=disk_tier,
    )


def _maybe_fp8(key: str, w: torch.Tensor, fp8: bool):
    if fp8:
        q, scale = _quant_fp8_per_row(w)
        yield f"{key}.weight", q
        yield f"{key}.weight_scale", scale
    else:
        yield f"{key}.weight", w.to(torch.bfloat16)


def _iter_kda_layer(reader, layer: int, attn_fp8: bool) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    if attn_fp8:
        # fp8 resident: q|k|v (the 201 MB/layer read) as one W8A16 GEMM with
        # per-row scales; the small gate projections b|f_a|g_a stay bf16.
        qkv = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in ("q_proj", "k_proj", "v_proj")],
            dim=0,
        )
        q, scale = _quant_fp8_per_row(qkv)
        yield f"{dst}.in_proj_qkv.weight", q
        yield f"{dst}.in_proj_qkv.weight_scale", scale
        bfg = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in ("b_proj", "f_a_proj", "g_a_proj")],
            dim=0,
        )
        yield f"{dst}.in_proj_bfg.weight", bfg
    else:
        # One fused input GEMM: q|k|v|b|f_a|g_a (output-axis concat).
        fused = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in _KDA_IN_PROJ], dim=0
        )
        yield f"{dst}.in_proj.weight", fused
    # One merged depthwise conv over the q|k|v stream (channel-axis concat).
    conv = torch.cat(
        [reader.get(f"{src}.{p}_conv1d.weight").to(torch.bfloat16) for p in ("q", "k", "v")],
        dim=0,
    )
    yield f"{dst}.conv1d.weight", conv
    for p in ("f_b_proj", "g_b_proj"):
        yield f"{dst}.{p}.weight", reader.get(f"{src}.{p}.weight").to(torch.bfloat16)
    yield from _maybe_fp8(f"{dst}.o_proj", reader.get(f"{src}.o_proj.weight"), attn_fp8)
    # Gate params stay fp32 (the recurrent kernels read them as fp32).
    yield f"{dst}.A_log", reader.get(f"{src}.A_log").to(torch.float32)
    yield f"{dst}.dt_bias", reader.get(f"{src}.dt_bias").to(torch.float32)
    yield f"{dst}.o_norm.weight", reader.get(f"{src}.o_norm.weight").to(torch.bfloat16)


def _iter_dsa_layer(reader, layer: int, attn_fp8: bool) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    fp8_projs = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") if attn_fp8 else ()
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        w = reader.get(f"{src}.{proj}.weight")
        yield from _maybe_fp8(f"{dst}.{proj}", w, proj in fp8_projs)
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


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3 stores routed experts as NVFP4 and only supports the offload backend; "
        "experts are loaded into the offload cache via load_nvfp4_expert_sources()."
    )
    assert include_non_moe
    if get_tp_info().size > 1:
        # The loader emits full fused KDA/DSA tensors; TP sharding (per-head q|k|v|b
        # splits, replicated f_a|g_a, row-parallel o_proj) is not implemented yet --
        # same status as every other linear-hybrid / offload-family model in tree.
        raise NotImplementedError("glm5_next weight loading currently supports TP=1 only")
    config = parse_config(cached_load_hf_config(model_path))
    args: Glm5NextArgs = config.glm5_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    attn_fp8 = config.attn_quant == "fp8_pertensor"
    mlp_fp8 = config.dense_quant == "fp8_pertensor"
    head_fp8 = config.lm_head_quant == "fp8_pertensor"
    if primary:
        from freetoken.utils import init_logger

        init_logger(__name__).info(
            f"GLM-5.3 resident quant: attn={config.attn_quant} dense={config.dense_quant} "
            f"lm_head={config.lm_head_quant} (FREETOKEN_GLM5_ATTN_FP8/FREETOKEN_GLM5_MLP_FP8; "
            "an FTW conversion records these choices implicitly -- serve with the same flags)"
        )
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.3 dense weights",
            disable=not primary,
        ):
            src = f"{_CKPT}.layers.{layer}"
            dst = f"{_MODEL}.layers.{layer}"
            if args.is_kda_layer(layer):
                yield from _iter_kda_layer(reader, layer, attn_fp8)
            else:
                yield from _iter_dsa_layer(reader, layer, attn_fp8)

            # mHC mixing tensors, fp32 on every layer.
            for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                yield f"{dst}.{hc}", reader.get(f"{src}.{hc}").to(torch.float32)

            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(
                    torch.bfloat16
                )

            if layer < config.first_k_dense_replace:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _maybe_fp8(
                        f"{dst}.mlp.{proj}", reader.get(f"{src}.mlp.{proj}.weight"), mlp_fp8
                    )
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
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _maybe_fp8(
                        f"{dst}.mlp.shared_experts.{proj}",
                        reader.get(f"{src}.mlp.shared_experts.{proj}.weight"),
                        mlp_fp8,
                    )

        yield f"{_MODEL}.embed_tokens.weight", reader.get(
            f"{_CKPT}.embed_tokens.weight"
        ).to(torch.bfloat16)
        yield f"{_MODEL}.norm.weight", reader.get(f"{_CKPT}.norm.weight").to(torch.bfloat16)
        head = reader.get("lm_head.weight")
        if head_fp8 and not config.tie_word_embeddings:
            q, scale = _quant_fp8_per_row(head)
            yield "lm_head.weight", q
            yield "lm_head.weight_scale", scale
        else:
            yield "lm_head.weight", head.to(torch.bfloat16)
    finally:
        reader.close()


__all__ = ["iter_weights", "load_nvfp4_expert_sources"]
