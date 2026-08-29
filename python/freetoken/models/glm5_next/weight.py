"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Checkpoint layout (LibertAIDAI NVFP4, verified from the shard headers):
  * language tower under ``model.language_model.*`` -> served as ``model.*``;
  * layer 45 is the MTP layer (eh_proj/enorm/hnorm/shared_head + its own experts) --
    discarded, like every other FreeToken model;
  * KDA linear layers ship SEPARATE q/k/v projections and SEPARATE per-projection
    depthwise convs; the module serves one merged GEMM + one grouped conv, and
    depthwise-ness makes the [q|k|v] channel concat EXACTLY equivalent;
  * routed experts are modelopt NVFP4 (packed uint8 + fp8 block-16 scales + fp32
    global) with separate gate/up -> the generic bank builder stacks them;
  * everything else bf16 except hc_*_base/scale, A_log, dt_bias (fp32).

Quality-first baseline: every resident weight streams through VERBATIM (bf16); the
FP8 requant switches exist (``FREETOKEN_GLM5_ATTN_FP8`` / ``_MLP_FP8`` via
parse_config) for the later A/B, same machinery as glm_moe_dsa.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.glm_moe_dsa.weight import _quant_fp8_per_row, _ShardReader
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config

# ---------------------------------------------------------------------------------
# Routed experts -> offload cache (generic NVFP4 bank builder, GLM-5.3 name layout).
# MTP layer 45 is excluded by ``layer >= config.num_layers`` (num_layers == 45).
# ---------------------------------------------------------------------------------
_ROUTED_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
def _layer_to_bank(layer: int, config) -> int | None:
    """Host-bank id for a checkpoint layer: dense over the non-resident MoE layers
    (resident layers live on the GPU as model weights; MTP layer 45 excluded by
    ``layer >= num_layers``)."""
    from .moe import offload_moe_layers

    if layer < config.first_k_dense_replace or layer >= config.num_layers:
        return None
    if layer in config.glm5_args.resident_layer_ids:
        return None
    return offload_moe_layers(config).index(layer)


_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_ROUTED_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts",
)


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers, chunk=chunk, layer_sink=layer_sink,
    )


# ---------------------------------------------------------------------------------
# Resident (dense) weights.
# ---------------------------------------------------------------------------------
def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3 routed experts are NVFP4 offload-only; they load via "
        "load_nvfp4_expert_sources()."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    a = config.glm5_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    attn_fp8 = config.attn_quant == "fp8_pertensor"
    mlp_fp8 = config.dense_quant == "fp8_pertensor"
    head_fp8 = config.lm_head_quant == "fp8_pertensor"

    def _proj(src_key: str, dst_key: str, fp8: bool):
        w = reader.get(f"{src_key}.weight")
        if fp8:
            q, scale = _quant_fp8_per_row(w)
            yield f"{dst_key}.weight", q
            yield f"{dst_key}.weight_scale", scale
        else:
            yield f"{dst_key}.weight", w

    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.3 dense weights",
            disable=not primary,
        ):
            src = f"model.language_model.layers.{layer}"
            dst = f"model.layers.{layer}"
            # Hyper-connections: checkpoint uses the DSV4 flat names our layer also uses.
            for t in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
                      "hc_attn_scale", "hc_ffn_scale"):
                yield f"{dst}.{t}", reader.get(f"{src}.{t}")
            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight")

            sa_s, sa_d = f"{src}.self_attn", f"{dst}.self_attn"
            if a.is_kda_layer(layer):
                # Separate q/k/v proj + per-proj depthwise convs -> merged [q|k|v]
                # GEMM / grouped conv (channel-concat is exact for depthwise).
                qkv = torch.cat(
                    [reader.get(f"{sa_s}.{p}_proj.weight") for p in ("q", "k", "v")], dim=0
                )
                if a.kda_fp8:
                    qq, qscale = _quant_fp8_per_row(qkv)
                    yield f"{sa_d}.in_proj_qkv.weight", qq
                    yield f"{sa_d}.in_proj_qkv.weight_scale", qscale
                else:
                    yield f"{sa_d}.in_proj_qkv.weight", qkv
                conv = torch.cat(
                    [reader.get(f"{sa_s}.{p}_conv1d.weight") for p in ("q", "k", "v")], dim=0
                )
                yield f"{sa_d}.conv1d.weight", conv
                for p in ("f_a_proj", "f_b_proj", "b_proj", "g_a_proj", "g_b_proj"):
                    yield f"{sa_d}.{p}.weight", reader.get(f"{sa_s}.{p}.weight")
                yield from _proj(f"{sa_s}.o_proj", f"{sa_d}.o_proj", a.kda_fp8)
                for t in ("dt_bias", "A_log"):
                    yield f"{sa_d}.{t}", reader.get(f"{sa_s}.{t}")
                yield f"{sa_d}.o_norm.weight", reader.get(f"{sa_s}.o_norm.weight")
            else:
                fp8_projs = (
                    ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")
                    if attn_fp8 else ()
                )
                for p in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                          "o_proj"):
                    yield from _proj(f"{sa_s}.{p}", f"{sa_d}.{p}", p in fp8_projs)
                for n in ("q_a_layernorm", "kv_a_layernorm"):
                    yield f"{sa_d}.{n}.weight", reader.get(f"{sa_s}.{n}.weight")
                # DSA lightning indexer + k-pool compressor (always bf16-faithful;
                # ape/gate load into the module's fp32/bf16 slots via _materialize).
                for p in ("wq_b", "wk", "weights_proj"):
                    yield f"{sa_d}.indexer.{p}.weight", reader.get(f"{sa_s}.indexer.{p}.weight")
                yield f"{sa_d}.indexer.k_norm.weight", reader.get(f"{sa_s}.indexer.k_norm.weight")
                yield f"{sa_d}.indexer.k_norm.bias", reader.get(f"{sa_s}.indexer.k_norm.bias")
                yield (f"{sa_d}.indexer.kpool_gate.weight",
                       reader.get(f"{sa_s}.indexer.index_kpool_compress_gate"))
                yield (f"{sa_d}.indexer.kpool_ape",
                       reader.get(f"{sa_s}.indexer.index_kpool_compress_ape"))

            m_s, m_d = f"{src}.mlp", f"{dst}.mlp"
            if layer in a.resident_layer_ids:
                # Resident layer: the full packed banks load as MODEL weights (GPU).
                E = config.num_experts

                def _stack(proj, kind):
                    return torch.stack(
                        [reader.get(f"{m_s}.experts.{e}.{proj}.{kind}") for e in range(E)]
                    )

                def _glob(proj, rows):
                    return torch.stack([
                        reader.get(f"{m_s}.experts.{e}.{proj}.weight_scale_2")
                        .reshape(1).to(torch.float16).expand(rows)
                        for e in range(E)
                    ]).contiguous()

                gu_p = torch.cat([_stack("gate_proj", "weight"), _stack("up_proj", "weight")], dim=1)
                gu_s = torch.cat([_stack("gate_proj", "weight_scale"), _stack("up_proj", "weight_scale")], dim=1)
                i_sz = gu_p.shape[1] // 2
                gu_g = torch.cat([_glob("gate_proj", i_sz), _glob("up_proj", i_sz)], dim=1)
                yield f"{m_d}.experts.gate_up_packed", gu_p
                yield f"{m_d}.experts.gate_up_scale", gu_s
                yield f"{m_d}.experts.gate_up_global", gu_g
                yield f"{m_d}.experts.down_packed", _stack("down_proj", "weight")
                yield f"{m_d}.experts.down_scale", _stack("down_proj", "weight_scale")
                yield f"{m_d}.experts.down_global", _glob("down_proj", config.hidden_size)
            if a.is_dense_layer(layer):
                for p in ("gate_proj", "up_proj", "down_proj"):
                    yield from _proj(f"{m_s}.{p}", f"{m_d}.{p}", mlp_fp8)
            else:
                yield f"{m_d}.gate.weight", reader.get(f"{m_s}.gate.weight")
                yield (f"{m_d}.e_score_correction_bias",
                       reader.get(f"{m_s}.gate.e_score_correction_bias"))
                for p in ("gate_proj", "up_proj", "down_proj"):
                    yield from _proj(f"{m_s}.shared_experts.{p}", f"{m_d}.shared_experts.{p}", mlp_fp8)

        if a.mtp_enabled:
            L = a.mtp_layer_id
            src = f"model.language_model.layers.{L}"
            yield "mtp.enorm.weight", reader.get(f"{src}.enorm.weight")
            yield "mtp.hnorm.weight", reader.get(f"{src}.hnorm.weight")
            yield "mtp.eh_proj.weight", reader.get(f"{src}.eh_proj.weight")
            yield "mtp.shared_head.norm.weight", reader.get(f"{src}.shared_head.norm.weight")
            dstl = "mtp.layer"
            # layer 45 ships NO hc_* weights: plain pre-norm residual block.
            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dstl}.{norm}.weight", reader.get(f"{src}.{norm}.weight")
            sa_s, sa_d = f"{src}.self_attn", f"{dstl}.self_attn"
            fp8_projs = (
                ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") if attn_fp8 else ()
            )
            for pj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                yield from _proj(f"{sa_s}.{pj}", f"{sa_d}.{pj}", pj in fp8_projs)
            for n in ("q_a_layernorm", "kv_a_layernorm"):
                yield f"{sa_d}.{n}.weight", reader.get(f"{sa_s}.{n}.weight")
            for pj in ("wq_b", "wk", "weights_proj"):
                yield f"{sa_d}.indexer.{pj}.weight", reader.get(f"{sa_s}.indexer.{pj}.weight")
            yield f"{sa_d}.indexer.k_norm.weight", reader.get(f"{sa_s}.indexer.k_norm.weight")
            yield f"{sa_d}.indexer.k_norm.bias", reader.get(f"{sa_s}.indexer.k_norm.bias")
            yield (f"{sa_d}.indexer.kpool_gate.weight",
                   reader.get(f"{sa_s}.indexer.index_kpool_compress_gate"))
            yield (f"{sa_d}.indexer.kpool_ape",
                   reader.get(f"{sa_s}.indexer.index_kpool_compress_ape"))
            m_s, m_d = f"{src}.mlp", f"{dstl}.mlp"
            yield f"{m_d}.gate.weight", reader.get(f"{m_s}.gate.weight")
            yield (f"{m_d}.e_score_correction_bias",
                   reader.get(f"{m_s}.gate.e_score_correction_bias"))
            for pj in ("gate_proj", "up_proj", "down_proj"):
                yield from _proj(f"{m_s}.shared_experts.{pj}", f"{m_d}.shared_experts.{pj}", mlp_fp8)
            E = config.num_experts

            def _stack45(proj, kind):
                return torch.stack(
                    [reader.get(f"{m_s}.experts.{e}.{proj}.{kind}") for e in range(E)]
                )

            def _glob45(proj, rows):
                return torch.stack([
                    reader.get(f"{m_s}.experts.{e}.{proj}.weight_scale_2")
                    .reshape(1).to(torch.float16).expand(rows)
                    for e in range(E)
                ]).contiguous()

            gu_p = torch.cat([_stack45("gate_proj", "weight"), _stack45("up_proj", "weight")], dim=1)
            gu_s = torch.cat([_stack45("gate_proj", "weight_scale"), _stack45("up_proj", "weight_scale")], dim=1)
            i_sz = gu_p.shape[1] // 2
            gu_g = torch.cat([_glob45("gate_proj", i_sz), _glob45("up_proj", i_sz)], dim=1)
            yield f"{m_d}.experts.gate_up_packed", gu_p
            yield f"{m_d}.experts.gate_up_scale", gu_s
            yield f"{m_d}.experts.gate_up_global", gu_g
            yield f"{m_d}.experts.down_packed", _stack45("down_proj", "weight")
            yield f"{m_d}.experts.down_scale", _stack45("down_proj", "weight_scale")
            yield f"{m_d}.experts.down_global", _glob45("down_proj", config.hidden_size)

        yield "model.embed_tokens.weight", reader.get("model.language_model.embed_tokens.weight")
        yield "model.norm.weight", reader.get("model.language_model.norm.weight")
        head = reader.get("lm_head.weight")
        if head_fp8 and not config.tie_word_embeddings:
            q, scale = _quant_fp8_per_row(head)
            yield "lm_head.weight", q
            yield "lm_head.weight_scale", scale
        else:
            yield "lm_head.weight", head
    finally:
        reader.close()


__all__ = [
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
