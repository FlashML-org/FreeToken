"""Weight loading for MiniMax-M3 (``minimax_m3``).

The checkpoint is the multimodal wrapper layout: every text-tower tensor carries a
``language_model.`` prefix (``language_model.model.layers.N...``,
``language_model.lm_head.weight``); the ViT stack (``vision_tower.`` /
``multi_modal_projector.`` / ``patch_merge_mlp.``) is never read (text-only serving).

Resident (non routed-expert) dense projections are MXFP8 in the checkpoint
(fp8-e4m3 ``weight`` + uint8 e8m0 block-32 ``weight_scale_inv``). What this loader
yields follows the quant modes RESOLVED IN ``parse_config`` (``ModelConfig.attn_quant``
/ ``dense_quant``, from the FREETOKEN_M3_*_MXFP8 switches, default on): in the default
mode the fp8 weight and its scale codes stream through verbatim (merged output-wise for
the fused qkv / index-qk / gate-up projections -- the scales are per-output-row, so
fusion is exact); with a switch off the projections are dequantized to bf16 at load.
Norms, the router gate, ``e_score_correction_bias`` (kept fp32), embeddings and
lm_head are unquantized and stream through verbatim. Routed experts are NVFP4
(``w1/w3/w2`` = gate/up/down, same ModelOpt layout as MiniMax-M2) and go to the
offload cache via ``load_nvfp4_expert_sources``; expert ``input_scale`` calibration
tensors are unused (W4A16) and never match the bank spec.

FTW caveat: an FTW checkpoint stores whatever iter_weights yielded at CONVERSION time,
and the model is built from the env at SERVE time -- the two must agree (a mismatch
fails loudly in load_state_dict on the ``*.weight_scale_inv`` keys). The active modes
are logged at load so conversion logs record the choice.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config

_EXPERT_KEY_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\."
    r"experts\.(?P<expert>\d+)\.(?P<proj>w1|w2|w3)\."
    r"(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"w1": "gate", "w3": "up", "w2": "down"},
    # Experts exist only for layers [first_k_dense_replace, num_layers); banks pack by
    # MoE-layer index so the leading dense layers leave no holes (GLM precedent).
    layer_to_bank=lambda layer, config: (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    ),
    desc="MiniMax-M3 NVFP4 experts",
)


class _ShardReader:
    """Random-access reader over the safetensors index (GLM-5.2 precedent): the M3
    dense pass walks layers in model order and fuses q/k/v (etc.) output-wise, which
    a shard-ordered stream cannot do without buffering whole projections anyway."""

    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _read_proj(reader: _ShardReader, key: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One dense projection as ``(weight, scale_codes | None)``. MXFP8 tensors carry a
    ``weight_scale_inv`` sibling; a bf16 tensor (nothing quantized under this name)
    comes back scale-less and passes through every mode unchanged."""
    w = reader.get(f"{key}.weight")
    scale_key = f"{key}.weight_scale_inv"
    if reader.has(scale_key):
        s = reader.get(scale_key)
        assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.uint8, (
            f"unexpected MXFP8 dtypes at {key}: {w.dtype}/{s.dtype}"
        )
        # The GEMV folds w * 2**(code-127) into the 16-bit compute dtype, which
        # is lossless only while the product stays in range; codes above 245
        # could overflow bf16. Real checkpoints sit near 127 -- pin at load.
        assert int(s.max()) <= 245, (
            f"e8m0 scale code {int(s.max())} at {key} exceeds the bf16-exact "
            "fold bound (245); the W8A16 kernels would overflow"
        )
        return w, s
    return w, None


def _emit_fused(
    reader: _ShardReader,
    out_key: str,
    part_keys: list[str],
    keep_mxfp8: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Emit one (possibly multi-part output-wise fused) projection in the resolved
    quant mode. MXFP8 scales are per-output-row / per-32-input-block, so concatenating
    the parts' weights AND scales along the output dim is exact."""
    parts = [_read_proj(reader, key) for key in part_keys]
    if keep_mxfp8 and all(s is not None for _, s in parts):
        yield f"{out_key}.weight", torch.cat([w for w, _ in parts], dim=0)
        yield f"{out_key}.weight_scale_inv", torch.cat([s for _, s in parts], dim=0)
        return
    from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant

    dequant = [
        mxfp8_dequant(w, s, dtype=torch.bfloat16) if s is not None else w
        for w, s in parts
    ]
    yield f"{out_key}.weight", (
        torch.cat(dequant, dim=0) if len(dequant) > 1 else dequant[0]
    )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "MiniMax-M3 stores routed experts as NVFP4 and only supports the offload MoE "
        "backend; experts are loaded into the offload cache via load_nvfp4_expert_sources()."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    args = config.m3_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    attn_mx = config.attn_quant == "mxfp8"
    mlp_mx = config.dense_quant == "mxfp8"
    if primary:
        from freetoken.utils import init_logger

        init_logger(__name__).info(
            f"MiniMax-M3 resident quant: attn={config.attn_quant} dense={config.dense_quant} "
            f"lm_head={config.lm_head_quant} (FREETOKEN_M3_ATTN_MXFP8/FREETOKEN_M3_MLP_MXFP8; "
            "an FTW conversion records these choices implicitly -- serve with the same flags)"
        )
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading MiniMax-M3 dense weights",
            disable=not primary,
        ):
            src = f"language_model.model.layers.{layer}"
            dst = f"model.layers.{layer}"
            a_src, a_dst = f"{src}.self_attn", f"{dst}.self_attn"

            yield from _emit_fused(
                reader,
                f"{a_dst}.qkv_proj",
                [f"{a_src}.q_proj", f"{a_src}.k_proj", f"{a_src}.v_proj"],
                attn_mx,
            )
            yield from _emit_fused(reader, f"{a_dst}.o_proj", [f"{a_src}.o_proj"], attn_mx)
            for norm in ("q_norm", "k_norm"):
                yield f"{a_dst}.{norm}.weight", reader.get(f"{a_src}.{norm}.weight")

            # Block-sparse indexer (sparse layers only; the dense ablation builds no
            # indexer modules, so its tensors are skipped entirely).
            if args.is_sparse_layer(layer):
                yield from _emit_fused(
                    reader,
                    f"{a_dst}.index_qk_proj",
                    [f"{a_src}.index_q_proj", f"{a_src}.index_k_proj"],
                    attn_mx,
                )
                for norm in ("index_q_norm", "index_k_norm"):
                    yield f"{a_dst}.{norm}.weight", reader.get(f"{a_src}.{norm}.weight")

            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight")

            if layer in args.moe_layer_ids:
                m_src, m_dst = f"{src}.block_sparse_moe", f"{dst}.block_sparse_moe"
                # fp32 like the bias: both sit on the top-4 selection boundary.
                yield (
                    f"{m_dst}.gate.weight",
                    reader.get(f"{m_src}.gate.weight").to(torch.float32),
                )
                # fp32 in the checkpoint AND the module (top-k selection boundary).
                yield (
                    f"{m_dst}.e_score_correction_bias",
                    reader.get(f"{m_src}.e_score_correction_bias").to(torch.float32),
                )
                s_src, s_dst = f"{m_src}.shared_experts", f"{m_dst}.shared_experts"
                yield from _emit_fused(
                    reader,
                    f"{s_dst}.gate_up_proj",
                    [f"{s_src}.gate_proj", f"{s_src}.up_proj"],
                    mlp_mx,
                )
                yield from _emit_fused(
                    reader, f"{s_dst}.down_proj", [f"{s_src}.down_proj"], mlp_mx
                )
            else:
                yield from _emit_fused(
                    reader,
                    f"{dst}.mlp.gate_up_proj",
                    [f"{src}.mlp.gate_proj", f"{src}.mlp.up_proj"],
                    mlp_mx,
                )
                yield from _emit_fused(
                    reader, f"{dst}.mlp.down_proj", [f"{src}.mlp.down_proj"], mlp_mx
                )

        yield "model.embed_tokens.weight", reader.get("language_model.model.embed_tokens.weight")
        yield "model.norm.weight", reader.get("language_model.model.norm.weight")
        yield "lm_head.weight", reader.get("language_model.lm_head.weight")
    finally:
        reader.close()



def nvfp4_expert_source_spec(model_path: str, config):
    """The source spec the disk tier must index this checkpoint with.

    Same object the loader passes to ``load_nvfp4_expert_source_banks``, exposed so the
    shared provider can build the disk index without knowing the family: the index has to
    read the rows the loader placed, so one spec has to serve both."""
    return _NVFP4_SOURCE_SPEC

def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None, disk_tier=None
) -> dict[str, list[torch.Tensor]]:
    """CPU NVFP4 expert source banks for the offload cache; see load_nvfp4_expert_source_banks."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
        disk_tier=disk_tier,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None, disk_tier=None
):
    """parallel: same NVFP4 source banks via the common chunked multi-threaded O_DIRECT reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
        disk_tier=disk_tier,
    )


__all__ = ["iter_weights", "load_nvfp4_expert_sources", "load_nvfp4_expert_sources_parallel"]
