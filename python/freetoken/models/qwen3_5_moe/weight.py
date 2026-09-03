from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
from freetoken.models.loader import (
    CT_SCALE_SUFFIXES,
    ShardReader,
    ct_bf16_fuse,
    ct_nvfp4_fuse,
    drop_page_cache,
    iter_weight_files,
    nvfp4_parts_ct,
)
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import _compressed_tensors_nvfp4, parse_config

# Expert weights are stored pre-fused per layer: experts.gate_up_proj / experts.down_proj.
_PACKED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)

# NVFP4 routed experts (nvidia modelopt checkpoint): per-expert, un-fused, under the raw
# ``model.language_model.layers.N.mlp.experts.E.{proj}`` key. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP
# head's ``mtp.layers.N.mlp.experts.*`` tensors (served text-only, dropped).
_NVFP4_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.5 NVFP4 experts",
)
# Suffixes of the per-tensor modelopt quant scales; consumed alongside their ``.weight``,
# never yielded on their own.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# Gemma-style (1+weight) RMSNorm weights. Excludes GDN gated norm (linear_attn.norm),
# which is a standard weight*x norm.
_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)
# shared-expert gate/up merge -> shared_expert.gate_up_proj
_SHARED_GATE = ".mlp.shared_expert.gate_proj.weight"
_SHARED_UP = ".mlp.shared_expert.up_proj.weight"

# Fused projections: concat checkpoint matrices in this exact order to match the model's
# LinearColParallelMerged split. fused_suffix -> ordered parts.
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ),
    # Dense (non-MoE) layer MLP: merge gate|up -> gate_up_proj. Only fires on a bare
    # ``.mlp.gate_proj``; ``.mlp.shared_expert.gate_proj`` (MoE) does not end with this.
    ".mlp.gate_up_proj.weight": (
        ".mlp.gate_proj.weight", ".mlp.up_proj.weight",
    ),
}


def _dequant_fp8_weight(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Weight-only FP8 -> bf16 (per-tensor static scale). Activations stay bf16 (W8A16),
    which is at least as precise as the checkpoint's intended W8A8."""
    return weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16)


def _dequant_nvfp4_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    """Dense NVFP4 -> bf16 (W4A16): ``fp4 * block_scale * global_scale``. ``weight`` is
    [O, IN//2] uint8, ``weight_scale`` [O, IN//16] fp8-e4m3, ``weight_scale_2`` the per-tensor
    global scalar (broadcast to per-row, matching the offload-cache dequant kernel)."""
    # The dequant kernel is GPU-only; the checkpoint-conversion path loads dense weights on
    # CPU, so run on CUDA and return on the caller's device (no-op when already on GPU).
    orig_device = weight.device
    if orig_device.type != "cuda":
        dev = torch.device("cuda")
        weight, weight_scale, weight_scale_2 = (
            weight.to(dev), weight_scale.to(dev), weight_scale_2.to(dev)
        )
    out_features = weight.shape[0]
    global_scale = weight_scale_2.reshape(1).to(torch.float16).expand(out_features).contiguous()
    slots = torch.zeros(1, dtype=torch.int32, device=weight.device)
    out = dequant_nvfp4(
        weight.unsqueeze(0).contiguous(),
        weight_scale.unsqueeze(0).contiguous(),
        global_scale.unsqueeze(0),
        slots,
        dtype=torch.bfloat16,
    )[0]
    return out.to(orig_device)


def _load_maybe_quantized(f, raw_name: str, keyset: set[str]) -> torch.Tensor:
    """Load ``raw_name``; if it is a quantized ``.weight`` with sibling modelopt scales in
    the same shard, dequantize to bf16 (NVFP4 if ``weight_scale_2`` present, else FP8).
    Plain bf16 weights pass through unchanged."""
    tensor = f.get_tensor(raw_name)
    if not raw_name.endswith(".weight"):
        return tensor
    base = raw_name[: -len(".weight")]
    if base + ".weight_scale_2" in keyset:  # NVFP4 (two-level block scale)
        return _dequant_nvfp4_weight(
            tensor, f.get_tensor(base + ".weight_scale"), f.get_tensor(base + ".weight_scale_2")
        )
    if base + ".weight_scale" in keyset:  # FP8 (per-tensor scale)
        return _dequant_fp8_weight(tensor, f.get_tensor(base + ".weight_scale"))
    return tensor


def _rename(raw_name: str) -> str | None:
    """HF key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
        return None
    # ModelOpt FP8 KV-cache static scales (full-attention layers only). FreeToken keeps the
    # KV cache in the engine's native precision (>= the checkpoint's quantized KV), so these
    # per-tensor q/k/v scales are unused -- drop them rather than fail as unexpected keys.
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    name = raw_name
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model.") :]
    return name


def _is_gemma_norm(name: str) -> bool:
    return name == "model.norm.weight" or name.endswith(_GEMMA_NORM_SUFFIXES)


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """buffer a fusion part; return merged ``(name, tensor)`` once all parts arrive,
    ``()`` while incomplete, ``None`` if not a fusion part."""
    for fused_suffix, parts in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if name.endswith(part):
                key = name[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return key, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    if _compressed_tensors_nvfp4(hf_config):
        # Dense compressed-tensors NVFP4 (e.g. Qwen3.6-27B): attn (q/k/v/o, GDN out_proj) +
        # dense MLP are W4A16 NVFP4; GDN in_proj_*, lm_head, norms bf16.
        yield from _iter_weights_compressed_tensors(
            model_path, device,
            include_non_moe=include_non_moe, include_moe_experts=include_moe_experts,
            nvfp4=config.dense_quant == "nvfp4",
        )
        return
    if config.expert_quant == "fp8_block":
        # Dense (attn/GDN/shared-expert) weights are always block-fp8; routed experts are
        # yielded here only for the resident path (include_moe_experts=True). Under offload
        # they are excluded and loaded by setup_offload_expert_banks instead.
        yield from _iter_weights_fp8(
            model_path, device,
            include_non_moe=include_non_moe, include_moe_experts=include_moe_experts,
        )
        return
    if config.attn_quant == "fp8_pertensor":
        # modelopt MIXED_PRECISION: dense attn/GDN projections kept per-tensor FP8 (fp8
        # weight + per-row scale, W8A16 kernel); NVFP4 dense (shared_expert/lm_head) kept
        # native FP4 (W4A16) when dense_quant=="nvfp4", else dequantized to bf16; routed
        # NVFP4 experts excluded (offload cache).
        yield from _iter_weights_attn_fp8(
            model_path, device,
            include_non_moe=include_non_moe, include_moe_experts=include_moe_experts,
            dense_nvfp4=config.dense_quant == "nvfp4",
            lmhead_nvfp4=config.lm_head_quant == "nvfp4",
        )
        return
    tp_info = get_tp_info()
    if tp_info.size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading currently supports TP=1 only")

    # Pure-NVFP4 checkpoint (bf16 attn): the dense MLP projections (shared_expert) are still
    # stored as packed FP4 -- keep them native (W4A16) when dense_quant=="nvfp4" rather than
    # dequantizing to bf16. lm_head here is bf16 (pure NVFP4 doesn't quantize it).
    dense_nvfp4 = config.dense_quant == "nvfp4"
    lmhead_nvfp4 = config.lm_head_quant == "nvfp4"
    shared_buf: dict[str, dict[str, torch.Tensor]] = {}
    nvfp4_shared_buf: dict[str, dict[str, tuple]] = {}
    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not tp_info.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                # Per-expert NVFP4 tensors go to the offload cache (load_nvfp4_expert_sources),
                # not the dense pass. bf16-base stacked experts (experts.gate_up_proj) have no
                # ``.mlp.experts.<int>.`` so they are unaffected and still hit _PACKED_EXPERT.
                if _NVFP4_EXPERT_RE.search(raw_name):
                    continue
                # Standalone modelopt scales are consumed with their .weight, never yielded.
                if raw_name.endswith(_SCALE_SUFFIXES):
                    continue

                name = _rename(raw_name)
                if name is None:
                    continue

                is_expert = _PACKED_EXPERT_PATTERN.match(name) is not None
                if is_expert and not include_moe_experts:
                    continue
                if not is_expert and not include_non_moe:
                    continue

                # NVFP4 dense projections kept native (W4A16) where the model expects them
                # (shared_expert); everything else dequantizes to bf16 below as before.
                if (dense_nvfp4 or lmhead_nvfp4) and name.endswith(".weight") \
                        and raw_name[: -len(".weight")] + ".weight_scale_2" in keyset:
                    emit = _dense_nvfp4_emit(
                        f, name[: -len(".weight")], raw_name[: -len(".weight")],
                        shared_nvfp4=dense_nvfp4, lmhead_nvfp4=lmhead_nvfp4,
                        shared_buf=nvfp4_shared_buf,
                    )
                    if emit is not _NOT_DENSE_NVFP4:
                        yield from emit
                        continue

                tensor = _load_maybe_quantized(f, raw_name, keyset)

                # merge shared-expert gate/up -> gate_up_proj
                if name.endswith(_SHARED_GATE) or name.endswith(_SHARED_UP):
                    prefix = name.rsplit(".mlp.shared_expert.", 1)[0]
                    slots = shared_buf.setdefault(prefix, {})
                    slots["gate" if name.endswith(_SHARED_GATE) else "up"] = tensor
                    if "gate" in slots and "up" in slots:
                        merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                        del shared_buf[prefix]
                        yield f"{prefix}.mlp.shared_expert.gate_up_proj.weight", merged
                    continue

                # fuse q/k/v -> qkv_proj and GDN in_proj_{qkv,z,b,a} -> in_proj
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue

                if _is_gemma_norm(name):
                    tensor = tensor + 1.0  # (1 + weight) baked into the stored weight

                yield name, tensor

    assert not shared_buf, f"Incomplete shared-expert merges: {list(shared_buf.keys())}"
    assert not nvfp4_shared_buf, f"Incomplete NVFP4 shared-expert merges: {list(nvfp4_shared_buf.keys())}"
    assert not fuse_buf, f"Incomplete projection fusions: {list(fuse_buf.keys())}"


# ======================================================================================
# Mixed-precision modelopt checkpoint (per-tensor FP8 attn/GDN + NVFP4 experts/shared/lm_head)
# ======================================================================================
# FP8 projections fused along the output dim, each part carrying its own scalar weight_scale
# -> a per-output-row scale vector. Keys are the model buffer base (sans .weight/.weight_scale).
_PT_FP8_FUSE: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (
        ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
    ),
    ".linear_attn.in_proj_qkvz": (
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
    ),
}
# bf16 (unquantized) GDN b|a projections fused -> in_proj_ba (matches the fp8 split).
_PT_BF16_FUSE: dict[str, tuple[str, ...]] = {
    ".linear_attn.in_proj_ba": (".linear_attn.in_proj_b", ".linear_attn.in_proj_a"),
}


def _per_row_scale(scalar: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar -> per-output-row fp32 vector ``[rows]`` (exact broadcast)."""
    return scalar.reshape(1).to(torch.float32).expand(rows)


def _pt_fp8_fuse(base: str, weight: torch.Tensor, scalar: torch.Tensor,
                 act_scale: torch.Tensor | None, buf: dict):
    """Buffer an fp8 fusion part ``(weight, scalar, act_scale)``; once all parts arrive emit
    the concatenated ``(.weight fp8, .weight_scale per-row fp32)`` plus the shared
    ``.input_scale``. ``[]`` while incomplete, ``None`` if ``base`` is not an fp8 fusion part.

    The fused parts all read the *same* activation, so modelopt calibrates one activation
    range for all of them and their ``input_scale`` values come out bit-identical (verified on
    Qwen3.8-27B-NVFP4: q/k/v all 0.2053571492, GDN qkv/z both 0.1121651828). Taking the max is
    therefore exact here, and stays correct if a future checkpoint lets them drift."""
    for fused_suffix, parts in _PT_FP8_FUSE.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                key = base[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = (weight, scalar, act_scale)
                if len(slots) < len(parts):
                    return []
                del buf[key]
                ws = [slots[i][0] for i in range(len(parts))]
                ss = [_per_row_scale(slots[i][1], slots[i][0].shape[0]) for i in range(len(parts))]
                emit = [
                    (key + ".weight", torch.cat(ws, dim=0)),
                    (key + ".weight_scale", torch.cat(ss, dim=0).contiguous()),
                ]
                acts = [slots[i][2] for i in range(len(parts))]
                if all(a is not None for a in acts):
                    emit.append((key + ".input_scale", torch.stack(
                        [a.reshape(()).to(torch.float32) for a in acts]).max()))
                return emit
    return None


# Native-NVFP4 dense projections (W4A16): shared-expert gate/up merged on the output dim,
# down + lm_head standalone. Each carries weight (uint8 [O,IN//2]), block scale (fp8
# [O,IN//16]), and a per-output-row global scale (weight_scale_2 broadcast, fp16 [O]). The
# fused gate|up concatenates all three (each part keeps its own global), so it is exact.
_SHARED_GATE_BASE = ".mlp.shared_expert.gate_proj"
_SHARED_UP_BASE = ".mlp.shared_expert.up_proj"
# The MoE shared-expert MLP and the dense (non-MoE) decoder MLP have identical native-FP4
# structure (gate|up merged -> gate_up_proj + standalone down_proj); they differ only in the
# ``.mlp.shared_expert.`` vs bare ``.mlp.`` infix. ``endswith(".mlp.gate_proj")`` is False for
# ``.mlp.shared_expert.gate_proj``, and routed experts are excluded upstream (_NVFP4_EXPERT_RE).
_NVFP4_MLP_LAYOUTS = (
    (".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj",
     ".mlp.shared_expert.down_proj", ".mlp.shared_expert."),
    (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".mlp."),
)


def _nvfp4_parts(f, raw_base: str):
    """Load a native NVFP4 weight as ``(packed uint8 [O, IN//2], block scale fp8 [O, IN//16],
    per-output-row global fp16 [O])`` -- the dense W4A16 kernels' expected buffers."""
    w = f.get_tensor(raw_base + ".weight")            # uint8 packed FP4 (2 codes/byte)
    s = f.get_tensor(raw_base + ".weight_scale")      # fp8-e4m3 per-16 block scale
    g2 = f.get_tensor(raw_base + ".weight_scale_2")   # per-tensor global scalar
    g = g2.reshape(1).to(torch.float16).expand(w.shape[0]).contiguous()
    return w, s, g


# Sentinel: ``base`` is not a dense projection the model keeps native NVFP4 (caller dequantizes).
_NOT_DENSE_NVFP4 = object()


def _dense_nvfp4_emit(
    f, base: str, raw_base: str, *, shared_nvfp4: bool, lmhead_nvfp4: bool, shared_buf: dict
):
    """For a dense ``.weight`` whose checkpoint has a ``weight_scale_2`` (NVFP4), return the list
    of ``(key, tensor)`` to yield as native FP4 -- ``(.weight uint8, .weight_scale fp8 block,
    .weight_global fp16 per-row)`` -- when the model keeps that layer native:

    * the MoE ``shared_expert.{gate,up,down}_proj`` OR the dense (non-MoE) ``.mlp.{gate,up,down}
      _proj`` when ``shared_nvfp4`` (gate/up merged -> ``gate_up_proj``, each part keeping its own
      global scale so the fused weight is exact);
    * ``lm_head`` when ``lmhead_nvfp4``.

    Returns ``[]`` while a gate/up merge is still buffered, or ``_NOT_DENSE_NVFP4`` if the model
    does not keep this layer native (the caller dequantizes to bf16 exactly as before). Shared by
    the mixed-FP8 dense pass and the default (pure-NVFP4) dense pass."""
    is_lmhead = base == "lm_head" or base.endswith(".lm_head")
    if lmhead_nvfp4 and is_lmhead:
        w, s, g = _nvfp4_parts(f, raw_base)
        return [(base + ".weight", w), (base + ".weight_scale", s), (base + ".weight_global", g)]
    if not shared_nvfp4:
        return _NOT_DENSE_NVFP4
    for gate_b, up_b, down_b, infix in _NVFP4_MLP_LAYOUTS:
        if base.endswith(down_b):
            w, s, g = _nvfp4_parts(f, raw_base)
            return [(base + ".weight", w), (base + ".weight_scale", s), (base + ".weight_global", g)]
        if base.endswith(gate_b) or base.endswith(up_b):
            w, s, g = _nvfp4_parts(f, raw_base)
            prefix = base.rsplit(infix, 1)[0] + infix
            slots = shared_buf.setdefault(prefix, {})
            slots["gate" if base.endswith(gate_b) else "up"] = (w, s, g)
            if "gate" not in slots or "up" not in slots:
                return []
            gw, gs, gg = slots["gate"]
            uw, us, ug = slots["up"]
            del shared_buf[prefix]
            pre = f"{prefix}gate_up_proj"
            return [
                (pre + ".weight", torch.cat([gw, uw], dim=0)),
                (pre + ".weight_scale", torch.cat([gs, us], dim=0)),
                (pre + ".weight_global", torch.cat([gg, ug], dim=0)),
            ]
    return _NOT_DENSE_NVFP4


def _iter_weights_attn_fp8(
    model_path: str, device: torch.device, *, include_non_moe: bool, include_moe_experts: bool,
    dense_nvfp4: bool = False, lmhead_nvfp4: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense pass for the modelopt MIXED_PRECISION Qwen3.5 checkpoint.

    Per-tensor FP8 attn/GDN projections (``self_attn.{q,k,v,o}_proj``, ``linear_attn.
    {in_proj_qkv,in_proj_z,out_proj}``) are kept fp8-e4m3 and yielded as ``.weight`` (fp8) +
    ``.weight_scale`` (per-output-row fp32) instead of dequantized to bf16 -- this halves the
    decode weight traffic of the dense backbone. q/k/v -> ``qkv_proj``, GDN qkv|z ->
    ``in_proj_qkvz`` (fp8) and b|a -> ``in_proj_ba`` (bf16). NVFP4 dense weights
    (shared_expert, lm_head): kept native FP4 -- ``.weight`` (uint8) + ``.weight_scale``
    (fp8 block) + ``.weight_global`` (fp16 per-row) for the W4A16 kernels -- when
    ``dense_nvfp4`` else dequantized to bf16. Routed NVFP4 experts are excluded (served by
    the offload cache). Gemma (1+w) norms get +1."""
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading currently supports TP=1 only")
    if not include_non_moe:
        return  # experts-only call: NVFP4 experts are loaded by the offload bank provider

    tp_info = get_tp_info()
    fp8_buf: dict[str, dict[int, tuple]] = {}
    bf16_buf: dict[str, dict[int, torch.Tensor]] = {}
    shared_buf: dict[str, dict[str, torch.Tensor]] = {}
    nvfp4_shared_buf: dict[str, dict[str, tuple]] = {}

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading mixed-fp8 weights",
        disable=not tp_info.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                if _NVFP4_EXPERT_RE.search(raw_name):
                    continue  # routed experts -> offload cache
                if raw_name.endswith(_SCALE_SUFFIXES):
                    continue  # scales consumed with their .weight

                name = _rename(raw_name)
                if name is None:
                    continue
                if _PACKED_EXPERT_PATTERN.match(name) is not None:
                    continue  # no packed experts in this checkpoint; guard anyway

                if name.endswith(".weight"):
                    base = name[: -len(".weight")]
                    raw_base = raw_name[: -len(".weight")]
                    has_s2 = raw_base + ".weight_scale_2" in keyset
                    has_s = raw_base + ".weight_scale" in keyset
                    if has_s and not has_s2:  # per-tensor FP8 dense projection
                        w = f.get_tensor(raw_name)  # fp8-e4m3, kept verbatim
                        sc = f.get_tensor(raw_base + ".weight_scale")
                        # modelopt's calibrated activation scale: kept (not dropped with the
                        # other scale suffixes) so batched decode can run W8A8 instead of
                        # W8A16. Absent -> the layer stays on the W8A16 kernel.
                        act = (f.get_tensor(raw_base + ".input_scale")
                               if raw_base + ".input_scale" in keyset else None)
                        emit = _pt_fp8_fuse(base, w, sc, act, fp8_buf)
                        if emit is not None:
                            yield from emit
                            continue
                        # standalone fp8 (self_attn.o_proj, linear_attn.out_proj)
                        yield base + ".weight", w
                        yield base + ".weight_scale", _per_row_scale(sc, w.shape[0]).contiguous()
                        if act is not None:
                            yield base + ".input_scale", act.reshape(()).to(torch.float32)
                        continue
                    if has_s2:  # NVFP4 dense: keep native (W4A16) where the model expects it
                        emit = _dense_nvfp4_emit(
                            f, base, raw_base, shared_nvfp4=dense_nvfp4,
                            lmhead_nvfp4=lmhead_nvfp4, shared_buf=nvfp4_shared_buf,
                        )
                        if emit is not _NOT_DENSE_NVFP4:
                            yield from emit
                            continue
                    # NVFP4 -> bf16 (shared_expert, lm_head; dense_nvfp4 off); plain bf16 passes through.
                    tensor = _load_maybe_quantized(f, raw_name, keyset)
                    emit = _ct_bf16_fuse(base, tensor, bf16_buf, _PT_BF16_FUSE)
                    if emit is not None:
                        yield from emit
                        continue
                else:
                    tensor = f.get_tensor(raw_name)

                # shared-expert gate/up -> gate_up_proj (bf16, dequantized above)
                if name.endswith(_SHARED_GATE) or name.endswith(_SHARED_UP):
                    prefix = name.rsplit(".mlp.shared_expert.", 1)[0]
                    slots = shared_buf.setdefault(prefix, {})
                    slots["gate" if name.endswith(_SHARED_GATE) else "up"] = tensor
                    if "gate" in slots and "up" in slots:
                        merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                        del shared_buf[prefix]
                        yield f"{prefix}.mlp.shared_expert.gate_up_proj.weight", merged
                    continue

                if _is_gemma_norm(name):
                    tensor = tensor + 1.0  # (1 + weight) baked into the stored norm weight

                yield name, tensor

    assert not fp8_buf, f"Incomplete fp8 fusions: {list(fp8_buf.keys())}"
    assert not bf16_buf, f"Incomplete bf16 fusions: {list(bf16_buf.keys())}"
    assert not shared_buf, f"Incomplete shared-expert merges: {list(shared_buf.keys())}"
    assert not nvfp4_shared_buf, f"Incomplete NVFP4 shared-expert merges: {list(nvfp4_shared_buf.keys())}"


# ======================================================================================
# compressed-tensors NVFP4 checkpoint (dense Qwen3.x, e.g. Qwen3.6-27B)
# ======================================================================================
# NVFP4 (W4A16) targets every Linear except the per-module ``ignore`` list (lm_head, GDN
# in_proj_*, vision, mtp). Storage differs from modelopt: ``weight_packed`` (uint8 [O, IN//2])
# + ``weight_scale`` (fp8-e4m3 block [O, IN//16]) + a scalar ``weight_global_scale``. The
# stored global is the *quant-side* scale, so the dequant/native global is its reciprocal
# (``1/weight_global_scale``) -- vLLM inverts it identically. Dense MLP gate/up and attention
# q/k/v fuse on the output dim into ``gate_up_proj`` / ``qkv_proj`` (each part keeps its own
# global, so the fused FP4 weight is exact). GDN ``in_proj_{qkv,z,b,a}`` stay bf16 -> ``in_proj``.
_CT_NVFP4_FUSE: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}
_CT_BF16_FUSE: dict[str, tuple[str, ...]] = {
    ".linear_attn.in_proj": (
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
        ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
    ),
}
# The scale suffixes and parts/fuse machinery are shared with muse_glimmer and live
# in models/loader.py.
_CT_SCALE_SUFFIXES = CT_SCALE_SUFFIXES
_nvfp4_parts_ct = nvfp4_parts_ct
_ct_bf16_fuse = ct_bf16_fuse


def _ct_nvfp4_fuse(base: str, parts_tuple: tuple, buf: dict):
    return ct_nvfp4_fuse(base, parts_tuple, buf, _CT_NVFP4_FUSE)


def _iter_weights_compressed_tensors(
    model_path: str, device: torch.device, *, include_non_moe: bool, include_moe_experts: bool,
    nvfp4: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense pass for a compressed-tensors NVFP4 checkpoint (e.g. Qwen3.6-27B).

    Keeps the NVFP4 attention (q/k/v/o, GDN out_proj) and dense MLP (gate/up/down) native
    (W4A16) -- ``.weight`` (uint8) + ``.weight_scale`` (fp8 block) + ``.weight_global`` (fp16
    per-row) -- when ``nvfp4``; otherwise dequantizes each to bf16. q/k/v -> ``qkv_proj``, dense gate/up -> ``gate_up_proj`` (output-dim concat).
    GDN ``in_proj_{qkv,z,b,a}`` stay bf16 -> fused ``in_proj``; ``conv1d``/``A_log``/``dt_bias``/
    gated ``norm`` pass through (fp32 for A_log/dt_bias). Gemma (1+w) norms get +1. lm_head and
    embeddings are bf16. The model is dense (no routed experts), so there is no experts pass."""
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading currently supports TP=1 only")
    if not include_non_moe:
        return  # dense checkpoint: no routed experts to load

    tp_info = get_tp_info()
    nvfp4_buf: dict[str, dict[int, tuple]] = {}
    bf16_buf: dict[str, dict[int, torch.Tensor]] = {}

    def _emit_bf16_weight(name: str, tensor: torch.Tensor):
        """Plain bf16 ``.weight``: GDN in_proj fusion, Gemma (1+w) norms, else passthrough."""
        base = name[: -len(".weight")]
        emit = _ct_bf16_fuse(base, tensor, bf16_buf, _CT_BF16_FUSE)
        if emit is not None:
            yield from emit
            return
        if _is_gemma_norm(name):
            tensor = tensor + 1.0  # (1 + weight) baked into the stored norm weight
        yield name, tensor

    # Scale lookups go through the shard-map reader: a weight_packed's quant scales
    # can land in a different shard than the packed weight (the Muse-Glimmer layer-49
    # shape; nothing prevents an llm-compressor Qwen export from splitting the same way).
    reader = ShardReader(model_path, device)
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading compressed-tensors weights",
            disable=not tp_info.is_primary(),
        ):
            for raw_name in reader.names_in(file):
                if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
                    continue
                if raw_name.endswith(_CT_SCALE_SUFFIXES):
                    continue  # consumed with weight_packed (or unused W4A4 activation scales)

                name = _rename(raw_name)
                if name is None:
                    continue

                if raw_name.endswith(".weight_packed"):  # NVFP4 projection
                    base = name[: -len(".weight_packed")]
                    raw_base = raw_name[: -len(".weight_packed")]
                    w, s, g = _nvfp4_parts_ct(reader, raw_base)
                    # GDN in_proj_* compute in bf16 (model contract) but some checkpoints
                    # (e.g. sakamakismile/Qwen3.6-27B-NVFP4) quantize them too: dequant to
                    # bf16 here and let the bf16 fusion assemble ``in_proj`` as usual.
                    if any(base.endswith(p) for ps in _CT_BF16_FUSE.values() for p in ps):
                        bf16 = _dequant_nvfp4_weight(w, s, g[:1])
                        yield from _emit_bf16_weight(base + ".weight", bf16)
                        continue
                    if nvfp4:  # keep native (W4A16)
                        emit = _ct_nvfp4_fuse(base, (w, s, g), nvfp4_buf)
                        if emit is not None:
                            yield from emit
                        else:  # standalone: o_proj, linear_attn.out_proj, mlp.down_proj
                            yield base + ".weight", w
                            yield base + ".weight_scale", s
                            yield base + ".weight_global", g
                        continue
                    # bf16 A-B: dequant FP4 -> bf16, then merge q/k/v + gate/up as bf16. ``g`` is
                    # already the dequant global (1/weight_global_scale) per row; pass one element.
                    bf16 = _dequant_nvfp4_weight(w, s, g[:1])
                    emit = _ct_bf16_fuse(base, bf16, bf16_buf, _CT_NVFP4_FUSE)
                    if emit is not None:
                        yield from emit
                    else:
                        yield base + ".weight", bf16
                    continue

                if name.endswith(".weight"):
                    yield from _emit_bf16_weight(name, reader.get_tensor(raw_name))
                    continue

                # A_log / dt_bias (kept fp32 by the model; the load downcast exempts them).
                yield name, reader.get_tensor(raw_name)
    finally:
        reader.close()

    assert not nvfp4_buf, f"Incomplete NVFP4 fusions: {list(nvfp4_buf.keys())}"
    assert not bf16_buf, f"Incomplete bf16 fusions: {list(bf16_buf.keys())}"


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only parallel read via the common chunked multi-threaded O_DIRECT reader.
    Qwen3.5 stores experts pre-fused/pre-stacked per layer (already ``[E, ...]``), so no
    merge/stack -- just rename and yield; bank builder places by name as the serial path."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_5_moe parallel reader is experts-only (used by load_moe_expert_sources)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading currently supports TP=1 only")

    def _is_expert(raw_name: str) -> bool:
        name = _rename(raw_name)
        return name is not None and _PACKED_EXPERT_PATTERN.match(name) is not None

    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, _is_expert, workers=workers, chunk=chunk
    ):
        yield _rename(raw_name), tensor


# ======================================================================================
# Block-FP8 checkpoint (Qwen3.5-35B-A3B-FP8): dense weights + offload expert banks.
# ======================================================================================
# fused model buffer suffix -> ordered checkpoint part suffixes (matched without the
# trailing .weight / .weight_scale_inv). Both kinds ride the same fusion (concatenated
# along dim 0); in_proj_ba carries only .weight (b/a stay bf16, no block scale).
_FP8_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (
        ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
    ),
    ".linear_attn.in_proj_qkvz": (
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
    ),
    ".linear_attn.in_proj_ba": (
        ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
    ),
    ".mlp.shared_expert.gate_up_proj": (
        ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj",
    ),
    # Dense (non-MoE) layer MLP: merge gate|up -> gate_up_proj for both the fp8 ``.weight``
    # and the bf16 ``.weight_scale_inv`` (fused per kind by _split_kind). Only a bare
    # ``.mlp.gate_proj`` matches; the shared_expert entry above keeps the MoE case.
    ".mlp.gate_up_proj": (
        ".mlp.gate_proj", ".mlp.up_proj",
    ),
}
_FP8_KIND_SUFFIXES = (".weight_scale_inv", ".weight")

# Routed-expert checkpoint key (per-expert, un-fused). ``mtp.layers...`` is excluded by the
# ``model.language_model.`` anchor, so the parallel reader only sees the real experts.
_FP8_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
)


def _split_kind(name: str) -> tuple[str, str]:
    """``name`` -> ``(base, kind_suffix)``; ``kind_suffix`` is "" for keys without a
    weight/scale suffix (A_log, dt_bias)."""
    for suf in _FP8_KIND_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def _fp8_fuse(base: str, suf: str, tensor: torch.Tensor, buf: dict) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part keyed by (fused_full_name, kind); return the concatenated
    ``(name, tensor)`` once all parts for that kind arrive, ``()`` while incomplete,
    ``None`` if ``base`` is not a fusion part."""
    for fused_suffix, parts in _FP8_FUSIONS.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                fused_base = base[: -len(part)] + fused_suffix
                key = (fused_base + suf, suf)
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return fused_base + suf, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def _iter_weights_fp8(
    model_path: str, device: torch.device, *, include_non_moe: bool, include_moe_experts: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the block-fp8 weights, renamed + fused to the model buffers.

    fp8 weights (e4m3) and their bf16 ``weight_scale_inv`` pass through verbatim (no dtype
    cast -- the engine's load-time cast is a no-op against the fp8/bf16 model buffers).
    q/k/v -> qkv_proj, GDN in_proj_qkv|z -> in_proj_qkvz (fp8) and in_proj_b|a -> in_proj_ba
    (bf16), shared_expert gate|up -> gate_up_proj; Gemma (1+w) norms get +1.

    Routed experts: skipped under offload (loaded by setup_offload_expert_banks). Under the
    resident (non-offload) path ``include_moe_experts`` is True -> per-layer stacked fp8
    experts for the Fp8ResidentMoE buffers are yielded too."""
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe fp8 weight loading supports TP=1 only")
    if include_non_moe:
        fuse_buf: dict = {}
        for file in tqdm(iter_weight_files(model_path), desc="Loading fp8 weights",
                         disable=not get_tp_info().is_primary()):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = _rename(raw_name)
                    if name is None or ".mlp.experts." in name:
                        continue  # routed experts handled below / by the offload cache
                    tensor = f.get_tensor(raw_name)
                    base, suf = _split_kind(name)
                    fused = _fp8_fuse(base, suf, tensor, fuse_buf)
                    if fused is not None:
                        if fused != ():
                            yield fused
                        continue
                    if _is_gemma_norm(name):
                        tensor = tensor + 1.0  # (1 + weight) baked into the stored norm weight
                    yield name, tensor
        assert not fuse_buf, f"Incomplete fp8 fusions: {sorted(k for k, _ in fuse_buf)}"

    if include_moe_experts:
        # Resident (non-offload) experts: build the stacked fp8 banks once via the shared
        # parallel reader (pageable host -- the engine copies the per-layer slices to GPU),
        # then yield per-layer views into the Fp8ResidentMoE buffers.
        config = parse_config(cached_load_hf_config(model_path))
        if not config.is_moe:
            return  # dense checkpoint: no routed experts to build as resident banks
        L, E, H, I, dense = _moe_dims(config)
        banks = _build_fp8_expert_banks(model_path, config, dummy=False, pin=False)
        for li in range(L):
            pre = f"model.layers.{dense + li}.mlp.experts"
            yield f"{pre}.gate_up_proj", banks["gate_up"][li]
            yield f"{pre}.gate_up_scale_inv", banks["gate_up_scale"][li]
            yield f"{pre}.down_proj", banks["down"][li]
            yield f"{pre}.down_scale_inv", banks["down_scale"][li]


class _ShardReader:
    """Opens safetensors shards on demand and serves tensors by name on ``device``."""

    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._map = weight_map
        self._device = device
        self._handles: dict = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self._map[name]
        h = self._handles.get(shard)
        if h is None:
            h = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = h
        return h.get_tensor(name)

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        self._handles.clear()


def setup_offload_expert_banks(
    model_path: str, model_config, *, device: torch.device, dtype: torch.dtype,
    dummy: bool = False, parallel: bool = False, workers: int = 8, chunk: int = 8 << 20,
    decode_target: str = "gpu", layer_sink=None,
):
    """Build the routed-expert offload banks. The qwen3_5_moe module always exports this hook,
    so it intercepts *every* qwen3_5_moe offload load -- defer non-block-fp8 checkpoints (plain
    BF16, or modelopt NVFP4) to the generic provider for that ``expert_quant``.

    block-fp8 default ("fp8"): keep experts block-fp8 -- ``gate_up``/``down`` fp8 banks + their
    bf16 ``weight_scale_inv`` banks (half the host/cache bytes; routed rows are dequantized on
    demand in ``_expert_gemm``). ``FREETOKEN_FP8_EXPERTS=bf16`` instead dequantizes every expert
    to bf16 at load (reuses the bf16 offload path; ~2x the memory). Both modes build per-layer
    :class:`HostBank` banks (pin-after-fill), so the converter's ``layer_sink`` streams each
    completed layer straight through; ``layer_sink`` also reaches the deferred (nvfp4/none)
    providers for non-block-fp8 checkpoints.

    ``decode_target`` is forwarded so the cpu backend gets CPU-readable (native, non-
    GPU-tiled) bank layouts -- e.g. native ``nvfp4`` rows rather than marlin/b12x."""
    eq = getattr(model_config, "expert_quant", "none")
    if eq != "fp8_block":
        from freetoken.moe.expert_banks import _PROVIDERS  # nvfp4 -> _nvfp4_banks, none -> _bf16_banks

        return _PROVIDERS[eq](model_path, model_config, device, dtype, dummy,
                              parallel=parallel, workers=workers, chunk=chunk,
                              decode_target=decode_target, layer_sink=layer_sink)
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe fp8 expert banks support TP=1 only")
    from freetoken.moe.expert_banks import ExpertBanks

    mode = os.environ.get("FREETOKEN_FP8_EXPERTS", "fp8").strip().lower()
    if mode not in ("fp8", "bf16"):
        raise ValueError(f"FREETOKEN_FP8_EXPERTS must be 'fp8' or 'bf16', got {mode!r}")
    sink = None if dummy else layer_sink
    if mode == "bf16":
        return _setup_bf16_dequant_banks(model_path, model_config, device, dummy, layer_sink=sink)
    banks = _build_fp8_expert_banks(
        model_path, model_config, dummy=dummy, parallel=parallel, workers=workers, chunk=chunk,
        pin=True, layer_sink=sink,
    )
    return ExpertBanks("fp8_block", banks, streamed=sink is not None)


def _moe_dims(model_config):
    L = model_config.num_moe_layers
    return (
        L, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - L,  # dense prefix
    )


def _expert_reader(model_path, device):
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    return _ShardReader(folder, weight_map, device)


def _build_fp8_expert_banks(
    model_path, config, *, dummy: bool, parallel: bool | None = None,
    workers: int = 8, chunk: int = 8 << 20, pin: bool = True, layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """The single block-fp8 routed-expert reader, shared by offload (``pin=True`` -> per-layer
    :class:`HostBank` banks, pin-after-fill / streamable) and resident (``pin=False`` -> plain
    pageable tensors, never pinned or streamed). Reads via the common chunked multi-threaded
    O_DIRECT reader (drops page cache, no per-tensor serial overhead) when the experts are
    scattered per-tensor; else a serial shard fallback. Stacks gate|up -> gate_up into one
    ``[E, ...]`` tensor per layer per bank.

    Each expert contributes 6 ``place()`` writes per layer ({gate,up,down} x {weight,
    weight_scale_inv}), so a layer completes after ``E * 6`` writes. ``layer_sink=None``
    (serving) pins each layer's 4 banks as it completes via an internally-owned
    :class:`PinPipeline`; ``layer_sink`` given (converter) fires the completion tracker into it
    instead (nothing pinned; released banks -- caller owns that tradeoff). ``pin=False`` and the
    CUDA-less host stay on the plain materialize path (no pin, no stream)."""
    from freetoken.kernel.triton.fp8_block_linear import FP8
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel

    B = 128
    L, E, H, I, dense = _moe_dims(config)

    # 16B-align the per-expert scale rows (Qwen3.8: down_scale is 20x5 bf16 = 200 B) so the
    # fused multi-bank copy engages; the GEMMs read scales through explicit strides, so the
    # padding is inert. Unconditional: one layout per format, shared with the byte formulas.
    from freetoken.moe.offload_cache import fp8_block_scale_pad as _pad_cols

    specs = {
        "gate_up": ((E, 2 * I, H), FP8),
        "gate_up_scale": ((E, 2 * I // B, _pad_cols(2 * I // B, H // B)), torch.bfloat16),
        "down": ((E, H, I), FP8),
        "down_scale": ((E, H // B, _pad_cols(H // B, I // B)), torch.bfloat16),
    }
    hb = None
    if pin:
        from freetoken.moe.host_banks import alloc_layer_banks

        hb = alloc_layer_banks(specs, L)  # lazy anon mmaps (unpinned)
        banks = {name: [b.tensor for b in hb[name]] for name in specs}
    else:  # resident dequant source: plain pageable tensors (never pinned / streamed)
        banks = {name: [torch.empty(shape, dtype=dt) for _ in range(L)] for name, (shape, dt) in specs.items()}
    gate_up, gate_up_scale, down, down_scale = (
        banks["gate_up"], banks["gate_up_scale"], banks["down"], banks["down_scale"]
    )
    if dummy:
        for li in range(L):
            gate_up[li].view(torch.uint8).random_(0, 16)  # small fp8 codes (avoid NaN/inf)
            down[li].view(torch.uint8).random_(0, 16)
            gate_up_scale[li].fill_(1.0)
            down_scale[li].fill_(1.0)
        if hb is not None and torch.cuda.is_available():
            from freetoken.moe.host_banks import pin_banks

            pin_banks(hb)  # pin-after-fill (match the other dummies)
        return banks

    def place(raw_name: str, t: torch.Tensor) -> int | None:
        m = _FP8_EXPERT_RE.match(raw_name)
        if m is None:
            return None
        li, e = int(m["layer"]) - dense, int(m["expert"])
        proj, kind = m["proj"], m["kind"]
        if kind == "weight":
            (gate_up[li][e, :I] if proj == "gate" else
             gate_up[li][e, I:] if proj == "up" else down[li][e]).copy_(t)
        else:  # weight_scale_inv
            (gate_up_scale[li][e, : I // B, : H // B] if proj == "gate" else
             gate_up_scale[li][e, I // B :, : H // B] if proj == "up" else
             down_scale[li][e, :, : I // B]).copy_(t)
        return li

    if parallel is None:
        parallel = experts_scattered(model_path)

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> None:
        # {gate,up,down} x {weight, weight_scale_inv} per expert -> E*6 writes/layer.
        tracker = LayerCompletionTracker(E * 6, hb, sink) if sink is not None else None
        if parallel:
            for raw_name, t in iter_expert_tensors_parallel(
                model_path, lambda n: _FP8_EXPERT_RE.match(n) is not None, workers=workers, chunk=chunk
            ):
                li = place(raw_name, t)
                if tracker is not None and li is not None:
                    tracker.note(li)
        else:
            reader = _expert_reader(model_path, torch.device("cpu"))
            primary = get_tp_info().is_primary()
            try:
                for li in tqdm(range(L), desc="Loading fp8 experts (serial)", disable=not primary):
                    layer = dense + li
                    for e in range(E):
                        p = f"model.language_model.layers.{layer}.mlp.experts.{e}"
                        for proj in ("gate", "up", "down"):
                            for kind in ("weight", "weight_scale_inv"):
                                key = f"{p}.{proj}_proj.{kind}"
                                place(key, reader.get(key))
                                if tracker is not None:
                                    tracker.note(li)
            finally:
                reader.close()

    if not pin:
        assert layer_sink is None, "pin=False (resident source) cannot stream to a layer_sink"
        _load(None)
    elif layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned
    return banks


def _setup_bf16_dequant_banks(model_path, model_config, device, dummy: bool, *, layer_sink=None):
    from freetoken.models.weight import dummy_moe_expert_sources
    from freetoken.moe.expert_banks import ExpertBanks

    if dummy:
        gate_up, down = dummy_moe_expert_sources(model_config, dtype=torch.bfloat16)
        return ExpertBanks("bf16", {"gate_up": gate_up, "down": down})

    from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    L, E, H, I, dense = _moe_dims(model_config)
    specs = {"gate_up": ((E, 2 * I, H), torch.bfloat16), "down": ((E, H, I), torch.bfloat16)}
    hb = alloc_layer_banks(specs, L)  # lazy anon mmaps (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}
    gate_up, down = banks["gate_up"], banks["down"]
    reader = _expert_reader(model_path, device)
    primary = get_tp_info().is_primary()

    def _deq(prefix: str) -> torch.Tensor:
        return dequant_block_fp8(reader.get(f"{prefix}.weight"), reader.get(f"{prefix}.weight_scale_inv"))

    def _load(sink) -> None:
        # Whole-layer gate_up + down copies -> 2 writes/layer.
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        try:
            for li in tqdm(range(L), desc="Loading fp8 experts (dequant->bf16)", disable=not primary):
                layer = dense + li
                gu_rows = torch.empty(E, 2 * I, H, dtype=torch.bfloat16, device=device)
                dn_rows = torch.empty(E, H, I, dtype=torch.bfloat16, device=device)
                for e in range(E):
                    p = f"model.language_model.layers.{layer}.mlp.experts.{e}"
                    gu_rows[e, :I] = _deq(f"{p}.gate_proj")
                    gu_rows[e, I:] = _deq(f"{p}.up_proj")
                    dn_rows[e] = _deq(f"{p}.down_proj")
                gate_up[li].copy_(gu_rows)
                if tracker is not None:
                    tracker.note(li)
                down[li].copy_(dn_rows)
                if tracker is not None:
                    tracker.note(li)
        finally:
            reader.close()

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned
    return ExpertBanks("bf16", banks, streamed=layer_sink is not None)


def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, torch.Tensor]:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the
    output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
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
    )


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "setup_offload_expert_banks",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
