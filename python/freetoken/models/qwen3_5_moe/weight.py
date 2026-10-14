"""Qwen3.5 / 3.6 / 3.8 checkpoint reader.

The dense pass reads every Linear module under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, block-fp8, ModelOpt and llm-compressor exports in any mix all land as the model's state dict. Routed experts are read by the expert-bank loader (``nvfp4_expert_spec`` / ``iter_expert_pieces``); only the bf16 stacked experts it reads come from here.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import ShardReader, iter_weight_files
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.models.register import ModelSpec, get_model_spec
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config, parse_gguf_config
from freetoken.models.qwen3_vl.weight import rename_vl_prefix


from freetoken.models.gguf.nvfp4 import nvfp4_parts


def _gguf_nvfp4_parts(t, *, dense: bool = False):
    return nvfp4_parts(t, dense=dense)


def _gguf_bf16(t) -> torch.Tensor:
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape)


def _gguf_gdn_state(t) -> torch.Tensor:
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32).reshape(t.shape)


def _ungroup_v(tensor: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int):
    """Undo llama.cpp's tiled V-head order and restore FreeToken's grouped order."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    view = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    permutation = list(range(len(view)))
    permutation[dim], permutation[dim + 1] = permutation[dim + 1], permutation[dim]
    return tensor.reshape(*view).permute(*permutation).contiguous().reshape(*shape)


def _gguf_any_dequant(tensor, device: torch.device) -> torch.Tensor:
    """Materialize one GGUF projection that needs a column permutation."""
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    block, type_size = BLOCK_SHAPE[tensor.ggml_type]
    packed = tensor.packed().to(device).contiguous()
    return ggml_dequantize(
        packed, tensor.ggml_type, tensor.shape[0], tensor.shape[1], torch.bfloat16
    )


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read Qwen3.5's native NVFP4 GGUF without routing it through HF shards."""
    from freetoken.models.gguf.reader import GGML_NVFP4, iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen3.5 GGUF weight loading supports TP=1 only")
    if include_moe_experts:
        raise AssertionError("routed GGUF experts are loaded by the NVFP4 expert-bank provider")
    if not include_non_moe:
        return

    # GGUF block_count includes the trailing NextN/MTP block for this checkpoint, while
    # text-only FreeToken serving constructs only the decoder layers. Keep the iterator's
    # cutoff identical to parse_gguf_config so block 40 cannot become an unexpected key.
    gguf_config = parse_gguf_config(cached_load_hf_config(model_path))
    served_layers = gguf_config.num_layers

    fusions: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    legacy_gate_up: dict[int, dict[str, torch.Tensor]] = {}
    bf16_fusions: dict[str, dict[str, torch.Tensor]] = {}
    num_k_heads = 16
    num_v_heads = 32
    v_per_k = num_v_heads // num_k_heads
    v_head_dim = 128
    qk_rows = 2 * num_k_heads * v_head_dim

    def emit_fused(target: str, parts: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], order):
        values = [parts[name] for name in order]
        yield target + ".weight", torch.cat([v[0] for v in values], dim=0).to(device)
        yield target + ".weight_scale", torch.cat([v[1] for v in values], dim=0).to(device)
        yield target + ".weight_global", torch.cat([v[2] for v in values], dim=0).to(device)

    for tensor in iter_gguf_tensors(model_path):
        raw = tensor.name
        if raw.startswith("model.visual.") and not include_vision:
            continue
        if raw.startswith("model.visual."):
            continue
        if raw == "token_embd.weight":
            yield "model.embed_tokens.qweight", tensor.packed().to(device)
            continue
        if raw == "output.weight":
            yield "lm_head.qweight", tensor.packed().to(device)
            continue
        if raw == "output_norm.weight":
            yield "model.norm.weight", _gguf_bf16(tensor).to(device)
            continue
        if raw == "model.language_model.embed_tokens.weight":
            if tensor.ggml_type == GGML_NVFP4:
                packed, scales = _gguf_nvfp4_parts(tensor)
                yield "model.embed_tokens.weight", packed.to(device)
                yield "model.embed_tokens.weight_scale", scales.to(device)
                yield "model.embed_tokens.weight_global", torch.ones(tensor.shape[0], dtype=torch.float16, device=device)
            else:
                yield "model.embed_tokens.weight", _gguf_bf16(tensor).to(device)
            continue
        if raw == "lm_head.weight":
            if tensor.ggml_type == GGML_NVFP4:
                packed, scales = _gguf_nvfp4_parts(tensor, dense=True)
                yield "lm_head.weight", packed.to(device)
                yield "lm_head.weight_scale", scales.to(device)
                yield "lm_head.weight_global", torch.ones(tensor.shape[0], dtype=torch.float16, device=device)
            else:
                yield "lm_head.weight", _gguf_bf16(tensor).to(device)
            continue
        if raw == "model.language_model.norm.weight":
            yield "model.norm.weight", _gguf_bf16(tensor).to(device) + 1.0
            continue
        if raw.startswith("blk."):
            # Legacy GGML Qwen35MoE names are mapped to the model's HF-style tree.
            layer, _, suffix = raw[4:].partition(".")
            name = f"model.layers.{layer}."
            layer_id = int(layer)
            if layer_id >= served_layers:
                continue
            if suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
                slot = "gate" if suffix.startswith("ffn_gate") else "up"
                parts = legacy_gate_up.setdefault(layer_id, {})
                parts[slot] = tensor.packed().to(device)
                if len(parts) == 2:
                    yield f"model.layers.{layer}.mlp.shared_expert.gate_up_proj.qweight", torch.cat([parts["gate"], parts["up"]], dim=0)
                    del legacy_gate_up[layer_id]
                continue
            legacy = {
                "attn_norm.weight": "input_layernorm.weight",
                "post_attention_norm.weight": "post_attention_layernorm.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "ssm_a": "linear_attn.A_log",
                "ssm_alpha.weight": "linear_attn.in_proj.part3.qweight",
                "ssm_beta.weight": "linear_attn.in_proj.part2.qweight",
                "ssm_conv1d.weight": "linear_attn.conv1d.weight",
                "ssm_dt.bias": "linear_attn.dt_bias",
                "ssm_norm.weight": "linear_attn.norm.weight",
                "ssm_out.weight": "linear_attn.out_proj.weight",
                "attn_qkv.weight": "linear_attn.in_proj.part0.qweight",
                "attn_gate.weight": "linear_attn.in_proj.part1.qweight",
                "ffn_gate_inp.weight": "mlp.gate.weight",
                "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
                "ffn_gate_shexp.weight": "mlp.shared_expert.gate_up_proj.qweight",
                "ffn_up_shexp.weight": "mlp.shared_expert.gate_up_proj.qweight",
                "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.qweight",
            }
            mapped = legacy.get(suffix)
            if mapped is not None:
                name += mapped
            elif suffix in ("attn_k.weight", "attn_q.weight", "attn_v.weight", "attn_output.weight"):
                name += {"attn_k.weight": "self_attn.qkv_proj.part1.qweight", "attn_q.weight": "self_attn.qkv_proj.part0.qweight", "attn_v.weight": "self_attn.qkv_proj.part2.qweight", "attn_output.weight": "self_attn.o_proj.qweight"}[suffix]
            elif suffix.endswith("_exps.weight"):
                continue
            else:
                continue
            if suffix == "ssm_a":
                value = _gguf_gdn_state(tensor).to(device)
                value = _ungroup_v(value, 0, num_k_heads, v_per_k, 1)
                if not bool((value < 0).all()):
                    raise ValueError(f"{raw}: expected pre-transformed negative -exp(A_log) values")
                yield name, torch.log(-value)
                continue
            if suffix == "ssm_out.weight":
                value = _gguf_any_dequant(tensor, device)
                value = _ungroup_v(value, 1, num_k_heads, v_per_k, v_head_dim)
                yield name, value
                continue
            if suffix == "ssm_conv1d.weight":
                value = _gguf_gdn_state(tensor).to(device)
                qk, v = value[:qk_rows], value[qk_rows:]
                v = _ungroup_v(v, 0, num_k_heads, v_per_k, v_head_dim)
                yield name, torch.cat([qk, v], dim=0).reshape(qk_rows + v.shape[0], 1, -1)
                continue
            if tensor.ggml_type in (12, 14):
                if suffix == "attn_qkv.weight":
                    packed = tensor.packed().to(device)
                    packed = torch.cat(
                        [packed[:qk_rows], _ungroup_v(packed[qk_rows:], 0, num_k_heads, v_per_k, v_head_dim)],
                        dim=0,
                    )
                    yield name, packed
                    continue
                if suffix == "attn_gate.weight":
                    packed = _ungroup_v(tensor.packed().to(device), 0, num_k_heads, v_per_k, v_head_dim)
                    yield name, packed
                    continue
                if suffix in ("ssm_alpha.weight", "ssm_beta.weight"):
                    packed = _ungroup_v(tensor.packed().to(device), 0, num_k_heads, v_per_k, 1)
                    yield name, packed
                    continue
                yield name, tensor.packed().to(device)
            elif tensor.ggml_type != GGML_NVFP4:
                value = _gguf_gdn_state(tensor) if any(x in name for x in ("A_log", "dt_bias", "conv1d", ".norm.weight")) else _gguf_bf16(tensor)
                if name.endswith("linear_attn.conv1d.weight") and value.ndim == 2:
                    value = value.unsqueeze(1)
                if name.endswith("mlp.shared_expert_gate.weight") and value.ndim == 1:
                    value = value.unsqueeze(0)
                yield name, value.to(device)
            else:
                packed, scales = _gguf_nvfp4_parts(tensor, dense=True)
                yield name, packed.to(device)
                yield name.removesuffix(".weight") + ".weight_scale", scales.to(device)
                yield name.removesuffix(".weight") + ".weight_global", torch.ones(packed.shape[0], dtype=torch.float16, device=device)
            continue
        if not raw.startswith("model.language_model.layers."):
            continue

        _, _, rest = raw.partition("model.language_model.")
        name = "model." + rest
        if ".mlp.experts." in name:
            continue
        if tensor.ggml_type != GGML_NVFP4:
            packed = tensor.packed().to(device)
            if name.endswith(".linear_attn.A_log"):
                value = _gguf_gdn_state(tensor).to(device)
                yield name, value
                continue
            if name.endswith(".linear_attn.conv1d.weight"):
                value = _gguf_gdn_state(tensor).to(device)
                yield name, value.reshape(value.shape[0], 1, -1)
                continue
            if name.endswith(".linear_attn.out_proj.weight"):
                value = _gguf_bf16(tensor).to(device)
                yield name, value
                continue
            fusion_target = None
            fusion_slot = None
            for suffix, target_suffix, slot in (
                (".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj", "qkv"),
                (".linear_attn.in_proj_z.weight", ".linear_attn.in_proj", "z"),
                (".linear_attn.in_proj_b.weight", ".linear_attn.in_proj", "b"),
                (".linear_attn.in_proj_a.weight", ".linear_attn.in_proj", "a"),
                (".self_attn.q_proj.weight", ".self_attn.qkv_proj", "q"),
                (".self_attn.k_proj.weight", ".self_attn.qkv_proj", "k"),
                (".self_attn.v_proj.weight", ".self_attn.qkv_proj", "v"),
                (".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.gate_up_proj", "gate"),
                (".mlp.shared_expert.up_proj.weight", ".mlp.shared_expert.gate_up_proj", "up"),
            ):
                if name.endswith(suffix):
                    fusion_target = name.removesuffix(suffix) + target_suffix
                    fusion_slot = slot
                    break
            if fusion_target is not None:
                bf16_fusions.setdefault(fusion_target, {})[fusion_slot] = packed
                continue
            if name.endswith((
                ".self_attn.o_proj.weight",
                ".mlp.shared_expert.down_proj.weight",
            )):
                yield name.removesuffix(".weight") + ".qweight", packed
                continue
            if name.endswith(".linear_attn.out_proj.weight"):
                yield name, _gguf_bf16(tensor).to(device)
                continue
            if name.endswith((".linear_attn.conv1d.weight", ".linear_attn.A_log", ".linear_attn.dt_bias", ".linear_attn.norm.weight")):
                yield name, _gguf_gdn_state(tensor).to(device)
                continue
            if name.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight", ".self_attn.q_norm.weight", ".self_attn.k_norm.weight")):
                yield name, _gguf_bf16(tensor).to(device) + 1.0
            elif name.endswith(".linear_attn.norm.weight"):
                yield name, _gguf_bf16(tensor).to(device)
            else:
                yield name, _gguf_bf16(tensor).to(device)
            continue

        # Routers stay as dense BF16 modules in Qwen3.5 even when the GGUF stores
        # them in NVFP4; only the large expert and projection GEMMs use native FP4 ops.
        if name.endswith((".mlp.gate.weight", ".mlp.shared_expert_gate.weight")):
            yield name, _gguf_bf16(tensor).to(device)
            continue

        packed, scales = _gguf_nvfp4_parts(tensor, dense=True)
        values = (packed, scales, torch.ones(packed.shape[0], dtype=torch.float16))
        if name.endswith(".linear_attn.in_proj_qkv.weight"):
            target, slot = name.removesuffix(".in_proj_qkv.weight") + ".in_proj", "qkv"
        elif name.endswith(".linear_attn.in_proj_z.weight"):
            target, slot = name.removesuffix(".in_proj_z.weight") + ".in_proj", "z"
        elif name.endswith(".linear_attn.in_proj_b.weight"):
            target, slot = name.removesuffix(".in_proj_b.weight") + ".in_proj", "b"
        elif name.endswith(".linear_attn.in_proj_a.weight"):
            target, slot = name.removesuffix(".in_proj_a.weight") + ".in_proj", "a"
        elif name.endswith(".self_attn.q_proj.weight"):
            target, slot = name.removesuffix(".q_proj.weight") + ".qkv_proj", "q"
        elif name.endswith(".self_attn.k_proj.weight"):
            target, slot = name.removesuffix(".k_proj.weight") + ".qkv_proj", "k"
        elif name.endswith(".self_attn.v_proj.weight"):
            target, slot = name.removesuffix(".v_proj.weight") + ".qkv_proj", "v"
        elif name.endswith(".mlp.shared_expert.gate_proj.weight"):
            target, slot = name.removesuffix(".gate_proj.weight") + ".gate_up_proj", "gate"
        elif name.endswith(".mlp.shared_expert.up_proj.weight"):
            target, slot = name.removesuffix(".up_proj.weight") + ".gate_up_proj", "up"
        else:
            target = name.removesuffix(".weight")
            yield from emit_fused(target, {"one": values}, ["one"])
            continue
        parts = fusions.setdefault(target, {})
        parts[slot] = values
        order = ("qkv", "z", "b", "a") if target.endswith(".linear_attn.in_proj") else ("q", "k", "v") if target.endswith(".qkv_proj") else ("gate", "up")
        if all(part in parts for part in order):
            yield from emit_fused(target, parts, order)
            del fusions[target]
    for target, parts in bf16_fusions.items():
        order = (
            ("qkv", "z", "b", "a")
            if target.endswith(".linear_attn.in_proj")
            else ("q", "k", "v")
            if target.endswith(".qkv_proj")
            else ("gate", "up")
        )
        missing = [part for part in order if part not in parts]
        if missing:
            raise ValueError(f"incomplete BF16 Qwen GGUF fusion {target}: missing {missing}")
        yield target + ".qweight", torch.cat([parts[part] for part in order], dim=0)
    missing = {target: sorted(parts) for target, parts in fusions.items()}
    if missing:
        raise ValueError(f"incomplete Qwen3.5 GGUF fused tensors: {missing}")

# bf16 checkpoints store the routed experts pre-stacked per layer
_STACKED_EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")
# per-expert tensors of a quantized checkpoint: the offload cache's expert reader takes these
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# the ``model.language_model.`` anchor excludes the MTP head's ``mtp.layers.N.mlp.experts.*``
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2"}

# Gemma-style (1+weight) RMSNorm weights; the GDN gated norm (linear_attn.norm) is a plain weight*x norm
_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)
# leaves the model builds as Linear layers: only their tensors are read under the QuantConfig, the rest passes through as stored
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "in_proj_qkvz", "in_proj_ba", "out_proj",
    "gate_proj", "up_proj", "down_proj", "gate", "shared_expert_gate", "lm_head",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _is_gemma_norm(name: str) -> bool:
    return name == "model.norm.weight" or name.endswith(_GEMMA_NORM_SUFFIXES)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` fp8 scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"fp8 weight_scale has {flat.numel()} elements for a weight with {rows} output rows "
            f"(shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


def _dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_global: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 -> bf16 on CUDA (the kernel is GPU-only, the converter reads on CPU), returned on the caller's device."""
    device = weight.device
    if device.type != "cuda":
        weight, weight_scale, weight_global = (t.to("cuda") for t in (weight, weight_scale, weight_global))
    slots = torch.zeros(1, dtype=torch.int32, device=weight.device)
    out = dequant_nvfp4(
        weight.unsqueeze(0).contiguous(), weight_scale.unsqueeze(0).contiguous(), weight_global.unsqueeze(0),
        slots, dtype=torch.bfloat16,
    )[0]
    return out.to(device)


def _dequant(scheme: QuantScheme, part: dict[str, torch.Tensor]) -> torch.Tensor:
    """bf16 weight of a module the checkpoint quantized but the family serves unquantized."""
    weight = part["weight"]
    if scheme.kind is QuantKind.FP8_TENSOR:
        return (weight.to(torch.float32) * part["weight_scale"][:, None]).to(torch.bfloat16)
    if scheme.kind is QuantKind.FP8_BLOCK:
        from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

        return dequant_block_fp8(weight, part["weight_scale_inv"])
    if scheme.kind is QuantKind.NVFP4:
        return _dequant_nvfp4(weight, part["weight_scale"], part["weight_global"])
    raise NotImplementedError(f"no bf16 dequantization for {scheme}")


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in."""

    def __init__(self, quant: QuantConfig | None, spec: ModelSpec) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in spec.packed_modules_mapping if fused != "experts"}
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores})
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def stored(self, module: str) -> QuantScheme | None:
        """The scheme the checkpoint stores ``module`` under, before the family's unquantized_modules."""
        if self.quant is None:
            return None
        checkpoint_names = self.quant.name_map.to_checkpoint(module)
        if module.endswith(".in_proj_qkvz"):
            checkpoint_names += (
                module.removesuffix(".in_proj_qkvz") + ".in_proj_qkv",
                module.removesuffix(".in_proj_qkvz") + ".in_proj_z",
            )
        for name in checkpoint_names:
            scheme = self.quant.scheme_for_name(name)
            if scheme is not None:
                return scheme
        # Multimodal Qwen checkpoints can store tensors below model.language_model
        # while ModelOpt metadata names the logical text module as model.layers.
        logical_name = module.removeprefix("model.language_model.")
        return self.quant.scheme_for_name(logical_name)

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as gdn.py
            def has_scheme(name: str) -> bool:
                return (
                    self.scheme(name) is not None
                    or (self.quant is not None and self.quant.scheme_for_name(name) is not None)
                )

            split = any(
                has_scheme(f"{parent}.{leaf}")
                for leaf in ("in_proj_qkvz", "in_proj_qkv", "in_proj_z")
            )
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        stored = self.stored(module)
        roles = {"weight": "weight"} if stored is None else {e.name: r for r, e in self.quant.storage(stored).items()}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(
                f"{name}: the checkpoint's quant config declares {module} {stored or 'unquantized'}, stored as {sorted(roles)}"
            )
        if stored is None and tensor.dtype in _QUANT_DTYPES:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
        if stored is not None and role == "weight" and tensor.dtype is not _ELEM_DTYPES[stored.weight.elem]:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {stored}")
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, stored))
        parts.setdefault(idx, {})[role] = tensor
        expected[idx] = set(roles.values())
        required = {
            i: roles - {"input_scale"} for i, roles in expected.items()
        }
        if len(parts) < count or any(not required[i].issubset(parts[i]) for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], stored)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, stored) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            note = ""
            if lacking == ["input_scale"]:
                fix = "declares W4A16_NVFP4 or sets with_input_scale false" if stored is not None and stored.kind is QuantKind.NVFP4 else "sets with_input_scale false"
                note = f" (an export without activation scales {fix})"
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}{note}")
        return lines

    def _emit(self, target: str, parts: list[dict[str, torch.Tensor]], stored: QuantScheme | None):
        if stored is not None:
            parts = [self._check(target, stored, part) for part in parts]
            if self.scheme(target) is None:
                parts = [{"weight": _dequant(stored, part)} for part in parts]
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so ModelOpt calibrates one range for them: max is exact then and safe if they drift
                value = torch.stack(tensors).max()
            else:
                value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            out.append((f"{target}.{role}", value))
        return out

    def _check(self, target: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one part against ``scheme`` and put its scales in the layer's form."""
        part = {
            role: 1.0 / tensor.to(torch.float32) if self.quant.storage(scheme)[role].reciprocal else tensor
            for role, tensor in part.items()
        }
        weight = part["weight"]
        if weight.dtype is not _ELEM_DTYPES[scheme.weight.elem]:
            raise ValueError(f"{target}: weight is {weight.dtype} but the checkpoint's quant config declares {scheme}")
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            out[scale_role] = _per_row_scale(part[scale_role], rows)
        else:
            if rows % block_rows or cols % block_cols:
                raise ValueError(f"{target}: {rows}x{cols} weight is not a multiple of the {block_rows}x{block_cols} scale block of {scheme}")
            expected = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != expected:
                raise ValueError(f"{target}: {scale_role} is {tuple(part[scale_role].shape)}, expected {expected} for {scheme}")
            if scheme.weight.scale == "e4m3" and part[scale_role].dtype is not torch.float8_e4m3fn:
                raise ValueError(f"{target}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{target}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.to(torch.float16).expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense weights fused to the model's buffers, and with ``include_moe_experts`` the bf16 stacked experts as stored.

    Block-fp8 and NVFP4 experts always come from the expert-bank reader.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading supports TP=1 only")
    from freetoken.models.gguf.reader import is_gguf_path

    if is_gguf_path(model_path):
        yield from iter_gguf_weights(
            model_path,
            device,
            include_moe_experts=include_moe_experts,
            include_non_moe=include_non_moe,
            include_vision=include_vision,
        )
        return
    hf_config = cached_load_hf_config(model_path)
    config = parse_gguf_config(hf_config) if hasattr(hf_config, "metadata") else parse_config(hf_config)
    stacked = include_moe_experts and config.is_moe and config.expert_quant == "none"
    if include_non_moe or stacked:
        reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0])) if include_non_moe else None
        yield from _iter_shards(model_path, device, reader, stacked=stacked, include_vision=include_vision)


def _iter_shards(model_path: str, device: torch.device, reader: _DenseReader | None, *, stacked: bool, include_vision: bool):
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None or _EXPERT_RE.search(name):
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                if _STACKED_EXPERT_RE.match(name):
                    if stacked:
                        yield name, f.get_tensor(raw_name)
                    continue
                if reader is None:
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(name, tensor)
                if emitted is not None:
                    yield from emitted
                elif _is_gemma_norm(name):
                    yield name, tensor + 1.0  # (1 + weight) baked into the stored norm weight
                else:
                    yield name, tensor
    if reader is not None and reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


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
        "qwen3_5_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe weight loading currently supports TP=1 only")

    def _is_expert(raw_name: str) -> bool:
        name = _rename(raw_name)
        return name is not None and _STACKED_EXPERT_RE.match(name) is not None

    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, _is_expert, workers=workers, chunk=chunk
    ):
        yield _rename(raw_name), tensor


# ======================================================================================
# Block-FP8 routed experts (Qwen3.5-35B-A3B-FP8): expert pieces.
# ======================================================================================

# Routed-expert checkpoint key (per-expert, un-fused). ``mtp.layers...`` is excluded by the
# ``model.language_model.`` anchor, so the parallel reader only sees the real experts.
_FP8_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|{scale})$"
)


def _moe_dims(model_config):
    L = model_config.num_moe_layers
    return (
        L, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - L,  # dense prefix
    )


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their
    ``_scale`` (block scale) companions, named as the checkpoint's dialect stores them. Other expert kinds use the generic readers."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe fp8 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    L, E, H, I, dense = _moe_dims(config)
    scale = get_quant_config().stored_tensors(QuantKind.FP8_BLOCK)["weight_scale_inv"].name
    key_re = re.compile(_FP8_EXPERT_KEY_RE.format(scale=re.escape(scale)))
    suffix = {"weight": "", scale: "_scale"}

    def locate(raw_name: str):
        m = key_re.match(raw_name)
        if m is None:
            return None
        li = int(m["layer"]) - dense
        if not 0 <= li < L:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        return li, int(m["expert"]), m["proj"] + suffix[m["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: key_re.match(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = ShardReader(model_path, torch.device("cpu"))
        try:
            for li in tqdm(range(L), desc="Loading fp8 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(E):
                    base = f"model.language_model.layers.{dense + li}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind, suf in suffix.items():
                            name = f"{base}.{proj}_proj.{kind}"
                            yield name, reader.get_tensor(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.5 NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    """Load the stacked NVFP4 expert banks used by the Qwen GGUF dialect."""
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L, E, H, I = config.num_layers, config.num_experts, config.hidden_size, config.moe_intermediate_size
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
    seen = set()
    for tensor in iter_gguf_tensors(model_path):
        prefix = "model.language_model.layers."
        if not tensor.name.startswith(prefix) or ".mlp.experts." not in tensor.name:
            continue
        layer = int(tensor.name[len(prefix):].split(".", 1)[0])
        packed, scales = _gguf_nvfp4_parts(tensor)
        if tensor.name.endswith(".mlp.experts.gate_up_proj"):
            banks["gate_up_packed"][layer].copy_(packed.reshape(E, 2 * I, H // 2))
            banks["gate_up_scale"][layer].copy_(scales.reshape(E, 2 * I, H // 16))
            banks["gate_up_global"][layer].fill_(1)
            seen.add((layer, "gate_up"))
        elif tensor.name.endswith(".mlp.experts.down_proj"):
            banks["down_packed"][layer].copy_(packed.reshape(E, H, I // 2))
            banks["down_scale"][layer].copy_(scales.reshape(E, H, I // 16))
            banks["down_global"][layer].fill_(1)
            seen.add((layer, "down"))
    expected = {(layer, kind) for layer in range(L) for kind in ("gate_up", "down")}
    if seen != expected:
        raise ValueError(f"missing Qwen3.5 GGUF expert banks: {sorted(expected - seen)}")
    if layer_sink is None and torch.cuda.is_available():
        pin_banks(hb)
    return banks


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "iter_expert_pieces",
    "nvfp4_expert_spec",
    "load_nvfp4_expert_sources",
]
