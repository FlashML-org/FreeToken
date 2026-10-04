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
from freetoken.models.loader import ShardReader, iter_weight_files, shard_tensor
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.models.register import ModelSpec, get_model_spec
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

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
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
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
        return self.quant.scheme_for_name(self.quant.name_map.to_checkpoint(module)[0])

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
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
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
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


class _TPShard:
    """Per-rank tensor-parallel sharding for the dense pass. ``gdn`` is the GDN group's
    ``(num_key_heads, num_value_heads, key_head_dim, value_head_dim)`` (full, pre-TP) or
    None when the model has no linear-attention layers."""

    def __init__(self, rank: int, world: int, num_kv_heads: int, gdn: tuple | None):
        self.rank = rank
        self.world = world
        self.num_kv_heads = num_kv_heads
        self.gdn = gdn

    def __call__(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return tensor
        rank, world = self.rank, self.world
        # Vision tower (Qwen3-VL) is fully TP-parallel: qkv/fc1 column-parallel (incl. bias),
        # proj/fc2 row-parallel. No GQA, dims divide evenly, so plain chunk suffices. The
        # patch-embed conv, norms and row-parallel biases replicate (default pass-through).
        if name.startswith("visual."):
            if name.endswith((".attn.qkv.weight", ".attn.qkv.bias")):
                return torch.cat([x.chunk(world, 0)[rank] for x in tensor.chunk(3, 0)], 0).clone()
            if name.endswith((".mlp.linear_fc1.weight", ".mlp.linear_fc1.bias",
                              ".merger.linear_fc1.weight", ".merger.linear_fc1.bias")):
                return tensor.chunk(world, 0)[rank].clone()
            if name.endswith((".attn.proj.weight", ".mlp.linear_fc2.weight", ".merger.linear_fc2.weight")):
                return tensor.chunk(world, 1)[rank].clone()
            return tensor
        g = self.gdn
        if g is not None:
            nk, nv = g[0], g[1]
            # Match the GDN module regardless of role: an fp8_block checkpoint stores a
            # .weight_scale_inv (128-block) companion next to each fp8 .weight, and it must be
            # sharded the same way. The head helpers derive the per-head size from the tensor
            # itself (head_dim for the weight, head_dim/128 for the scale), so one rule serves both.
            leaf = _module_leaf(name)
            # in_proj_qkv / conv1d rows are [q(nk heads) | k(nk heads) | v(nv heads)]
            if leaf in ("in_proj_qkv", "conv1d"):
                return _shard_head_segments(tensor, [nk, nk, nv], self.rank, self.world)
            # z (nv heads), b / a / dt_bias / A_log (one unit per v-head) -> shard the v-heads
            if leaf in ("in_proj_z", "in_proj_b", "in_proj_a", "A_log", "dt_bias"):
                return _shard_by_heads(tensor, nv, self.rank, self.world, dim=0)
            # out_proj is row-parallel: its input columns are the v-head value dim
            if leaf == "out_proj":
                return _shard_by_heads(tensor, nv, self.rank, self.world, dim=1)
        # standard attention q/k/v/o, dense FFN gate/up/down, embed/lm_head vocab (shard_tensor
        # keys on the module substring, so it shards the fp8 .weight_scale_inv companion too)
        return shard_tensor(name, tensor, rank=self.rank, world_size=self.world, num_kv_heads=self.num_kv_heads)

    def expert(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Shard a pre-stacked routed-expert tensor by the intermediate dim (column-parallel
        experts, row-parallel down). gate_up is [E, 2*I, H] = [gate(I) | up(I)] on dim1 (qwen3_5
        is non-interleaved), down is [E, H, I]. The same split works for the block-fp8 companions:
        gate_up_scale_inv [E, 2*I//128, H//128] and down_scale_inv [E, H//128, I//128]."""
        if self.world == 1:
            return tensor
        if name.endswith((".gate_up_proj", ".gate_up_scale_inv")):
            gate, up = tensor.chunk(2, dim=1)
            return torch.cat(
                [gate.chunk(self.world, 1)[self.rank], up.chunk(self.world, 1)[self.rank]], dim=1
            ).clone()
        if name.endswith((".down_proj", ".down_scale_inv")):
            return tensor.chunk(self.world, dim=2)[self.rank].clone()
        return tensor


def _module_leaf(name: str) -> str:
    """The module name's leaf, with any quant-role suffix stripped (so ``...in_proj_qkv.weight``
    and ``...in_proj_qkv.weight_scale_inv`` both resolve to ``in_proj_qkv``)."""
    for suf in (".weight_scale_inv", ".weight_scale", ".weight", ".bias"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name.rpartition(".")[2]


def _shard_head_segments(t: torch.Tensor, head_counts, rank: int, world: int) -> torch.Tensor:
    """Shard dim0 of a tensor whose rows are consecutive per-head segments (head counts in
    ``head_counts``), by head. The per-head unit is derived from the tensor so one call serves
    both the fp8 weight (unit=head_dim) and its 128-block scale (unit=head_dim/128); this needs
    head_k_dim==head_v_dim (gdn.py asserts it). GDN requires every segment's head count to divide
    evenly by tp (gdn.py enforces it), matching the in_proj layer's plain per-segment split."""
    total_heads = sum(head_counts)
    assert t.shape[0] % total_heads == 0, f"dim0 {t.shape[0]} not a multiple of {total_heads} heads"
    unit = t.shape[0] // total_heads
    out, off = [], 0
    for num_heads in head_counts:
        width = num_heads * unit
        out.append(_shard_by_heads(t.narrow(0, off, width), num_heads, rank, world, dim=0))
        off += width
    return torch.cat(out, dim=0)


def _shard_by_heads(t: torch.Tensor, num_heads: int, rank: int, world: int, *, dim: int) -> torch.Tensor:
    """Slice ``dim`` (size ``num_heads * head_dim``) to this rank's heads; replicate one head
    per rank when ``num_heads < world``."""
    head_dim = t.shape[dim] // num_heads
    if num_heads < world:
        assert world % num_heads == 0, f"{world=} not a multiple of {num_heads=} for replication"
        h0 = rank * num_heads // world
        return t.narrow(dim, h0 * head_dim, head_dim).clone()
    assert num_heads % world == 0, f"{num_heads=} not divisible by {world=}"
    local = num_heads // world
    return t.narrow(dim, rank * local * head_dim, local * head_dim).clone()


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
    tp = get_tp_info()
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    if tp.size > 1 and include_moe_experts and config.is_moe and config.expert_quant not in ("none", "fp8_block"):
        raise NotImplementedError(
            f"qwen3_5_moe TP expert sharding supports bf16 stacked + fp8_block experts; got {config.expert_quant}"
        )
    group = config.linear_attention_group()
    gdn = (group.num_key_heads, group.num_value_heads, group.key_head_dim, group.value_head_dim) if group else None
    shard = _TPShard(tp.rank, tp.size, config.num_kv_heads, gdn)
    stacked = include_moe_experts and config.is_moe and config.expert_quant == "none"
    if include_non_moe or stacked:
        reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0])) if include_non_moe else None
        yield from _iter_shards(model_path, device, reader, stacked=stacked, include_vision=include_vision, shard=shard)


def _iter_shards(model_path: str, device: torch.device, reader: _DenseReader | None, *, stacked: bool, include_vision: bool, shard: _TPShard):
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
                        yield name, shard.expert(name, f.get_tensor(raw_name))
                    continue
                if reader is None:
                    continue
                # Shard the raw part to this rank before the reader merges fused projections;
                # the reader's per-role torch.cat of already-sharded parts yields the local fusion.
                tensor = shard(name, f.get_tensor(raw_name))
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
    ``_scale`` (block scale) companions, named as the checkpoint's dialect stores them. Other expert kinds use the generic readers.

    Yields WHOLE per-expert pieces (full intermediate); ``build_expert_banks`` shards them per
    rank into the TP-local banks."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
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


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "iter_expert_pieces",
    "nvfp4_expert_spec",
]
