from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from tqdm import tqdm

from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.loader import iter_weight_files
from freetoken.models.register import get_model_spec
from freetoken.utils import cached_load_hf_config

_LINEAR_LEAVES = frozenset({"q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in."""

    def __init__(self, quant: QuantConfig | None, packed: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.quant = quant
        self.by_part = {part: (fused, idx, len(parts)) for fused, parts in packed for idx, part in enumerate(parts)}
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores}, scheme)
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        parent, _, leaf = module.rpartition(".")
        if leaf not in _LINEAR_LEAVES:
            return None
        scheme = None if self.quant is None else self.quant.scheme_for(module)
        if scheme is not None and scheme.kind is not QuantKind.NVFP4:
            raise NotImplementedError(f"{module}: ricursive reads bf16 and NVFP4 checkpoints, the quant config declares {scheme}")
        roles = {"weight": "weight"} if scheme is None else {n: r for r, e in self.quant.storage(scheme).items() for n in e.names}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(f"{name}: the checkpoint's quant config declares {module} {scheme or 'unquantized'}, stored as {sorted(roles)}")
        if scheme is None and tensor.dtype in _QUANT_DTYPES:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
        fused, idx, count = self.by_part.get(leaf, (None, 0, 1))
        target = module if fused is None else f"{parent}.{fused}"
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, scheme))
        parts.setdefault(idx, {})[role] = tensor
        expected[idx] = set(roles.values())
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], scheme)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, _scheme) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}")
        return lines

    def _emit(self, target: str, parts: list[dict[str, torch.Tensor]], scheme: QuantScheme | None):
        if scheme is not None:
            parts = [self._check(target, scheme, part) for part in parts]
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
        """Validate one NVFP4 part and put its scales in the layer's form."""
        part = {
            role: 1.0 / tensor.to(torch.float32) if self.quant.storage(scheme)[role].reciprocal else tensor
            for role, tensor in part.items()
        }
        weight, scale = part["weight"], part["weight_scale"]
        if weight.dtype is not torch.uint8:
            raise ValueError(f"{target}: weight is {weight.dtype} but the checkpoint's quant config declares {scheme}")
        rows, cols = weight.shape[0], weight.shape[1] * 2
        block_rows, block_cols = scheme.weight.group
        if rows % block_rows or cols % block_cols:
            raise ValueError(f"{target}: {rows}x{cols} weight is not a multiple of the {block_rows}x{block_cols} scale block of {scheme}")
        if tuple(scale.shape) != (rows // block_rows, cols // block_cols) or scale.dtype is not torch.float8_e4m3fn:
            raise ValueError(f"{target}: weight_scale is {scale.dtype} {tuple(scale.shape)}, expected e4m3 {(rows // block_rows, cols // block_cols)}")
        out = dict(part)
        g = part["weight_global"].reshape(-1).to(torch.float32)
        if g.numel() != 1:
            raise ValueError(f"{target}: weight_global has {g.numel()} elements, expected one per-tensor scale")
        out["weight_global"] = g.expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out


def iter_weights(
    model_path: str, device: torch.device, *, include_moe_experts: bool, include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return
    spec = get_model_spec(cached_load_hf_config(model_path).architectures[0])
    reader = _DenseReader(get_quant_config(), spec.packed_modules_mapping)
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name.endswith(".value_res_lambda"):
                    yield name.replace(".value_res_lambda", ".value_res_blend"), torch.sigmoid(tensor.float()).to(tensor.dtype)
                    continue
                emitted = reader.add(name, tensor)
                if emitted is None:
                    yield name, tensor
                else:
                    yield from emitted
    if reader.pending:
        raise ValueError(f"{model_path}: incomplete projections: {reader.missing()}")
