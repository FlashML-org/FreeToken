"""Expert pieces: the checkpoint-side stream ``build_expert_banks`` packs into host banks.

A piece is ``(bank_layer, e0, e1, {role: tensor[e1 - e0, ...]})``: rows ``e0:e1`` of one MoE
layer's experts, in the checkpoint's own form (per-projection ``gate`` / ``up`` / ``down`` with
their ``_scale`` / ``_global`` companions, or an already fused ``gate_up``). The expert kernel's
``pack`` turns pieces into its bank layout, so readers never know the layout.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

import torch
from freetoken.layers.quantization import QuantKind
from freetoken.models.register import _load_attr, get_model_spec

Piece = tuple[int, int, int, dict[str, torch.Tensor]]


def num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0) or 0)


def bank_layer_of(config, layer: int) -> int | None:
    """MoE-layer index of checkpoint layer ``layer``; None for the leading dense layers."""
    bank_layer = layer - int(getattr(config, "first_k_dense_replace", 0) or 0)
    return bank_layer if 0 <= bank_layer < num_moe_layers(config) else None


def _model_hook(spec, name: str):
    try:
        return _load_attr(spec.module, name)
    except AttributeError:
        return None


def iter_expert_pieces(
    model_path: str, config, kind: QuantKind, *, parallel: bool = False, workers: int = 8, chunk: int = 8 << 20
) -> Iterator[Piece]:
    """The pieces of ``model_path``'s routed experts, stored as ``kind``.

    A family may ship its own ``iter_expert_pieces(model_path, config, kind, *, parallel, workers,
    chunk)`` for the storage forms only it uses (it returns None for the rest); otherwise bf16
    experts come from the family's stacked ``iter_weights`` and NVFP4 experts from its
    ``nvfp4_expert_spec``. The reader is resolved here, before any bank is allocated, so a
    missing parallel reader raises ``NotImplementedError`` while a serial fallback is still cheap.
    """
    spec = get_model_spec(config.architectures[0])
    hook = _model_hook(spec, "iter_expert_pieces")
    if hook is not None:
        pieces = hook(model_path, config, kind, parallel=parallel, workers=workers, chunk=chunk)
        if pieces is not None:
            return pieces
    if kind is QuantKind.NONE:
        return _bf16_pieces(model_path, config, spec, parallel=parallel, workers=workers, chunk=chunk)
    if kind is QuantKind.NVFP4:
        spec_hook = _model_hook(spec, "nvfp4_expert_spec")
        if spec_hook is None:
            raise NotImplementedError(f"{spec.module} provides no nvfp4_expert_spec")
        from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces

        return iter_nvfp4_expert_pieces(
            model_path, config, spec_hook(model_path, config), parallel=parallel, workers=workers, chunk=chunk
        )
    raise NotImplementedError(f"{spec.module} provides no expert reader for {kind!r} experts")


def packed_expert_source_info(key: str) -> tuple[int, str] | None:
    """``model.layers.N....experts.{gate_up_proj,down_proj}`` -> (N, leaf); None otherwise."""
    parts = key.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    if parts[-2] != "experts" or parts[-1] not in {"gate_up_proj", "down_proj"}:
        return None
    try:
        return int(parts[2]), parts[-1]
    except ValueError:
        return None


def stacked_expert_pieces(tensors: Iterable[tuple[str, torch.Tensor]], config) -> Iterator[Piece]:
    """Whole-layer ``experts.gate_up_proj`` / ``experts.down_proj`` tensors -> one piece per layer."""
    pending: dict[int, dict[str, torch.Tensor]] = {}
    num_experts = config.num_experts
    for name, tensor in tensors:
        info = packed_expert_source_info(name)
        if info is None:
            raise ValueError(f"Unexpected expert weight key: {name}")
        layer, packed_name = info
        bank_layer = bank_layer_of(config, layer)
        if bank_layer is None:
            raise ValueError(f"Unexpected MoE expert layer {layer}; expected a routed-expert layer")
        if tensor.size(0) != num_experts:
            raise ValueError(f"Unexpected {packed_name} expert count {tensor.size(0)}; expected {num_experts}")
        piece = pending.setdefault(bank_layer, {})
        piece["gate_up" if packed_name == "gate_up_proj" else "down"] = tensor
        if len(piece) == 2:
            del pending[bank_layer]
            yield bank_layer, 0, num_experts, piece
    if pending:
        raise ValueError(f"Missing MoE expert source layers: {sorted(pending)}")


def _bf16_pieces(model_path: str, config, spec, *, parallel: bool, workers: int, chunk: int) -> Iterator[Piece]:
    device = torch.device("cpu")
    if parallel:
        iter_weights = _model_hook(spec, "iter_weights_parallel")
        if iter_weights is None:
            raise NotImplementedError(f"{spec.module} provides no iter_weights_parallel")
        tensors = iter_weights(
            model_path, device, include_moe_experts=True, include_non_moe=False, workers=workers, chunk=chunk
        )
    else:
        iter_weights = _load_attr(spec.module, spec.iter_weights)
        tensors = iter_weights(model_path, device, include_moe_experts=True, include_non_moe=False)
    return stacked_expert_pieces(tensors, config)


def per_expert_pieces(
    tensors: Iterable[tuple[str, torch.Tensor]],
    locate: Callable[[str], tuple[int, int, str] | None],
    *,
    tensors_per_expert: int,
) -> Iterator[Piece]:
    """Group per-expert tensors into one piece per expert, in whatever order they arrive.

    ``locate(name)`` -> ``(bank_layer, expert, role)`` or None to skip; a role tensor is one
    expert's ``[...]`` row and becomes the piece's ``[1, ...]`` entry. Globals may be scalars.
    """
    pending: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for name, tensor in tensors:
        where = locate(name)
        if where is None:
            continue
        bank_layer, expert, role = where
        piece = pending.setdefault((bank_layer, expert), {})
        piece[role] = tensor.reshape(1, -1) if role.endswith("_global") else tensor.unsqueeze(0)
        if len(piece) == tensors_per_expert:
            del pending[(bank_layer, expert)]
            yield bank_layer, expert, expert + 1, piece
    if pending:
        missing = sorted(pending)[:8]
        raise ValueError(f"Incomplete expert tensors in checkpoint (layer, expert): {missing}")


__all__ = [
    "Piece",
    "bank_layer_of",
    "iter_expert_pieces",
    "num_moe_layers",
    "packed_expert_source_info",
    "per_expert_pieces",
    "stacked_expert_pieces",
]
