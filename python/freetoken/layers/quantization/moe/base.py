"""The MoE side: per-layer config, bank descriptions, the kernel ABC and the Method base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ..method import QuantMethod
from ..scheme import QuantScheme


@dataclass(frozen=True)
class MoEConfig:
    """Everything an expert kernel may look at to accept or decline a layer and to size its banks."""

    num_experts: int
    hidden: int
    intermediate: int
    top_k: int
    tp_rank: int = 0
    tp_size: int = 1
    scheme: QuantScheme | None = None
    activation: str = "silu"
    alpha: float = 1.0
    beta: float = 0.0
    limit: float | None = None
    interleaved: bool = False
    has_bias: bool = False
    apply_router_weight_on_input: bool = False
    strategy: str = "resident"
    decode_target: str = "gpu"

    @classmethod
    def from_layer(cls, layer: Any, scheme: QuantScheme | None) -> "MoEConfig":
        return cls(
            num_experts=layer.num_experts,
            hidden=layer.hidden_size,
            intermediate=layer.intermediate_size,
            top_k=layer.top_k,
            tp_rank=layer.tp_rank,
            tp_size=layer.tp_size,
            scheme=scheme,
            activation=layer.activation,
            alpha=float(layer.alpha),
            beta=float(layer.beta),
            limit=layer.limit,
            interleaved=bool(layer.interleaved),
            has_bias=bool(layer.has_bias),
            apply_router_weight_on_input=bool(layer.apply_router_weight_on_input),
            strategy=layer.strategy,
            decode_target=layer.decode_target,
        )

    @property
    def local_intermediate(self) -> int:
        return self.intermediate // self.tp_size

    @property
    def plain_silu(self) -> bool:
        return (self.activation, self.alpha, self.beta, self.limit) == ("silu", 1.0, 0.0, None)


@dataclass(frozen=True)
class BankSpec:
    """One bank of an expert layout, per expert (the E dim is prepended by the allocator)."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    # GPU-resident per-expert vector (marlin / b12x alphas) instead of a host bank
    resident: bool = False


@dataclass
class ExpertView:
    """The weights an apply() call reads: layout roles -> [E, ...] (resident, materialized layer) or [S, ...] (slot cache)."""

    tensors: dict[str, torch.Tensor]
    slots: torch.Tensor | None = None
    n: int | None = None
    alphas: tuple[torch.Tensor, torch.Tensor] | None = None


def fused_piece(pieces: dict[str, torch.Tensor], role: str) -> torch.Tensor:
    """A gate_up-family piece, concatenating separate gate / up pieces along the row axis."""
    if role in pieces:
        return pieces[role]
    suffix = role[len("gate_up"):]
    return torch.cat([pieces["gate" + suffix], pieces["up" + suffix]], dim=1)


def global_rows(piece: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-expert global scale as one fp16 value per output row (a scalar per expert is broadcast)."""
    flat = piece.reshape(piece.shape[0], -1).to(torch.float16)
    if flat.shape[1] == rows:
        return flat
    assert flat.shape[1] == 1, (flat.shape, rows)
    return flat.expand(-1, rows)


def fused_global(pieces: dict[str, torch.Tensor], half_rows: int) -> torch.Tensor:
    if "gate_up_global" in pieces:
        return global_rows(pieces["gate_up_global"], 2 * half_rows)
    return torch.cat([global_rows(pieces["gate_global"], half_rows), global_rows(pieces["up_global"], half_rows)], dim=1)


def limit_or_inf(layer: Any) -> float:
    limit = getattr(layer, "limit", None)
    return float("inf") if limit is None else float(limit)


def gated_epilogue_reason(cfg: "MoEConfig") -> str | None:
    """Why ``gated_act_and_mul`` cannot run ``cfg``'s activation quadruple, or None; the plain kinds have no alpha / limit and nothing takes beta or interleaved rows."""
    from freetoken.layers import GATED_ACTIVATIONS

    if cfg.activation not in GATED_ACTIVATIONS:
        return f"no {cfg.activation!r} epilogue"
    if cfg.interleaved:
        return "the epilogue reads uninterleaved [gate; up] halves"
    if cfg.beta != 0.0:
        return f"the epilogue has no beta (got {cfg.beta})"
    if cfg.activation in ("silu", "gelu", "gelu_tanh") and (cfg.alpha != 1.0 or cfg.limit is not None):
        return f"{cfg.activation} takes no alpha / limit (got alpha={cfg.alpha}, limit={cfg.limit})"
    return None


def is_resident(layer: Any) -> bool:
    return layer.quant_method.cfg.strategy == "resident"


class MoEKernel(ABC):
    """One backend for one expert kind: bank layout, packing from checkpoint pieces, and the fused forward."""

    name: ClassVar[str]
    cpu_format: ClassVar[str | None] = None
    max_slots: ClassVar[int | None] = None

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        return None

    def worth_it(self, cfg: MoEConfig) -> bool:
        return True

    def slot_limit(self, cfg: MoEConfig) -> int | None:
        """Most GPU cache slots the kernel can address for ``cfg``; None for no limit."""
        return self.max_slots

    def _common_reject(self, cfg: MoEConfig, *, resident_ok: bool, tp_ok: bool, cpu_ok: bool, plain_silu_only: bool) -> str | None:
        if not resident_ok and cfg.strategy == "resident":
            return "not served resident; use --moe-strategy offload or cpu"
        if not tp_ok and cfg.tp_size > 1:
            return "TP > 1 is not supported for this expert format"
        if not cpu_ok and cfg.decode_target != "gpu":
            return "has no CPU executor format; decode must run on the GPU"
        if cfg.has_bias:
            return "kernel has no bias epilogue"
        if plain_silu_only and not cfg.plain_silu:
            return f"kernel is plain-silu only, experts use ({cfg.activation}, alpha={cfg.alpha}, beta={cfg.beta}, limit={cfg.limit})"
        return None

    @abstractmethod
    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]: ...

    @abstractmethod
    def pack(self, pieces: dict[str, torch.Tensor], cfg: MoEConfig, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Write one batch of expert pieces into ``out`` rows; returns the GPU-resident per-expert values, if any."""

    @abstractmethod
    def apply(self, layer: Any, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, view: ExpertView, *, is_prefill: bool) -> torch.Tensor: ...


class MoEMethod(QuantMethod):
    """Declares resident experts for its kind; layout, pack and apply go to the kernel."""

    def layout(self) -> dict[str, BankSpec]:
        return self.kernel.layout(self.cfg)

    def pack(self, pieces: dict[str, torch.Tensor], out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.kernel.pack(pieces, self.cfg, out)

    def slot_limit(self) -> int | None:
        return self.kernel.slot_limit(self.cfg)

    def apply(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, view: ExpertView, *, layer: Any, is_prefill: bool) -> torch.Tensor:
        return self.kernel.apply(layer, x, topk_weights, topk_ids, view, is_prefill=is_prefill)

    @property
    def cpu_format(self) -> str | None:
        return self.kernel.cpu_format

    @abstractmethod
    def create_weights(self, layer: Any) -> None: ...

    def finalize(self, layer: Any) -> None:
        pass

    @abstractmethod
    def resident_view(self, layer: Any) -> ExpertView:
        """Layout roles -> the resident layer's tensors, for apply()."""
