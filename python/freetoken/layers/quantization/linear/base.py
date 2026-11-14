"""The linear side: per-layer config, the kernel ABC and the Method base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ..method import QuantMethod
from ..scheme import QuantScheme


@dataclass(frozen=True)
class LinearConfig:
    """What a linear kernel may look at to accept or decline a layer: the TP-local shape and the weight scheme."""

    in_features: int
    out_features: int
    # fused segments along N; a plain linear is one segment
    output_sizes: tuple[int, ...] = ()
    scheme: QuantScheme | None = None

    def __post_init__(self) -> None:
        if not self.output_sizes:
            object.__setattr__(self, "output_sizes", (self.out_features,))
        if sum(self.output_sizes) != self.out_features:
            raise ValueError(f"output_sizes {self.output_sizes} do not sum to {self.out_features}")

    @classmethod
    def from_layer(cls, layer: Any, scheme: QuantScheme | None) -> "LinearConfig":
        return cls(layer.in_features, layer.out_features, tuple(layer.output_sizes), scheme)


class LinearKernel(ABC):
    """One backend for one linear kind: says whether it can run a layer, then finalizes and applies it."""

    name: ClassVar[str]

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        return None

    def worth_it(self, cfg: LinearConfig) -> bool:
        return True

    def finalize(self, layer: Any) -> None:
        pass

    def input_format(self, layer: Any, x: torch.Tensor) -> str | None:
        """Name of the quantized copy of ``x`` that ``apply`` takes as ``xq``; None if ``apply`` has no ``xq``."""
        return None

    @abstractmethod
    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor: ...


class LinearMethod(QuantMethod):
    """Declares a linear layer's weights for its kind; finalize and apply go to the kernel."""

    @abstractmethod
    def create_weights(self, layer: Any) -> None: ...

    def finalize(self, layer: Any) -> None:
        self.kernel.finalize(layer)

    def input_format(self, layer: Any, x: torch.Tensor) -> str | None:
        return self.kernel.input_format(layer, x)

    def apply(self, layer: Any, x: torch.Tensor, xq: torch.Tensor | None = None) -> torch.Tensor:
        return self.kernel.apply(layer, x) if xq is None else self.kernel.apply(layer, x, xq=xq)
