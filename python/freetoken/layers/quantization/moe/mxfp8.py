"""MXFP8 experts: registered so the table is complete, not served yet."""

from __future__ import annotations

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import ExpertView, MoEMethod


@register_method(QuantKind.MXFP8, LayerKind.MOE)
class Mxfp8MoEMethod(MoEMethod):
    candidates = ()

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("mxfp8 experts are not served yet")

    def create_weights(self, layer) -> None:
        raise NotImplementedError

    def resident_view(self, layer) -> ExpertView:
        raise NotImplementedError
