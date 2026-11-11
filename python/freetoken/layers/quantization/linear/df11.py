"""DF11 Linear: the bf16 weight compressed losslessly at load, decoded bit-for-bit into a shared scratch before each GEMM."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod
from .unquantized import TorchLinearKernel

DF11_KEYS = ("low8", "bitstream", "chunk_start", "lut")

# decodes run one layer at a time, so every DF11 linear on a device shares one scratch sized to the largest weight
_SCRATCH: dict[torch.device, torch.Tensor] = {}
_GEMM = TorchLinearKernel()


def _scratch(n: int, device: torch.device) -> torch.Tensor:
    buf = _SCRATCH.get(device)
    if buf is None or buf.numel() < n:
        buf = torch.empty(n, dtype=torch.int16, device=device)
        _SCRATCH[device] = buf
    return buf[:n]


def df11_install(layer: Any, compressed: dict) -> None:
    """Replace ``layer.weight`` with the DF11 tensors of ``compressed``."""
    del layer.weight
    for key in DF11_KEYS:
        setattr(layer, key, compressed[key])
    layer.df11_meta = compressed["meta"]


def df11_bundle(layer: Any) -> dict:
    return {**{key: getattr(layer, key) for key in DF11_KEYS}, "meta": layer.df11_meta}


class Df11LinearKernel(LinearKernel):
    name = "triton"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        # the model is built under its dtype, which the loader casts the checkpoint weight to
        dtype = torch.get_default_dtype()
        return None if dtype == torch.bfloat16 else f"DF11 compresses bf16 weights, the model runs {dtype}"

    def finalize(self, layer: Any) -> None:
        from freetoken.kernel.triton.df11 import df11_compress

        df11_install(layer, df11_compress(layer.weight.cuda()))
        # sized here, not in the first forward, so a CUDA graph capture never grows it
        _scratch(layer.out_features * layer.in_features, layer.low8.device)

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.df11_decode import df11_decompress

        out = _scratch(layer.out_features * layer.in_features, x.device).view(layer.out_features, layer.in_features)
        return _GEMM.apply(SimpleNamespace(weight=df11_decompress(df11_bundle(layer), out=out), bias=layer.bias), x)


@register_method(QuantKind.DF11, LayerKind.LINEAR)
class Df11LinearMethod(LinearMethod):
    candidates = (Df11LinearKernel,)
    online = True

    @classmethod
    def refuses(cls, layer: Any) -> str | None:
        from freetoken.layers.embedding import ParallelLMHead

        if isinstance(layer, ParallelLMHead):
            return "the LM head would decode its whole weight every step"
        return None

    def create_weights(self, layer: Any) -> None:
        layer.weight = torch.empty(self.cfg.out_features, self.cfg.in_features)
