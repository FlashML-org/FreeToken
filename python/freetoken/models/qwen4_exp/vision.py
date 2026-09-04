"""Qwen3.8-Flash-Next vision tower: HF ``Qwen4ExpVisionModel`` behind the BaseOP state-dict contract.

The tower is bf16, ~0.9 GiB, never quantized and identical on every TP rank (each rank encodes
its own copy of the request's images; the soft tokens are tiny next to the tower). Its tensors
travel under the ``visual.*`` keys of the model state dict so the engine loads them with the
dense weights: counted in the weight bytes before the expert-cache planner sizes the cache,
dummy-weight builds fill them like any other tensor, and TP ranks all see the same keys.

Meta-device trap: ``create_model`` runs under ``torch.device("meta")``. The HF module built
there is shapes only; ``load_state_dict`` rebuilds a meta module, assigns the loaded tensors
(no second copy on the GPU) and re-creates the non-persistent rotary ``inv_freq`` buffer on the
device, which ``assign=True`` would otherwise leave on meta.
"""

from __future__ import annotations

from typing import Any

import torch
from freetoken.layers import BaseOP
from freetoken.layers.base import _concat_prefix


def _vision_model_cls():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    return Qwen4ExpVisionModel


class Qwen4ExpVisionTower(BaseOP):
    def __init__(self, vision_config: Any) -> None:
        # The standalone vision config carries no attention backend; sdpa splits packed images
        # by cu_seqlens, so several images per call are fine.
        vision_config._attn_implementation = "sdpa"
        self._config = vision_config
        self._hf = _vision_model_cls()(vision_config)
        self.spatial_merge_size = int(vision_config.spatial_merge_size)

    def state_dict(self, *, prefix: str = "", result=None):
        result = result if result is not None else {}
        for name, tensor in self._hf.state_dict().items():
            result[_concat_prefix(prefix, name)] = tensor
        return result

    def load_state_dict(
        self, state_dict, *, prefix: str = "", _internal: bool = False
    ) -> None:
        own = {
            k: state_dict.pop(_concat_prefix(prefix, k)) for k in self._hf.state_dict()
        }
        device = next(iter(own.values())).device
        with torch.device("meta"):
            hf = _vision_model_cls()(self._config)
        hf.load_state_dict(own, strict=True, assign=True)
        head_dim = self._config.hidden_size // self._config.num_heads
        with torch.device(device):
            hf.rotary_pos_emb = type(hf.rotary_pos_emb)(head_dim // 2)
        stuck = [n for n, b in hf.named_buffers() if b.is_meta]
        assert not stuck, f"vision buffers left on meta: {stuck}"
        self._hf = hf.eval().requires_grad_(False)
        if not _internal and state_dict:
            raise RuntimeError(
                f"Unexpected keys in state_dict: {list(state_dict.keys())}"
            )

    @torch.inference_mode()
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """``[sum(t*h*w) / merge**2, text_hidden]`` soft tokens for the packed images."""
        param = next(self._hf.parameters())
        out = self._hf(
            pixel_values.to(param.device, param.dtype),
            grid_thw=grid_thw.to(param.device),
            return_dict=True,
        )
        return out.pooler_output


__all__ = ["Qwen4ExpVisionTower"]
