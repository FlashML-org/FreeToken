from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearRowParallel,
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch

    from freetoken.core import Batch
    from freetoken.layers.quantization import QuantConfig

    from .config import ModelConfig


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one forward dispatch: enter before it is enqueued, exit right after. A backend that feeds the forward from host memory overrides this."""
        yield


@runtime_checkable
class SupportsMultimodal(Protocol):
    """What the engine asks of a model that serves multimodal items; the text forward takes its token embeddings from embed_input_ids."""

    def encode(self, item: MMItem) -> torch.Tensor:
        """``[item.num_tokens, D]`` soft tokens for one item of any modality the family serves, on the model device."""
        ...

    def place_encoder_weights(self, mode: str) -> None:
        """Every encoder tower's weights: ``gpu`` resident or ``host`` streamed from pinned banks."""
        ...


def embed_input_ids(embed_tokens, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Token embeddings of the batch; on a chunk with multimodal rows, batch.mm_embeds' leading columns replace the rows batch.mm_rows."""
    if batch.mm_embeds is None:
        return embed_tokens.forward(input_ids)
    # multimodal rows carry content pad ids above the vocab: clamp for the lookup, the copy overwrites those rows
    x = embed_tokens.forward(input_ids.clamp(max=embed_tokens.num_embeddings - 1))
    x.index_copy_(0, batch.mm_rows, batch.mm_embeds[:, : x.shape[1]].to(x.dtype))
    return x


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig, *, quant_config: QuantConfig | None = None, prefix: str = ""):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        fn_map = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
        act_fn = fn_map.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


__all__ = ["BaseLLMModel", "GatedMLP"]
