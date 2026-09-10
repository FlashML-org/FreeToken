from __future__ import annotations

from typing import List

import torch
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_even

from .base import BaseOP
from .quantization import LayerKind, QuantConfig, quant_method_for


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism.

    The weights are declared and applied by ``quant_method``, picked from ``quant_config`` by
    the layer's ``prefix``; without a config the layer is plain bf16."""

    quant_layer_kind = LayerKind.LINEAR

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        *,
        output_sizes: List[int] | None = None,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.has_bias = has_bias
        self.prefix = prefix
        # the TP-local shape the quant method declares weights for; fused projections list their segments
        self.in_features = local_isize
        self.out_features = local_osize
        self.output_sizes = tuple(output_sizes or (local_osize,))
        self.quant_method = quant_method_for(quant_config, self, prefix)
        self.quant_method.create_weights(self)
        self.bias = torch.empty(local_osize) if has_bias else None

    def finalize(self) -> None:
        self.quant_method.finalize(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
            quant_config=quant_config,
            prefix=prefix,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        local_output_sizes: List[int] | None = None,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        # check that all output sizes are divisible by tp_size (a caller that replicates
        # GQA kv heads across ranks passes the per-rank sizes explicitly)
        tp_info = get_tp_info()
        if local_output_sizes is None:
            local_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        tp_output_sizes = local_output_sizes
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(
            input_size, output_size, input_size, tp_output_size, has_bias,
            output_sizes=tp_output_sizes, quant_config=quant_config, prefix=prefix,
        )


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(
            full_isize, full_osize, local_isize, local_osize, has_bias,
            output_sizes=[local_num_qo * head_dim, local_num_kv * head_dim, local_num_kv * head_dim],
            quant_config=quant_config, prefix=prefix,
        )


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            full_isize, full_osize, local_isize, local_osize, has_bias,
            quant_config=quant_config, prefix=prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.quant_method.apply(self, x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size, output_size, local_input_size, local_output_size, has_bias,
            quant_config=quant_config, prefix=prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.quant_method.apply(self, x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
