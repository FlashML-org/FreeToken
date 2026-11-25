"""DeepSeek-V4's partitions under --tp N (distributed/plan.py).

Attention splits by head: ``wq_b`` and ``attn_sink`` by head, ``wo_a`` and ``wo_b`` by output group, so N divides
``o_groups``. ``wq_a``, the latent ``wkv``, the compressors and the indexer stay whole on every rank: the KV cache holds
one latent per token whatever N is (DCP is what will split it). The shared expert splits column/row like a dense MLP,
the vocab splits for the embedding and the head, and the routed experts split their intermediate dim through the
expert placement, their input already all-reduced by attention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from freetoken.distributed import DistributedInfo, get_tp_info
from freetoken.distributed.plan import (
    BaseParallelPlan,
    ExpertGroup,
    ParallelArgs,
    ParallelPlan,
    TensorPartition,
)

# fp8 block scales: every split edge of an fp8 weight must be a block edge
_FP8_BLOCK = 128
# dense tensors split along rows / columns, named as iter_weights names them; every other one is replicated
_ROWS = re.compile(
    r"model\.(embed|head)\.weight"
    r"|model\.layers\.\d+\.attn\.(attn_sink|wo_a|wq_b\.(weight|weight_scale_inv))"
    r"|model\.layers\.\d+\.ffn\.shared_experts\.w[13]\.(weight|weight_scale_inv)"
)
_COLS = re.compile(r"model\.layers\.\d+\.(attn\.wo_b|ffn\.shared_experts\.w2)\.(weight|weight_scale_inv)")
# a tensor of a split projection that the rules above do not cover (another quant format's scales)
_SPLIT_MODULE = re.compile(r"model\.layers\.\d+\.(attn\.(wq_b|wo_b)|ffn\.shared_experts\.w[123])\.")


@dataclass
class Dsv4ParallelPlan(BaseParallelPlan):
    # attention heads per rank, which every attention layer must be built with
    local_heads: int = field(kw_only=True)

    def __post_init__(self):
        self._rows = TensorPartition(0, self.dense.rank, self.dense.size)
        self._cols = TensorPartition(1, self.dense.rank, self.dense.size)

    def shard(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if _ROWS.fullmatch(name):
            return self._rows.slice(tensor)
        if _COLS.fullmatch(name):
            return self._cols.slice(tensor)
        if _SPLIT_MODULE.match(name):
            raise ValueError(f"DeepSeek-V4 TP splits fp8 block-scaled projections only; {name} is in another format")
        return tensor

    def bind(self, model: Any, cache: Any) -> None:
        from freetoken.layers import iter_ops

        from .attention import Attention

        if get_tp_info() != self.dense:
            raise ValueError(f"the layers split over {get_tp_info()}, the plan over {self.dense}")
        heads = self.local_heads
        wrong = [op.layer_id for op in iter_ops(model) if isinstance(op, Attention) and op.attn_sink.shape[0] != heads]
        if wrong:
            raise ValueError(f"attention layers {wrong} were not built for {heads} heads per rank")
        super().bind(model, cache)


def build_parallel_plan(model_config, args: ParallelArgs) -> ParallelPlan:
    a, tp = model_config.dsv4_args, args.tp_size
    splits = {  # what splits -> its parts per rank must be whole
        f"{a.o_groups} output groups (of {a.n_heads // a.o_groups} heads)": a.o_groups % tp == 0,
        f"a {a.vocab_size}-token vocab": a.vocab_size % tp == 0,
        f"wq_b, wo_b and the {a.moe_inter_dim}-wide shared expert into fp8 blocks of {_FP8_BLOCK}": all(
            (n // tp) % _FP8_BLOCK == 0 and n % tp == 0
            for n in (a.n_heads * a.head_dim, a.o_groups * a.o_lora_rank, a.moe_inter_dim)
        ),
    }
    for what, ok in splits.items():
        if not ok:
            raise ValueError(f"DeepSeek-V4 cannot split {what} {tp} ways; pass a --tp that does")
    world = DistributedInfo(args.rank, tp)
    return Dsv4ParallelPlan(args, world, ExpertGroup(tp, a.moe_inter_dim), local_heads=a.n_heads // tp)
