"""GLM-5.3's partitions under --tp N (distributed/plan.py).

Both attention kinds split by head. The 11 MLA layers split ``q_b_proj`` and ``kv_b_proj`` by head and ``o_proj`` by its
input; ``q_a``, the latent ``kv_a`` and the indexer stay whole, so the KV cache holds one latent per token on every rank.
The 34 KDA layers split q, k, v, beta and both gate up-projections by head, and their recurrent and conv state with
them; the low-rank ``f_a`` and ``g_a`` stay whole. The dense MLPs and the shared expert split column/row, the vocab
splits for the embedding and the head, and the routed experts split their intermediate dim through the expert placement,
their input already all-reduced by attention. mHC, the norms and the router stay whole.

The loader fuses KDA's projections (weight.py): ``in_proj`` is q|k|v|b|f_a|g_a and ``conv1d`` is q|k|v, so each segment
of those is cut on its own.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from freetoken.distributed import DistributedInfo, get_tp_info
from freetoken.distributed.plan import BaseParallelPlan, ExpertGroup, ParallelArgs, ParallelPlan, TensorPartition

# dense tensors split along rows / columns, named as iter_weights names them; every other one is replicated
_ROWS = re.compile(
    r"model\.embed_tokens\.weight|lm_head\.weight"
    r"|model\.layers\.\d+\.self_attn\.((q_b_proj|kv_b_proj|f_b_proj|g_b_proj)\.weight|A_log|dt_bias)"
    r"|model\.layers\.\d+\.mlp\.(shared_experts\.)?(gate_proj|up_proj)\.weight"
)
_COLS = re.compile(r"model\.layers\.\d+\.(self_attn\.o_proj|mlp\.(shared_experts\.)?down_proj)\.weight")
_IN_PROJ = re.compile(r"model\.layers\.\d+\.self_attn\.in_proj\.weight")
_CONV = re.compile(r"model\.layers\.\d+\.self_attn\.conv1d\.weight")
# KDA's fused projections, in the one order the loader concatenates them (weight.py), the op splits them (kda.py) and
# the plan cuts them: (checkpoint name, rows, split by head). Rows are "proj" (heads x head_dim), "heads" (one per head)
# or "head_dim" (a low-rank width that feeds every head, so it stays whole).
KDA_IN_PROJ = (
    ("q_proj", "proj", True), ("k_proj", "proj", True), ("v_proj", "proj", True), ("b_proj", "heads", True),
    ("f_a_proj", "head_dim", False), ("g_a_proj", "head_dim", False),
)
KDA_CONV = (("q_conv1d", "proj", True), ("k_conv1d", "proj", True), ("v_conv1d", "proj", True))
# the conv runs over in_proj's leading q|k|v rows as one stream
assert [n.removesuffix("_proj") for n, _, _ in KDA_IN_PROJ[: len(KDA_CONV)]] == [n.removesuffix("_conv1d") for n, _, _ in KDA_CONV]


def kda_segments(table, num_heads: int, head_dim: int) -> tuple[tuple[int, bool], ...]:
    """(rows, split by head) of each segment of a fused KDA projection over ``num_heads`` heads."""
    rows = {"proj": num_heads * head_dim, "heads": num_heads, "head_dim": head_dim}
    return tuple((rows[kind], split) for _, kind, split in table)


# a tensor of a split projection that the rules above do not cover (another format's scales)
_SPLIT_MODULE = re.compile(
    r"model\.layers\.\d+\.(self_attn\.(q_b_proj|kv_b_proj|f_b_proj|g_b_proj|o_proj|in_proj)"
    r"|mlp\.(shared_experts\.)?(gate_proj|up_proj|down_proj))\."
)


def dense_tp(config) -> int:
    """How many ways the parallel plan splits GLM-5.3's attention and dense layers; 1 without a plan."""
    plan = getattr(config, "parallel_plan", None)  # a stub config in a test may carry none
    return plan.dense.size if plan is not None else 1


def routed_expert_group(config) -> ExpertGroup | None:
    """The ranks the parallel plan splits GLM-5.3's routed experts over; None without a plan."""
    plan = getattr(config, "parallel_plan", None)
    return plan.experts if plan is not None else None


@dataclass
class Glm5NextParallelPlan(BaseParallelPlan):
    # (rows, split) per segment of KDA's fused in_proj and conv1d, in the loader's fusion order
    in_proj_segments: tuple[tuple[int, bool], ...] = field(kw_only=True)
    conv_segments: tuple[tuple[int, bool], ...] = field(kw_only=True)

    def __post_init__(self):
        self._rows = TensorPartition(0, self.dense.rank, self.dense.size)
        self._cols = TensorPartition(1, self.dense.rank, self.dense.size)

    def validate_checkpoint(self, model_path: str) -> None:
        super().validate_checkpoint(model_path)
        from freetoken.utils import download_hf_weight

        index = os.path.join(download_hf_weight(model_path), "model.safetensors.index.json")
        if not os.path.exists(index):
            return
        with open(index) as f:
            names = json.load(f)["weight_map"]
        # the zai-org release stores the dense projections as fp8 with 128x128 block scales, whose blocks the head
        # splits (KDA's beta: 64 rows) do not align with
        fp8 = next((n for n in names if n.endswith(".weight_scale_inv") and ".mlp.experts." not in n), None)
        if fp8 is not None:
            raise ValueError(f"GLM-5.3 tensor parallelism splits bf16 dense weights; {model_path} stores {fp8}, serve it with --tp 1")

    def shard(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if _ROWS.fullmatch(name):
            return self._rows.slice(tensor)
        if _COLS.fullmatch(name):
            return self._cols.slice(tensor)
        if _IN_PROJ.fullmatch(name):
            return self._segments(name, tensor, self.in_proj_segments)
        if _CONV.fullmatch(name):
            return self._segments(name, tensor, self.conv_segments)
        if _SPLIT_MODULE.match(name):
            raise ValueError(f"GLM-5.3 TP splits bf16 projections only; {name} is in another format")
        return tensor

    def _segments(self, name: str, tensor: torch.Tensor, segments: tuple[tuple[int, bool], ...]) -> torch.Tensor:
        if sum(rows for rows, _ in segments) != tensor.shape[0]:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, the plan's segments {segments} sum to another count")
        parts = tensor.split([rows for rows, _ in segments])
        return torch.cat([self._rows.slice(part) if split else part for part, (_, split) in zip(parts, segments)])

    def bind(self, model: Any, cache: Any) -> None:
        if get_tp_info() != self.dense:
            raise ValueError(f"the layers split over {get_tp_info()}, the plan over {self.dense}")
        super().bind(model, cache)


def build_parallel_plan(model_config, args: ParallelArgs) -> ParallelPlan:
    a, tp = model_config.glm5_args, args.tp_size
    shared = model_config.moe_intermediate_size * max(1, model_config.n_shared_experts)
    splits = {  # what splits -> its parts per rank must be whole
        f"{a.num_heads} MLA heads": a.num_heads % tp == 0,
        f"{a.linear_num_heads} KDA heads": a.linear_num_heads % tp == 0,
        f"the {model_config.intermediate_size}-wide dense MLPs": model_config.intermediate_size % tp == 0,
        f"the {shared}-wide shared expert": shared % tp == 0,
        f"a {model_config.vocab_size}-token vocab": model_config.vocab_size % tp == 0,
    }
    for what, ok in splits.items():
        if not ok:
            raise ValueError(f"GLM-5.3 cannot split {what} {tp} ways; pass a --tp that does")
    return Glm5NextParallelPlan(
        args,
        DistributedInfo(args.rank, tp),
        ExpertGroup(tp, model_config.moe_intermediate_size),
        in_proj_segments=kda_segments(KDA_IN_PROJ, a.linear_num_heads, a.linear_head_dim),
        conv_segments=kda_segments(KDA_CONV, a.linear_num_heads, a.linear_head_dim),
    )


def build_gguf_parallel_plan(model_config, args: ParallelArgs) -> ParallelPlan:
    raise ValueError("GLM-5.3 GGUF files serve --tp 1 only: their ggml experts cannot be sliced across GPUs")
