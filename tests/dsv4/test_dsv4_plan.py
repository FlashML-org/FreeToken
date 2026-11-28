"""DeepSeek-V4's --tp plan against the attention a rank builds: the tensors it splits by head or output group keep one
rank's heads together, in the shapes that rank's layers are built with."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.distributed.plan import ParallelArgs
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.attention import Attention
from freetoken.models.deepseek_v4.multi import build_parallel_plan

# the smallest shapes whose split edges stay on fp8 block edges at --tp 4
ARGS = DeepseekV4Args(
    dim=256, n_layers=1, n_heads=8, head_dim=64, rope_head_dim=32, q_lora_rank=64, o_groups=4, o_lora_rank=128,
    moe_inter_dim=512, vocab_size=64, compress_ratios=(0,), n_routed_experts=4, n_activated_experts=2,
)


@pytest.mark.parametrize("tp", [2, 4])
def test_head_split_tensors_keep_each_ranks_heads_together(tp):
    a = ARGS
    heads_per_group = a.n_heads // a.o_groups
    # every row (or wo_b column) holds the head or output group it belongs to
    full = {
        "wq_b.weight": torch.arange(a.n_heads).repeat_interleave(a.head_dim)[:, None].expand(-1, a.q_lora_rank).float(),
        "attn_sink": torch.arange(a.n_heads).float(),
        "wo_a": torch.arange(a.o_groups).repeat_interleave(a.o_lora_rank)[:, None].expand(-1, a.n_heads * a.head_dim // a.o_groups).float(),
        "wo_b.weight": torch.arange(a.o_groups).repeat_interleave(a.o_lora_rank)[None, :].expand(a.dim, -1).float(),
    }
    seen = set()
    for rank in range(tp):
        plan = build_parallel_plan(SimpleNamespace(dsv4_args=a), ParallelArgs(rank, tp))
        attn = Attention(0, a, tp=plan.dense.size)
        cut = {name: plan.shard(f"model.layers.0.attn.{name}", t) for name, t in full.items()}

        assert cut["wq_b.weight"].shape == attn.wq_b.weight.shape
        assert cut["attn_sink"].shape == attn.attn_sink.shape
        assert cut["wo_a"].shape == attn.wo_a.shape
        assert cut["wo_b.weight"].shape == attn.wo_b.weight.shape

        heads = set(cut["wq_b.weight"][:, 0].int().tolist())
        groups = set(cut["wo_a"][:, 0].int().tolist())
        assert set(cut["attn_sink"].int().tolist()) == heads
        assert {h for g in groups for h in range(g * heads_per_group, (g + 1) * heads_per_group)} == heads
        assert set(cut["wo_b.weight"][0].int().tolist()) == groups
        assert len(heads) == plan.local_heads == attn.n_local_heads
        seen |= heads
    assert seen == set(range(a.n_heads))
