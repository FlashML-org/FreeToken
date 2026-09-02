"""Triton moe_align_block_size vs a pure-torch reference, on the real GPU.

Regression for the AMD-backend miscompile (Triton 3.8 / gfx1201) where a predicate
derived from a ``tl.histogram`` result is compiled away: ``expert_ids`` stayed
uninitialized (garbage bank rows -> HSA aperture violations in the grouped MoE
GEMM) and, for numel > the fused-kernel cap, the per-program histogram atomics
were dropped (``num_tokens_post_padded == 0`` -> the GEMM computed nothing and the
serve emitted garbage for prompts >= 128 tokens).

Checks exactly what the grouped GEMM relies on: every real id in block ``b`` routes
to ``expert_ids[b]``, each expert's region holds exactly its ids, padding is the
sentinel, and the padded total matches.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.fused import moe_align_block_size

E, TOP_K = 256, 6


def _ref(topk_ids: torch.Tensor, block_size: int) -> tuple[list[int], int]:
    flat = topk_ids.reshape(-1).cpu().tolist()
    expert_ids, total = [], 0
    for e in range(E):
        n = sum(1 for x in flat if x == e)
        if n:
            nblk = -(-n // block_size)
            expert_ids += [e] * nblk
            total += nblk * block_size
    return expert_ids, total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("T", [1, 43, 127, 128, 140, 256, 1024, 2048])  # spans the fused (<=1024 routes) and multi-kernel paths
@pytest.mark.parametrize("block_size", [16, 64])
def test_moe_align_matches_reference(T, block_size):
    g = torch.Generator(device="cpu").manual_seed(T)
    slots = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(T)])
    slots = slots.to("cuda", torch.int32).contiguous()
    numel = T * TOP_K

    sorted_ids, expert_ids, ntpp = moe_align_block_size(slots, block_size, E)
    torch.cuda.synchronize()
    n = int(ntpp.item())
    ref_eids, ref_total = _ref(slots, block_size)
    assert n == ref_total, (n, ref_total)

    eids = expert_ids[: n // block_size].cpu().tolist()
    assert eids == ref_eids

    sids = sorted_ids[:n].cpu()
    flat = slots.reshape(-1).cpu()
    real = sids[sids < numel]
    # every real route appears exactly once; padding is the sentinel (== numel)
    assert torch.equal(torch.sort(real).values, torch.arange(numel, dtype=sids.dtype))
    assert bool((sids[sids >= numel] == numel).all())
    # every real id in block b routes to expert_ids[b]
    blocks = sids.view(-1, block_size)
    owner = torch.tensor(eids, dtype=torch.int64).unsqueeze(1).expand_as(blocks)
    valid = blocks < numel
    assert torch.equal(flat[blocks[valid].long()], owner[valid].to(flat.dtype))
