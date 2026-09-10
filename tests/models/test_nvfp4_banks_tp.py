"""TP sharding of the NVFP4 routed-expert stream.

Every rank reads the same checkpoint tensors and keeps only its own slice of the expert
intermediate axis, so the ranks' slices must partition the original exactly -- the MoE layer
all-reduces the partial sums, which is only correct if nothing is dropped or double-counted.
The expert kernel sizes its banks from ``MoEConfig.local_intermediate``, the same split.
"""

import pytest
import torch
from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.models.nvfp4_banks import _tp_shard

INTER, HIDDEN = 640, 2560


def _roles():
    """One expert's checkpoint tensors, in the modelopt layout."""
    return {
        "gate": torch.arange(INTER * (HIDDEN // 2), dtype=torch.uint8).reshape(INTER, HIDDEN // 2),
        "gate_scale": torch.arange(INTER * (HIDDEN // 16), dtype=torch.uint8).reshape(INTER, HIDDEN // 16),
        "gate_global": torch.tensor([2.5]),
        "up": torch.arange(INTER * (HIDDEN // 2), dtype=torch.uint8).reshape(INTER, HIDDEN // 2),
        "up_scale": torch.arange(INTER * (HIDDEN // 16), dtype=torch.uint8).reshape(INTER, HIDDEN // 16),
        "up_global": torch.tensor([2.5]),
        "down": torch.arange(HIDDEN * (INTER // 2), dtype=torch.uint8).reshape(HIDDEN, INTER // 2),
        "down_scale": torch.arange(HIDDEN * (INTER // 16), dtype=torch.uint8).reshape(HIDDEN, INTER // 16),
        "down_global": torch.tensor([3.5]),
    }


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_rank_slices_partition_the_original_tensor(tp_size):
    full = _roles()
    for role, tensor in full.items():
        parts = [_tp_shard(role, tensor, INTER, tp_size, r) for r in range(tp_size)]
        if role.endswith("_global"):
            # per-tensor scalar: no I axis, every rank keeps it whole
            assert all(torch.equal(p, tensor) for p in parts)
            continue
        axis = 1 if role.startswith("down") else 0
        assert torch.equal(torch.cat(parts, dim=axis), tensor), role
        assert all(p.shape[axis] == tensor.shape[axis] // tp_size for p in parts), role


def test_tp1_is_the_identity():
    for role, tensor in _roles().items():
        assert _tp_shard(role, tensor, INTER, 1, 0) is tensor or torch.equal(
            _tp_shard(role, tensor, INTER, 1, 0), tensor
        )


def test_a_shard_that_splits_a_scale_block_is_rejected():
    # scales cover 16 values; a rank slice that is not a whole number of blocks cannot be
    # sliced consistently across the weight and its scale
    with pytest.raises(AssertionError, match="16-wide scale blocks"):
        _tp_shard("gate", torch.zeros(24, 8, dtype=torch.uint8), 24, 3, 0)


def test_moe_config_local_intermediate_matches_the_shard():
    for tp_size in (1, 2, 4):
        cfg = MoEConfig(num_experts=8, hidden=HIDDEN, intermediate=INTER, top_k=2, tp_size=tp_size)
        assert cfg.local_intermediate == INTER // tp_size
        assert _tp_shard("gate", _roles()["gate"], INTER, tp_size, 0).shape[0] == cfg.local_intermediate
