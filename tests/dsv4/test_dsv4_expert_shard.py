"""The DSV4 expert reader must cut every piece on the axis its bank actually uses.

Under TP the routed-expert pieces are sliced along the intermediate dim, but that axis
is not in the same place -- nor the same scale -- in every tensor:

  * ``gate`` / ``up`` (checkpoint ``w1`` / ``w3``) carry I on the ROW axis, packed two
    e2m1 codes per byte along H, so the rows are unscaled: ``[i_lo : i_lo+I/tp]``.
  * their ``_scale`` companions are ``[I, H/32]``: the 32-wide e8m0 grid blocks along
    H, NOT along I, so the scale slices on the SAME axis as its weight undivided.
  * ``down`` (``w2``) carries I on the COLUMN axis, packed -> ``[..., i_lo/2 : ...]``.
  * ``down_scale`` carries I/32 on the column axis -> ``[..., i_lo/32 : ...]``.

Cutting a scale on the wrong axis, or forgetting the /2 and /32, does not raise: it
produces a bank whose values are simply the wrong experts' numbers. So these tests
assert the strongest available property -- the per-rank slices must concatenate back
into the original tensor exactly, on the axis each role uses.
"""

import pytest
import torch

pytest.importorskip("freetoken.models.deepseek_v4.weight")

H, I, E = 512, 256, 4
TP = 2

# role -> (full per-expert shape, axis carrying I, divisor on that axis)
ROLES = {
    "gate": ((I, H // 2), 0, 1),
    "up": ((I, H // 2), 0, 1),
    "gate_scale": ((I, H // 32), 0, 1),
    "up_scale": ((I, H // 32), 0, 1),
    "down": ((H, I // 2), 1, 2),
    "down_scale": ((H, I // 32), 1, 32),
}


def _shard(role, t, rank):
    from freetoken.models.deepseek_v4.weight import shard_expert_piece

    return shard_expert_piece(role, t, rank=rank, tp_size=TP)


@pytest.mark.parametrize("role", sorted(ROLES))
def test_rank_slices_rebuild_the_piece_exactly(role):
    """Disjoint, gapless, in order: concatenating every rank reproduces the source."""
    shape, axis, div = ROLES[role]
    t = torch.arange(shape[0] * shape[1], dtype=torch.int32).view(*shape)

    parts = [_shard(role, t, r) for r in range(TP)]

    i_local = I // TP
    n_axis = shape[axis] // div
    assert n_axis % TP == 0, f"{shape[axis]} on axis {axis} must divide over {TP} ranks"
    assert all(p.shape[axis] == n_axis // TP for p in parts), f"{role}: wrong slice width"
    assert torch.equal(torch.cat(parts, dim=axis), t), f"{role}: ranks do not tile the I axis"


@pytest.mark.parametrize("role", sorted(ROLES))
def test_tp1_yields_the_checkpoint_tensor_untouched(role):
    """The single-GPU path must not pay for a copy or a slice it does not need."""
    from freetoken.models.deepseek_v4.weight import shard_expert_piece

    shape = ROLES[role][0]
    t = torch.zeros(*shape, dtype=torch.bfloat16)
    out = shard_expert_piece(role, t, rank=0, tp_size=1)
    assert out is t, "tp_size=1 must be identity, not a copy"


def test_pieces_are_in_theirs_own_storage():
    """A narrow() view keeps the WHOLE parent tensor alive behind the slice.

    Every rank would then pay for the experts it just threw away -- measured at
    5.1 GiB per GPU on DeepSeek-V4-Flash at TP=4, silently charged to the model and
    taken out of the KV / expert-cache budget.
    """
    shape, axis, _ = ROLES["down"]
    t = torch.zeros(*shape, dtype=torch.uint8)
    out = _shard("down", t, 0)
    assert out.stride() == (out.shape[1], 1), "slice must be contiguous"
    assert out.untyped_storage().size() < t.untyped_storage().size(), "slice still aliases its parent"
