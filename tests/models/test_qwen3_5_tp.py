import torch
from freetoken.models.qwen3_5_moe.weight import _shard_tp, _shard_tp_parts


def test_shard_tp():
    t = torch.arange(32).reshape(8, 4)
    s0, s1 = _shard_tp(t, rank=0, world_size=2, dim=0), _shard_tp(t, rank=1, world_size=2, dim=0)
    assert s0.shape == (4, 4) and s1.shape == (4, 4)
    torch.testing.assert_close(torch.cat([s0, s1]), t)
    assert _shard_tp(t, rank=0, world_size=1, dim=0).equal(t)
    assert _shard_tp(torch.arange(32).reshape(4, 8), rank=0, world_size=2, dim=1).shape == (4, 4)


def test_shard_tp_parts():
    t = torch.arange(48).reshape(12, 4)
    s0 = _shard_tp_parts(t, (4, 4, 4), rank=0, world_size=2)
    s1 = _shard_tp_parts(t, (4, 4, 4), rank=1, world_size=2)
    assert s0.shape == (6, 4)
    for i in range(3):
        torch.testing.assert_close(torch.cat([s0[i*2:i*2+2], s1[i*2:i*2+2]]), t[i*4:i*4+4])


def test_shard_tp_parts_replicate():
    t = torch.arange(16).reshape(8, 2)
    local = (2, 4)
    kw = dict(tensor=t, part_sizes=(4, 4), world_size=4, local_part_sizes=local)
    s0, s1, s2, s3 = [_shard_tp_parts(rank=r, **kw) for r in range(4)]
    assert s0.shape == (6, 2)
    torch.testing.assert_close(s0[:2], s1[:2])   # ranks 0,1 share head 0
    torch.testing.assert_close(s2[:2], s3[:2])   # ranks 2,3 share head 1
    assert not torch.equal(s0[:2], s2[:2])        # different heads
    torch.testing.assert_close(s0[2:], s2[2:])    # replicated part identical
