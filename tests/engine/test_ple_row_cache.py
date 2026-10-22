import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig


@pytest.mark.parametrize("rows", [-8, 1, 7, 9, (1 << 24) + 8])
def test_ple_cache_rejects_invalid_capacity(rows):
    with pytest.raises(ValueError, match="ple_row_cache_rows"):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, ple_row_cache_rows=rows)


def test_ple_cache_requires_disk_backend():
    with pytest.raises(ValueError, match="disk PLE"):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16,
                     ple_backend="pinned", ple_row_cache_rows=8)


@pytest.mark.parametrize("rows", [0, 8, 65536, 1 << 24])
def test_ple_cache_accepts_bounded_capacity(rows):
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, ple_row_cache_rows=rows)
    assert config.ple_row_cache_rows == rows
