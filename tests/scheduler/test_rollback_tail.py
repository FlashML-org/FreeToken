"""rollback_tail after a widened decode step: the request keeps exactly the pages of its committed tokens, the rest go back, and the next step allocates no page twice."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req
from freetoken.scheduler.cache import CacheManager
from freetoken.utils import div_ceil

K = 3
NUM_PAGES = 64


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cached", [8, 9, 11])
@pytest.mark.parametrize("n", range(1, K + 2))
def test_rollback_keeps_committed_pages_only(page_size, cached, n):
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, page_size, page_table, "naive")
    req = Req(input_ids=torch.zeros(cached, dtype=torch.int32), table_idx=0, cached_len=0, output_len=32,
              uid=0, sampling_params=SimpleNamespace(), cache_handle=None)
    cm.allocate_paged([req])
    req.complete_one()  # cached_len = cached, the pending token sits at position `cached`

    for _ in range(2):
        entry = req.cached_len
        req.widen_pending(K)
        cm.allocate_paged([req])
        req.commit_pending(n)
        with cm.lazy_free_region():
            cm.rollback_tail(req, req.cached_len, entry + K + 1)
        kept = div_ceil(req.cached_len, page_size)
        assert len(cm.free_slots) == NUM_PAGES - kept
        assert not set(cm.free_slots.tolist()) & set(cm.page_table[0, : kept * page_size : page_size].tolist())
