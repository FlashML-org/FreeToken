"""radix_lookahead=1 (bigram keys): a prefix hit also matches the token after it, so a draft KV slot that depends on the next token (MTP) is only reused when that token agrees. CPU, real CacheManager + real trees."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.mm import MM_PAD_SHIFT_VALUE
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pend(ids):
    return SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))


def _publish(cm, table_idx, ids, *, finished=True, track=None):
    """A request whose ids end with one sampled token past its committed cached_len, committed into the tree."""
    handle = cm.match_req(_pend(ids)).cuda_handle
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx, cached_len=0, output_len=0,
              uid=table_idx, sampling_params=SamplingParams(), cache_handle=handle)
    cm.lock(handle)
    req.device_len = len(ids) - 1
    cm.allocate_paged([req])
    req.cached_len = len(ids) - 1
    if cm.is_hybrid:
        pool = cm.linear_state_pool
        req.linear_slot_idx = pool.alloc(1)[0]
        req.mamba_ping_pong = tuple(pool.alloc(2))
        req.mamba_next_track_idx = 1
        req.mamba_last_track_seqlen = track
    with cm.lazy_free_region():
        cm.cache_req(req, finished=finished)
    return req


def _hybrid(lookahead):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    pool = LinearStatePool(group=g, num_slots=32, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1)
    return CacheManager(256, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool,
                        radix_lookahead=lookahead)


@pytest.mark.parametrize("page_size", [1, 4])
def test_a_hit_takes_the_slot_before_the_fork_from_the_branch_it_follows(page_size):
    a = list(range(1, 18))
    b = a[:8] + list(range(100, 109))  # forks from a at position 8
    query = b[:12] + list(range(200, 205))
    rows = {}
    for lookahead in (0, 1):
        page_table = torch.zeros(4, 64, dtype=torch.int32)
        cm = CacheManager(64, page_size, page_table, "radix", radix_lookahead=lookahead)
        _publish(cm, 0, a, finished=False)
        req_b = _publish(cm, 1, b, finished=False)
        hit = cm.match_req(_pend(query)).cuda_handle
        rows[lookahead] = (hit.cached_len, hit.get_matched_indices(), page_table[req_b.table_idx].clone())
    plain_len, plain_slots, _ = rows[0]
    bigram_len, bigram_slots, b_row = rows[1]
    # the query shares 12 tokens with b; a bigram key needs the token after, so one position less (page-aligned)
    assert plain_len == 12 // page_size * page_size
    assert bigram_len == 11 // page_size * page_size
    # slot 7's draft KV depends on the token at 8, where b differs from a: it must be b's slot
    assert bigram_slots[7] == b_row[7]
    assert plain_slots[7] != b_row[7] or page_size > 1


def test_the_prompt_end_key_uses_the_sampled_token():
    prompt, sampled = list(range(1, 11)), 77
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix", radix_lookahead=1)
    _publish(cm, 0, prompt + [sampled], finished=False)
    assert cm.match_req(_pend(prompt + [sampled, 5])).cuda_handle.cached_len == len(prompt)
    assert cm.match_req(_pend(prompt + [sampled + 1, 5])).cuda_handle.cached_len == len(prompt) - 1


def test_multimodal_pseudo_ids_keep_distinct_keys():
    top = MM_PAD_SHIFT_VALUE + (1 << 30) - 1
    image_a, image_b = [top] * 4, [top - 1] * 4
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix", radix_lookahead=1)
    _publish(cm, 0, [1, 2] + image_a + [3, 4], finished=False)
    assert cm.match_req(_pend([1, 2] + image_a + [3, 4, 9])).cuda_handle.cached_len == 7
    # the first image row is keyed with the second, so a different image matches only before it
    assert cm.match_req(_pend([1, 2] + image_b + [3, 4, 9])).cuda_handle.cached_len == 1


def test_hybrid_snapshot_hits_only_when_the_token_after_it_agrees():
    tokens = list(range(1, 18))
    for lookahead, expect_same, expect_diff in ((0, 8, 8), (1, 8, None)):
        cm = _hybrid(lookahead)
        # the frozen snapshot at 8 (an anchor or a track boundary), then the finish donates the live state
        _publish(cm, 0, tokens, finished=True, track=8)
        same = cm.match_req(_pend(tokens[:9] + [500, 501]))
        diff = cm.match_req(_pend(tokens[:8] + [500, 501]))
        assert (same.cuda_handle.cached_len if same.mamba_value is not None else None) == expect_same
        assert (diff.cuda_handle.cached_len if diff.mamba_value is not None else None) == expect_diff


def test_swa_radix_keeps_token_keys():
    with pytest.raises(AssertionError):
        CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "swa_radix", sliding_window_size=8, radix_lookahead=1)
