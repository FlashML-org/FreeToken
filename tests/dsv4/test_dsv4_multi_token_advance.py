"""Advancing a request by more than one token.

A speculative step emits a block, and each request keeps a different prefix of it. The
engine's decode step advances every request by exactly one, so this is the first thing
that has to change -- and getting it wrong desynchronises a request's position from its
KV, which produces wrong text rather than an error.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Req, SamplingParams


def _req(prompt_len: int = 8, budget: int = 64) -> Req:
    return Req(
        input_ids=torch.arange(prompt_len, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=budget,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )


class TestCompleteN:
    def test_it_advances_by_exactly_n(self):
        r = _req()
        before = r.device_len
        r.complete_n(3)
        assert r.device_len == before + 3

    def test_cached_len_marks_where_the_step_started(self):
        # The accepted prefix becomes cached history, and the next step extends from
        # the end of it. Pointing cached_len anywhere else re-runs or skips tokens.
        r = _req()
        start = r.device_len
        r.complete_n(4)
        assert r.cached_len == start
        assert r.extend_len == 4

    def test_n_equals_one_matches_complete_one(self):
        a, b = _req(), _req()
        a.complete_one()
        b.complete_n(1)
        assert (a.device_len, a.cached_len) == (b.device_len, b.cached_len)

    def test_a_step_that_advances_by_nothing_is_refused(self):
        # A speculative step always emits at least one token (the target's own token at
        # the first rejection). Zero means the caller lost that guarantee.
        with pytest.raises(ValueError, match="at least one"):
            _req().complete_n(0)

    def test_repeated_steps_accumulate(self):
        r = _req()
        start = r.device_len
        for n in (3, 1, 5, 2):
            r.complete_n(n)
        assert r.device_len == start + 11

    def test_append_host_and_complete_n_stay_consistent(self):
        # The token buffer and the position counter must agree: device_len is what the
        # KV and page table are sized against, input_ids is what the client receives.
        r = _req(prompt_len=8)
        r.append_host(torch.tensor([101, 102, 103], dtype=torch.int32))
        r.complete_n(3)
        assert r.input_ids.numel() == r.device_len == 11
