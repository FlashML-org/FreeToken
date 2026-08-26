"""Paper-faithful adaptive verification tests that do not require CUDA."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.engine.graph import GraphRunner, SharedSpecCarryJournal
from freetoken.models.deepseek_v4.dspark import (
    DSparkAcceptanceFallback,
    choose_adaptive_draft_width,
)
from freetoken.scheduler.scheduler import Scheduler


def test_acceptance_fallback_trips_only_after_minimum_sample():
    fallback = DSparkAcceptanceFallback(0.60, min_drafted=8, cooldown_steps=3)

    assert fallback.record(2, 5) is None
    trip = fallback.record(1, 3)

    assert trip == pytest.approx((3 / 8, 8))


def test_acceptance_fallback_skips_exact_cooldown_then_probes():
    fallback = DSparkAcceptanceFallback(0.60, min_drafted=4, cooldown_steps=3)
    assert fallback.record(1, 4) is not None

    assert [fallback.take_decision() for _ in range(3)] == [
        (False, False),
        (False, False),
        (False, False),
    ]
    assert fallback.take_decision() == (True, True)
    assert fallback.take_decision() == (True, False)


def test_acceptance_fallback_keeps_high_acceptance_speculating():
    fallback = DSparkAcceptanceFallback(0.60, min_drafted=8, cooldown_steps=3)

    assert fallback.record(5, 8) is None
    assert fallback.take_decision() == (True, False)


def test_acceptance_fallback_reset_clears_request_state():
    fallback = DSparkAcceptanceFallback(0.60, min_drafted=4, cooldown_steps=3)
    fallback.record(1, 4)

    fallback.reset()

    assert fallback.take_decision() == (True, False)
    assert fallback.record(1, 3) is None


def test_scheduler_fallback_bypasses_block_before_appending_noise():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._speculative = SimpleNamespace(block_size=5, noise_token_id=17)
    scheduler.engine = SimpleNamespace(should_speculate=lambda _req: False)
    scheduler._warned_non_greedy = False
    req = SimpleNamespace(remain_len=100, input_ids=torch.tensor([1, 2]))
    batch = SimpleNamespace(reqs=[req], is_decode=True, phase="decode")

    scheduler._maybe_make_speculative(batch)

    assert batch.phase == "decode"
    assert req.input_ids.tolist() == [1, 2]


def test_profile_cliff_stops_before_expensive_graph_bucket():
    width = choose_adaptive_draft_width(
        [0.9, 0.9],
        draft_cost_ms=0.0,
        verify_cost_ms=[1.0, 1.0, 100.0],
    )
    assert width == 1


def test_flat_verify_cost_admits_the_whole_prefix():
    width = choose_adaptive_draft_width(
        [0.8, 0.7, 0.6],
        draft_cost_ms=2.0,
        verify_cost_ms=[10.0, 10.0, 10.0, 10.0],
    )
    assert width == 3


def test_selector_uses_cumulative_survival_not_raw_confidence():
    # The second token's contribution is 0.5 * 0.9, not 0.9 by itself.
    width = choose_adaptive_draft_width(
        [0.5, 0.9],
        draft_cost_ms=0.0,
        verify_cost_ms=[1.0, 1.0, 1.4],
    )
    assert width == 1


def test_tie_keeps_the_smaller_prefix_like_argmax():
    assert choose_adaptive_draft_width(
        [1.0], draft_cost_ms=0.0, verify_cost_ms=[1.0, 2.0]
    ) == 0


def test_cost_curve_must_cover_every_prefix():
    with pytest.raises(ValueError, match="every width"):
        choose_adaptive_draft_width(
            [0.9, 0.9], draft_cost_ms=1.0, verify_cost_ms=[1.0, 2.0]
        )


def test_prefix_graphs_share_the_maximum_carry_journal_storage():
    key = (3, "compress", 128)
    storage = {key: [torch.zeros(1) for _ in range(6)]}
    journal = SharedSpecCarryJournal(storage)
    journal.reset(2)
    journal.record(key, torch.tensor([4.0]))
    journal.record(key, torch.tensor([5.0]))

    [(seen_key, pieces)] = list(journal.items())
    assert seen_key == key
    assert len(pieces) == 2
    assert torch.cat(pieces).tolist() == [4.0, 5.0]
    assert storage[key][2].item() == 0.0


def test_shared_carry_journal_rejects_prefix_overflow():
    key = (3, "compress", 128)
    journal = SharedSpecCarryJournal({key: [torch.zeros(1)]})
    journal.reset(1)
    journal.record(key, torch.ones(1))
    with pytest.raises(RuntimeError, match="overflow"):
        journal.record(key, torch.ones(1))


class _ChooseTwo:
    block_size = 5

    def record_and_choose(self, confidence, uid):
        assert confidence.tolist() == pytest.approx([0.9] * 5)
        assert uid == 7
        return 2


def test_engine_compacts_only_target_views_and_keeps_full_allocation_frontier():
    engine = Engine.__new__(Engine)
    engine._adaptive_verification = _ChooseTwo()
    ids = torch.arange(20, dtype=torch.int32)
    req = SimpleNamespace(input_ids=ids[:15], _ids_buf=ids, uid=7, device_len=15)
    metadata = SimpleNamespace(segments=[(0, 6, 3, 9)])
    batch = SimpleNamespace(
        reqs=[req],
        padded_size=1,
        spec_block=5,
        draft_confidence=torch.full((5,), 0.9),
        draft_tokens=torch.arange(5, dtype=torch.int32),
        draft_probs=torch.empty(5, 4),
        input_ids=torch.arange(6, dtype=torch.int32),
        positions=torch.arange(9, 15, dtype=torch.int32),
        out_loc=torch.arange(6, dtype=torch.int32),
        attn_metadata=metadata,
    )

    engine.adapt_speculative_batch(batch)

    assert batch.spec_block == 2
    assert batch.input_ids.tolist() == [0, 1, 2]
    assert batch.positions.tolist() == [9, 10, 11]
    assert batch.out_loc.tolist() == [0, 1, 2]
    assert batch.draft_tokens.tolist() == [0, 1]
    assert batch.draft_confidence.tolist() == pytest.approx([0.9, 0.9])
    assert metadata.segments == [(0, 3, 3, 9)]
    assert req.input_ids.numel() == 12
    assert req.device_len == 15, "tail release still needs the gamma-wide frontier"


def test_greedy_verify_reduces_logits_before_crossing_pcie():
    import inspect

    body = inspect.getsource(Engine._finish_speculative)
    assert 'logits.argmax(dim=-1).to(torch.int32).to("cpu"' in body
    assert 'logits.to("cpu"' not in body


def test_probabilistic_verify_does_not_compute_unused_target_argmax():
    import inspect

    body = inspect.getsource(Engine._finish_speculative)
    assert "any_greedy = any(req.sampling_params.is_greedy" in body
    assert "if any_greedy" in body
    assert "else None" in body


def test_capacity_profile_flushes_cache_once_not_before_every_replay():
    import inspect

    body = inspect.getsource(GraphRunner._profile_graph_ms)
    assert body.count("self._reset_moe_offload_cache()") == 1
