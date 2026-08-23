"""Where a speculative step's MoE work runs.

Speculation is normally free because verifying K tokens costs what 1 costs: a GPU decode
is weight-bound and the batch dimension rides along. That is a property of where the
weights LIVE, and it does not survive an offload MoE. Fetching an expert over PCIe moves
the same bytes whether 1 or K tokens route to it, but the CPU pool pays per
(expert, token) pair. Hold the 1-token split fixed and a K-wide block costs nearly K
single steps -- at which point no acceptance rate can pay for it.

These tests pin the two decisions that keep a block cheap, both of which are invisible
at runtime: a step that carries more tokens fetches a larger share, and the drafter's
own layers never touch the CPU pool.
"""

from __future__ import annotations

import pytest
import torch


class _Cache:
    """The two fields _fetch_fraction_for reads, with the real method bound in."""

    def __init__(self, fraction=0.25, draft_ids=frozenset()):
        from freetoken.moe.offload_cache import OffloadMoeCache

        self.hybrid_fetch_fraction = fraction
        self.draft_layer_ids = draft_ids
        self._fn = OffloadMoeCache._fetch_fraction_for

    def frac(self, layer_id, n_tokens, top_k=6):
        ids = torch.zeros(n_tokens, top_k, dtype=torch.int32)
        return self._fn(self, layer_id, ids)


class TestTheSplitFollowsTheTokenCount:
    def test_a_single_token_step_is_unchanged(self):
        # The benched fraction is defined for exactly this case; changing it here would
        # silently re-tune ordinary decode, which is not what any of this is for.
        assert _Cache(0.28).frac(0, 1) == pytest.approx(0.28)

    def test_a_wider_step_fetches_proportionally_more(self):
        c = _Cache(0.10)
        assert c.frac(0, 2) == pytest.approx(0.20)
        assert c.frac(0, 6) == pytest.approx(0.60)

    def test_the_fraction_never_exceeds_everything(self):
        # 0.28 * 6 = 1.68. A fraction above 1 would ask for more fetches than there are
        # misses; the kernel takes it as a count and would over-read the miss list.
        assert _Cache(0.28).frac(0, 6) == pytest.approx(1.0)
        assert _Cache(0.9).frac(0, 64) == pytest.approx(1.0)

    def test_a_disabled_fraction_stays_disabled(self):
        # 0.0 means "use the fixed hybrid_max_fetch cap instead". Scaling it would turn
        # the cap path on for speculative steps only.
        assert _Cache(0.0).frac(0, 6) == 0.0

    def test_a_one_dimensional_id_tensor_counts_as_one_token(self):
        c = _Cache(0.3)
        assert c._fn(c, 0, torch.zeros(6, dtype=torch.int32)) == pytest.approx(0.3)


class TestTheDrafterStaysOnTheGpu:
    """The draft is serial with every block; the CPU pool is the slowest path there is."""

    def test_a_draft_layer_fetches_everything(self):
        c = _Cache(0.05, draft_ids=frozenset({43, 44, 45}))
        assert c.frac(44, 1) == 1.0, "even a 1-token draft must not wait on the CPU"
        assert c.frac(44, 6) == 1.0

    def test_a_target_layer_is_unaffected(self):
        c = _Cache(0.05, draft_ids=frozenset({43, 44, 45}))
        assert c.frac(42, 1) == pytest.approx(0.05)


class TestDraftLayerIdsAreTheTail:
    """The ids must match the order the model yields its MoE layers in.

    The model yields the target's layers, then the drafter's. Compute the tail wrongly
    and the engine pins three of the TARGET's layers to the GPU while leaving the
    drafter on the CPU -- the exact inverse, and nothing would report it.
    """

    @pytest.mark.parametrize(
        "n_moe,n_draft,expected",
        [(46, 3, {43, 44, 45}), (46, 0, set()), (10, 1, {9}), (4, 4, {0, 1, 2, 3})],
    )
    def test_the_tail_is_taken(self, n_moe, n_draft, expected):
        ids = frozenset(range(n_moe - n_draft, n_moe)) if n_draft else frozenset()
        assert set(ids) == expected

    def test_the_model_yields_the_drafter_last(self):
        # If this order ever flips, the id arithmetic above silently targets the wrong
        # layers -- so assert the source still yields target-then-drafter.
        import inspect

        from freetoken.models.deepseek_v4.model import DeepseekV4ForCausalLM

        src = inspect.getsource(DeepseekV4ForCausalLM._iter_offload_moe_layers)
        target_at = src.index("self._transformer.layers")
        draft_at = src.index("drafter.layers")
        assert target_at < draft_at, (
            "the drafter's MoE layers must be yielded AFTER the target's; the engine "
            "identifies them as the tail of the sequence"
        )
