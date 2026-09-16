"""The hybrid MoE split follows Equation 4 of the FreeToken paper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _Cache:
    """Bind the real split method without constructing a GPU cache."""

    def __init__(self, fraction=0.25):
        from freetoken.moe.offload_cache import OffloadMoeCache

        self.hybrid_fetch_fraction = fraction
        self._fn = OffloadMoeCache._fetch_fraction_for

    def frac(self, layer_id, n_tokens, top_k=6):
        ids = torch.zeros(n_tokens, top_k, dtype=torch.int32)
        return self._fn(self, layer_id, ids)


class TestTheSplitIsTheProfiledBandwidthRatio:
    @pytest.mark.parametrize("n_tokens", [1, 2, 5, 6, 64])
    def test_token_width_does_not_rescale_equation_four(self, n_tokens):
        # The cache kernel deduplicates all routes first, so its miss count m already
        # reflects the wider batch. Equation 4 applies B_P/B_H to that m exactly once.
        assert _Cache(0.28).frac(0, n_tokens) == pytest.approx(0.28)

    def test_a_disabled_fraction_stays_disabled(self):
        # 0.0 means "use the fixed hybrid_max_fetch cap instead". Scaling it would turn
        # the cap path on for speculative steps only.
        assert _Cache(0.0).frac(0, 6) == 0.0

    def test_tensor_shape_does_not_change_the_ratio(self):
        c = _Cache(0.3)
        assert c._fn(c, 0, torch.zeros(6, dtype=torch.int32)) == pytest.approx(0.3)


class TestCpuExecutorScratchWidth:
    def test_anchor_plus_five_drafts_needs_six_rows(self):
        from freetoken.engine.engine import _cpu_moe_max_tokens

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                dsv4_args=SimpleNamespace(dspark_enabled=True, dspark_block_size=5)
            ),
            max_running_req=1,
            cuda_graph_max_bs=None,
        )
        assert _cpu_moe_max_tokens(config) == 6

    def test_bound_scales_with_requests_and_graph_capture(self):
        from freetoken.engine.engine import _cpu_moe_max_tokens

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                dsv4_args=SimpleNamespace(dspark_enabled=True, dspark_block_size=5)
            ),
            max_running_req=3,
            cuda_graph_max_bs=32,
        )
        assert _cpu_moe_max_tokens(config) == 32 * 6
