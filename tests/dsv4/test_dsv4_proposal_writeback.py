"""DSpark proposal layout from the paper's gamma queries to target verification."""

from __future__ import annotations

import pathlib


def _method(name: str) -> str:
    src = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python" / "freetoken" / "engine" / "engine.py"
    ).read_text()
    i = src.index(f"    def {name}(")
    end = src.find("\n    def ", i + 10)
    return src[i : len(src) if end < 0 else end]


def _scheduler_method(name: str) -> str:
    src = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python" / "freetoken" / "scheduler" / "scheduler.py"
    ).read_text()
    i = src.index(f"    def {name}(")
    end = src.find("\n    def ", i + 10)
    return src[i : len(src) if end < 0 else end]


class TestProposalLayout:
    def test_all_gamma_proposals_fill_target_rows_one_through_gamma(self):
        body = _method("draft_into_batch")
        assert "i * span + 1:(i + 1) * span" in body
        assert "proposed[i * k:(i + 1) * k]" in body

    def test_proposals_stay_on_gpu_until_acceptance(self):
        body = _method("draft_into_batch")
        assert "batch.draft_tokens = proposed" in body
        assert "req.input_ids[-k:]" not in body

    def test_acceptance_reads_the_saved_proposals_not_noise_placeholders(self):
        body = _method("_finish_speculative")
        assert "proposed_gpu = batch.draft_tokens" in body
        assert "proposed_cpu[i * k:(i + 1) * k]" in body

    def test_only_the_accepted_prefix_reaches_the_request_buffer(self):
        body = _method("_finish_speculative")
        assert "req.append_host(proposed[:n_acc]" in body

    def test_target_span_is_anchor_plus_gamma(self):
        assert "span = 1 + k" in _method("draft_into_batch")

    def test_draft_masks_are_built_on_gpu_from_the_checkpoint_noise_id(self):
        dspark = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python" / "freetoken" / "models" / "deepseek_v4" / "dspark.py"
        ).read_text()
        start = dspark.index("    def propose(")
        end = dspark.index("\n    def embed_block", start)
        body = dspark[start:end]
        assert "torch.full(" in body
        assert "self.noise_token_id" in body
        assert "draft_ids[:, 0] = input_ids" in body

    def test_bonus_writeback_uses_the_post_acceptance_frontier(self):
        body = _scheduler_method("_forward")
        assert "[req.device_len - 1 for req in batch.reqs]" in body
        assert "self.token_pool[rows, positions]" in body

    def test_acceptance_uses_each_requests_sampling_params(self):
        body = _method("_finish_speculative")
        assert "sp = req.sampling_params" in body
        assert "sp = batch.reqs[0].sampling_params" not in body
