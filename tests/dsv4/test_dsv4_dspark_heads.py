"""The dSpark heads, and the two contracts a review pass caught after they were written.

Both bugs fixed here were silent, and both were invisible to the shape checks that
already existed:

* The drafter reused the target's head WEIGHT. Under tensor parallelism that head is
  vocabulary-parallel, so a bare F.linear returns this rank's vocabulary slice, while
  the Markov bias is full-vocabulary because the Markov head is replicated. Adding them
  broadcasts into nonsense or raises, depending on the TP size -- and at TP=1, where
  most testing happens, it is perfectly correct.
* store_context_kv sliced the RoPE table from the front. Every context after the first
  block starts at a non-zero position, so it would rotate the drafter's keys as though
  they began at zero. Attention still runs; acceptance just quietly drops.

CPU-only.
"""

from __future__ import annotations

import pytest
import torch

import freetoken.distributed.info as info_mod
from freetoken.distributed import DistributedInfo
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.dspark import ConfidenceHead, MarkovHead

VOCAB, DIM, RANK = 64, 32, 8


def _args(**kw) -> DeepseekV4Args:
    base = dict(
        vocab_size=VOCAB, dim=DIM, dspark_markov_rank=RANK, dspark_block_size=5,
        n_mtp_layers=3, dspark_enabled=True, n_layers=4, n_hash_layers=1,
        compress_ratios=(0, 4, 128, 4), max_seq_len=4096, max_batch_size=1,
    )
    base.update(kw)
    return DeepseekV4Args(**base)


@pytest.fixture(autouse=True)
def _tp1():
    info_mod._TP_INFO = DistributedInfo(0, 1)
    yield
    info_mod._TP_INFO = DistributedInfo(0, 1)


class TestMarkovHead:
    def test_the_bias_covers_the_whole_vocabulary(self):
        # It must, because the Markov head is replicated under TP while the target's
        # output head is sharded. A per-rank bias could not be added to gathered logits.
        head = MarkovHead(_args())
        bias = head.bias(head.embed(torch.tensor([3, 7])))
        assert bias.shape == (2, VOCAB)

    def test_the_embedding_is_the_low_rank_width(self):
        head = MarkovHead(_args())
        assert head.embed(torch.tensor([1, 2, 3])).shape == (3, RANK)

    def test_different_previous_tokens_give_different_biases(self):
        # The head's entire purpose is to condition on the previous token; a bias that
        # ignores it would leave the drafter proposing the same continuation everywhere.
        head = MarkovHead(_args())
        torch.nn.init.normal_(head.markov_w1.weight)
        torch.nn.init.normal_(head.markov_w2)
        a = head.bias(head.embed(torch.tensor([1])))
        b = head.bias(head.embed(torch.tensor([2])))
        assert not torch.allclose(a, b)


class TestConfidenceHead:
    def test_it_scores_one_value_per_position(self):
        head = ConfidenceHead(_args())
        out = head(torch.randn(4, DIM), torch.randn(4, RANK))
        assert out.shape == (4,)

    def test_scores_are_probabilities(self):
        # draft_width compares them against a threshold, so anything outside [0, 1]
        # would make the adaptive width meaningless.
        head = ConfidenceHead(_args())
        torch.nn.init.normal_(head.proj, std=10.0)  # push the pre-sigmoid range wide
        out = head(torch.randn(16, DIM), torch.randn(16, RANK))
        assert bool((out >= 0).all() and (out <= 1).all())

    def test_it_consumes_both_the_hidden_and_the_markov_embedding(self):
        head = ConfidenceHead(_args())
        assert head.proj.shape == (1, DIM + RANK), (
            "confidence reads the hidden state AND the previous-token embedding"
        )


class TestLogitsUnderTensorParallelism:
    """The bug the review caught: the drafter must not use the head weight directly."""

    def test_the_draft_logits_are_full_vocabulary_at_tp4(self):
        from freetoken.models.deepseek_v4.dspark import DSparkDrafter

        info_mod._TP_INFO = DistributedInfo(0, 4)
        args = _args(dim=512, moe_inter_dim=512, n_heads=8, o_groups=4, vocab_size=64)
        with torch.device("meta"):
            drafter = DSparkDrafter(args)

        # The target's logits() all-gathers the vocabulary; stand in for it here.
        def target_logits(h):
            return torch.zeros(h.shape[0], args.vocab_size)

        drafter.markov_head = MarkovHead(args)  # real (small) weights, off meta
        drafter.confidence_head = ConfidenceHead(args)
        logits, conf = drafter.logits(
            torch.zeros(3, args.dim), target_logits, torch.tensor([1, 2, 3])
        )
        assert logits.shape == (3, args.vocab_size), (
            "draft logits must span the full vocabulary; a per-rank slice cannot be "
            "added to the replicated Markov bias"
        )
        assert conf.shape == (3,)


class TestContextKvPositions:
    """The other bug: RoPE must key on absolute positions, not a front slice."""

    def test_a_position_length_mismatch_is_refused(self):
        from freetoken.models.deepseek_v4.dspark import DSparkDrafter

        args = _args(dim=512, moe_inter_dim=512, n_heads=8, o_groups=4)
        with torch.device("meta"):
            drafter = DSparkDrafter(args)
        with pytest.raises(ValueError, match="positions"):
            drafter.store_context_kv(
                torch.zeros(5, args.dim), torch.arange(3), torch.arange(5)
            )


class TestBlockInputIds:
    """What a draft block is actually fed.

    dSpark is a BLOCK predictor: it gets the last committed token, then noise at every
    unknown position, and fills them in one pass. Feeding zeros or repeating the last
    token puts it off its training distribution -- proposals stop being accepted, which
    reads as "the drafter is weak" rather than "we fed it the wrong thing".
    """

    def _drafter(self, **kw):
        from freetoken.models.deepseek_v4.dspark import DSparkDrafter

        opts = dict(dim=512, moe_inter_dim=512, n_heads=8, o_groups=4,
                    dspark_noise_token_id=128799)
        opts.update(kw)
        args = _args(**opts)
        with torch.device("meta"):
            return DSparkDrafter(args), args

    def test_the_first_position_is_the_real_last_token(self):
        d, _ = self._drafter()
        ids = d.block_input_ids(4242, torch.device("cpu"))
        assert int(ids[0]) == 4242

    def test_every_other_position_is_the_noise_token(self):
        d, _ = self._drafter()
        ids = d.block_input_ids(7, torch.device("cpu"))
        assert ids.numel() == d.block_size
        assert all(int(t) == 128799 for t in ids[1:]), (
            "unknown positions must carry the checkpoint's noise token"
        )

    def test_a_checkpoint_without_a_noise_token_is_refused(self):
        d, _ = self._drafter(dspark_noise_token_id=-1)
        with pytest.raises(ValueError, match="noise"):
            d.block_input_ids(1, torch.device("cpu"))
