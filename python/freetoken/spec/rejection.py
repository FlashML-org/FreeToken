"""Rejection sampling for a speculative verify.

With argmax drafts the draft distribution q is one-hot at the drafted id. Chain rejection with a one-hot q accepts d with probability p(d) and on a reject samples p with d removed.
Drawing y ~ p at every row and accepting drafts while y == d gives exactly that: P(accept) = p(d), and a mismatching y is p conditioned on y != d.
So the verify samples each row as decode would and compares prefixes; greedy rows reduce to the argmax prefix match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from freetoken.engine.sample import BatchSamplingArgs, Sampler


class RejectionSampler:
    def __call__(
        self,
        logits: torch.Tensor,   # [bs*(k+1), V], k+1 rows per request
        drafts: torch.Tensor,   # [bs, k] int64
        args: BatchSamplingArgs,
        sampler: Sampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(out [bs, k+1], emitted [bs]): emitted counts the accepted drafts (0..k); a request commits ``out[i, : emitted[i] + 1]``."""
        bs, k = drafts.shape
        target = sampler.sample_verify(logits, args, drafts).view(bs, k + 1).to(torch.int64)
        emitted = (drafts == target[:, :k]).to(torch.int64).cumprod(dim=1).sum(dim=1)
        return target, emitted
