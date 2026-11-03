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


def reject(
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


def sparse_q_rejection(
    logits: torch.Tensor,   # [bs*(k+1), V]
    drafts: torch.Tensor,   # [bs, k] int64, slot j drawn from q[:, j]
    cand: torch.Tensor,     # [bs, k, K] the support of q
    q: torch.Tensor,        # [bs, k, K] q over cand
    args: BatchSamplingArgs,
    sampler: Sampler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``reject`` for drafts sampled from q: slot j accepts d with probability min(1, p(d)/q(d)) and on a reject samples max(0, p - q); greedy rows and rows whose q is not a distribution take the one-hot result."""
    from freetoken.engine.sample import sampling_module

    out, emitted = reject(logits, drafts, args, sampler)
    p = sampler.probs(logits, args, drafts)
    if p is None:
        return out, emitted
    bs, k = drafts.shape
    p = p.view(bs, k + 1, -1)
    # a NaN or a wrong total in q would reach the target's output through the residual
    ok = torch.isfinite(q).flatten(1).all(dim=1) & ((q.sum(dim=-1) - 1.0).abs() < 1e-3).all(dim=1)
    if args.greedy_mask is not None:
        ok &= ~args.greedy_mask
    q = torch.where(ok.view(-1, 1, 1), q, torch.zeros_like(q))
    d = drafts.unsqueeze(-1)
    p_d = p[:, :k].gather(2, d).squeeze(-1)
    q_d = (q * (cand == d)).sum(dim=-1)
    accept = torch.rand_like(p_d) * q_d < p_d
    emitted_q = accept.to(torch.int64).cumprod(dim=1).sum(dim=1)
    # the residual max(0, p - q) at every draft slot; the bonus row is p itself
    resid = p.clone()
    resid[:, :k].scatter_(2, cand, (p[:, :k].gather(2, cand) - q).clamp_(min=0.0))
    rows = torch.arange(bs, device=p.device)
    pick = resid[rows, emitted_q]
    total = pick.sum(dim=-1, keepdim=True)
    pick = torch.where(total > 0, pick / total.clamp_min(1e-30), p[rows, emitted_q])
    recovered = sampling_module().sampling_from_probs(pick).to(torch.int64)
    out_q = torch.cat([drafts, drafts[:, -1:]], dim=1)
    out_q.scatter_(1, emitted_q.view(-1, 1), recovered.view(-1, 1))
    return torch.where(ok.view(-1, 1), out_q, out), torch.where(ok, emitted_q, emitted)
