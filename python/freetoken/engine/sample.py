from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class LogitsPlan:
    """Inputs of the logits processors for the rows of one batch that asked for any
    (presence / frequency / repetition penalties, logit_bias, min_tokens, min_p).

    Built on the host in ``Sampler.prepare`` from the requests' SamplingParams and
    token histories, applied on the device by ``apply_logits_processors`` before the
    sampling kernel. ``None`` when no request in the batch uses a processor, so the
    default path stays exactly as it was. Row indices inside the plan are LOCAL (0..r-1,
    in the order of ``rows``); ``rows`` maps them back to the batch.
    """

    rows: torch.Tensor  # [r] int64 batch rows
    temps: torch.Tensor  # [r, 1] float32 temperatures (min_p works on probabilities)
    presence: torch.Tensor | None = None  # [r, 1] float32
    frequency: torch.Tensor | None = None  # [r, 1] float32
    repetition: torch.Tensor | None = None  # [r, 1] float32 (1.0 = off)
    # Generated ids per row, right-padded with vocab_size (a scratch column that is
    # dropped), for the presence / frequency counts and the repetition mask.
    hist_ids: torch.Tensor | None = None  # [r, L] int64
    # Prompt ids per row, same padding: repetition_penalty also covers the prompt.
    prompt_ids: torch.Tensor | None = None  # [r, P] int64
    bias_rows: torch.Tensor | None = None  # [b] int64 local rows
    bias_ids: torch.Tensor | None = None  # [b] int64
    bias_vals: torch.Tensor | None = None  # [b] float32
    # Rows still under min_tokens and the ids (EOS + stop_token_ids, padded) they must
    # not sample yet.
    min_rows: torch.Tensor | None = None  # [m] int64 local rows
    min_ids: torch.Tensor | None = None  # [m, S] int64
    min_p: torch.Tensor | None = None  # [r, 1] float32 (0 = off for that row)


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    plan: LogitsPlan | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":  # the unit tests build plans on the CPU
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _padded_rows(rows: list[list[int]], pad: int, device: torch.device) -> torch.Tensor:
    """[len(rows), max_len] int64 with ``pad`` on the right. A synchronous copy: the
    histories can be long (prompt + output) and a pinned buffer must not be freed under
    an in-flight non_blocking copy."""
    width = max(1, max((len(r) for r in rows), default=1))
    out = torch.full((len(rows), width), pad, dtype=torch.int64)
    for i, r in enumerate(rows):
        if r:
            out[i, : len(r)] = torch.as_tensor(r, dtype=torch.int64)
    return out.to(device)


def apply_logits_processors(
    logits: torch.Tensor, plan: LogitsPlan, vocab_size: int
) -> torch.Tensor:
    """A float32 copy of ``logits`` with the plan applied to its rows, in vLLM's order:
    repetition -> presence -> frequency penalties, logit_bias, the min_tokens mask, then
    min_p. Rows outside the plan are copied unchanged."""
    out = logits.to(torch.float32, copy=True)
    sub = out[plan.rows]  # [r, V] (advanced indexing copies)
    r = sub.shape[0]
    device = sub.device
    neg_inf = float("-inf")

    if plan.hist_ids is not None:
        counts = torch.zeros((r, vocab_size + 1), dtype=torch.float32, device=device)
        counts.scatter_add_(1, plan.hist_ids, torch.ones_like(plan.hist_ids, dtype=torch.float32))
        counts = counts[:, :vocab_size]
        if plan.repetition is not None:
            seen = counts > 0
            if plan.prompt_ids is not None:
                in_prompt = torch.zeros((r, vocab_size + 1), dtype=torch.bool, device=device)
                in_prompt.scatter_(1, plan.prompt_ids, torch.ones_like(plan.prompt_ids, dtype=torch.bool))
                seen = seen | in_prompt[:, :vocab_size]
            # HF / vLLM semantics: divide positive logits, multiply negative ones.
            penalized = torch.where(sub > 0, sub / plan.repetition, sub * plan.repetition)
            sub = torch.where(seen, penalized, sub)
        if plan.presence is not None:
            sub = sub - plan.presence * (counts > 0).to(sub.dtype)
        if plan.frequency is not None:
            sub = sub - plan.frequency * counts

    if plan.bias_ids is not None:
        sub.index_put_((plan.bias_rows, plan.bias_ids), plan.bias_vals, accumulate=True)

    if plan.min_ids is not None:
        block = torch.zeros((plan.min_ids.shape[0], vocab_size + 1), dtype=torch.bool, device=device)
        block.scatter_(1, plan.min_ids, torch.ones_like(plan.min_ids, dtype=torch.bool))
        rows = sub[plan.min_rows]
        sub[plan.min_rows] = torch.where(block[:, :vocab_size], torch.full_like(rows, neg_inf), rows)

    if plan.min_p is not None:
        probs = torch.softmax(sub / plan.temps, dim=-1)
        keep = probs >= plan.min_p * probs.amax(dim=-1, keepdim=True)
        keep = keep | (plan.min_p <= 0.0)  # rows without min_p stay untouched
        sub = torch.where(keep, sub, torch.full_like(sub, neg_inf))

    out[plan.rows] = sub
    return out


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def _plan(self, batch: Batch, params) -> LogitsPlan | None:
        """The LogitsPlan for this batch, or None when no request asks for a processor.
        Token histories come from the host-side ``req.input_ids``; under overlap
        scheduling the previous step's token may not be appended yet, so a penalty sees
        the history one token late. That is the accepted cost of keeping the counts off
        the device."""
        MIN_T = 1e-6
        rows = [i for i, p in enumerate(params) if p.needs_logits_processing]
        if not rows:
            return None
        picked = [(batch.reqs[i], params[i]) for i in rows]
        temps = [[max(0.0 if p.is_greedy else p.temperature, MIN_T)] for _, p in picked]
        plan = LogitsPlan(
            rows=make_device_tensor(rows, torch.int64, self.device),
            temps=make_device_tensor(temps, torch.float32, self.device),
        )
        pad = self.vocab_size

        want_hist = any(
            p.presence_penalty != 0.0 or p.frequency_penalty != 0.0 or p.repetition_penalty != 1.0
            for _, p in picked
        )
        if want_hist:
            hist: list[list[int]] = []
            prompts: list[list[int]] = []
            want_prompt = False
            for req, p in picked:
                prompt_len = req.max_device_len - req.output_len
                ids = req.input_ids
                penalized = p.presence_penalty != 0.0 or p.frequency_penalty != 0.0 or p.repetition_penalty != 1.0
                hist.append(ids[prompt_len:].tolist() if penalized else [])
                if p.repetition_penalty != 1.0:
                    want_prompt = True
                    prompts.append(ids[:prompt_len].tolist())
                else:
                    prompts.append([])
            plan.hist_ids = _padded_rows(hist, pad, self.device)
            if want_prompt:
                plan.prompt_ids = _padded_rows(prompts, pad, self.device)
            if any(p.presence_penalty != 0.0 for _, p in picked):
                plan.presence = make_device_tensor([[p.presence_penalty] for _, p in picked], torch.float32, self.device)
            if any(p.frequency_penalty != 0.0 for _, p in picked):
                plan.frequency = make_device_tensor([[p.frequency_penalty] for _, p in picked], torch.float32, self.device)
            if any(p.repetition_penalty != 1.0 for _, p in picked):
                plan.repetition = make_device_tensor([[p.repetition_penalty] for _, p in picked], torch.float32, self.device)

        bias_rows: list[int] = []
        bias_ids: list[int] = []
        bias_vals: list[float] = []
        for local, (_, p) in enumerate(picked):
            for tid, val in (p.logit_bias or {}).items():
                if 0 <= tid < self.vocab_size:
                    bias_rows.append(local)
                    bias_ids.append(int(tid))
                    bias_vals.append(float(val))
        if bias_ids:
            plan.bias_rows = make_device_tensor(bias_rows, torch.int64, self.device)
            plan.bias_ids = make_device_tensor(bias_ids, torch.int64, self.device)
            plan.bias_vals = make_device_tensor(bias_vals, torch.float32, self.device)

        min_rows: list[int] = []
        min_ids: list[list[int]] = []
        for local, (req, p) in enumerate(picked):
            if p.min_tokens > 0 and p.min_tokens_stop_ids:
                generated = len(req.input_ids) - (req.max_device_len - req.output_len)
                if generated < p.min_tokens:
                    min_rows.append(local)
                    min_ids.append([t for t in p.min_tokens_stop_ids if 0 <= t < self.vocab_size])
        if min_rows:
            plan.min_rows = make_device_tensor(min_rows, torch.int64, self.device)
            plan.min_ids = _padded_rows(min_ids, pad, self.device)

        if any(p.min_p > 0.0 for _, p in picked):
            plan.min_p = make_device_tensor([[p.min_p] for _, p in picked], torch.float32, self.device)
        return plan

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        plan = self._plan(batch, params)
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None, plan=plan)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, plan=plan)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.plan is not None:
                logits = apply_logits_processors(logits, args.plan, self.vocab_size)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
