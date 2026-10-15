from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None
    penalties: list[tuple[int, torch.Tensor, float, float]] = field(default_factory=list)


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


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

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        is_greedy = [p.is_greedy for p in params]
        penalties = []
        for row, req in enumerate(batch.reqs):
            p = req.sampling_params
            if not (p.presence_penalty or p.frequency_penalty) or not req.can_decode:
                continue
            if req.output_token_counts is None:
                req.output_token_counts = torch.zeros(
                    self.vocab_size, dtype=torch.int32, device=self.device
                )
            penalties.append(
                (row, req.output_token_counts, p.presence_penalty, p.frequency_penalty)
            )
        if all(is_greedy):
            return BatchSamplingArgs(temperatures=None, penalties=penalties)

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)]
        top_ks = [
            p.top_k if not g and p.top_k >= 1 else self.vocab_size
            for p, g in zip(params, is_greedy)
        ]
        top_ps = [
            1.0 if g else min(max(p.top_p, MIN_P), 1.0)
            for p, g in zip(params, is_greedy)
        ]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        greedy_mask = (
            make_device_tensor(is_greedy, torch.bool, self.device) if any(is_greedy) else None
        )
        return BatchSamplingArgs(
            temperatures, top_k=top_k, top_p=top_p, greedy_mask=greedy_mask, penalties=penalties
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.penalties:
                logits = logits.float().clone()
                for row, counts, presence, frequency in args.penalties:
                    logits[row] -= frequency * counts + presence * (counts > 0)
            if args.temperatures is None:  # greedy sampling
                tokens = torch.argmax(logits, dim=-1)
            else:
                tokens = sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
                if args.greedy_mask is not None:
                    # Mixed batches still run probability sampling for all rows, but
                    # greedy rows must follow argmax's deterministic tie-breaking.
                    greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
                    tokens = torch.where(args.greedy_mask, greedy_tokens, tokens)
            # Update on the sampling stream: overlapped scheduling can prepare the next
            # batch before the previous token reaches Req.input_ids on the CPU.
            for row, counts, _, _ in args.penalties:
                counts.scatter_add_(
                    0, tokens[row : row + 1].long(), counts.new_ones(1)
                )
            return tokens
