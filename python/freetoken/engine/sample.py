from __future__ import annotations

from dataclasses import dataclass
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
    apply_penalties: bool = False


def apply_penalties(logits: torch.Tensor, reqs: List["Req"]) -> None:
    """Apply presence/frequency penalties in-place, using generated suffix only."""
    for row, req in enumerate(reqs):
        params = req.sampling_params
        presence, frequency = params.presence_penalty, params.frequency_penalty
        if presence == 0.0 and frequency == 0.0:
            continue
        generated = req.input_ids[req.prompt_len :]
        if generated.numel() == 0:
            continue
        token_ids, counts = torch.unique(generated, return_counts=True)
        values = torch.full_like(counts, presence, dtype=torch.float32)
        values.add_(frequency * counts.to(torch.float32))
        logits[row, token_ids.to(logits.device)] -= values.to(logits.device)


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
        penalties_enabled = any(
            p.presence_penalty != 0.0 or p.frequency_penalty != 0.0 for p in params
        )
        if all(p.is_greedy for p in params) and not penalties_enabled:
            return BatchSamplingArgs(temperatures=None)

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
        return BatchSamplingArgs(
            temperatures,
            top_k=top_k,
            top_p=top_p,
            apply_penalties=penalties_enabled,
        )

    @nvtx_annotate("Sampler")
    def sample(
        self, logits: torch.Tensor, args: BatchSamplingArgs, batch: Batch
    ) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.apply_penalties:
                apply_penalties(logits, batch.reqs)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
