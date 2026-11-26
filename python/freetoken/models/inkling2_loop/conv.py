"""Inkling 2 short causal convolutions over a per-request window of the last ``K - 1`` inputs.

Each cache layer convolves four streams (key, value, attention output, MLP output); their windows live in
the linear-state pool as the ``CONV_STATES`` slot states, one row per cache layer. Prefill packs every
request's window in front of its rows, decode appends its rows to the window. A verify forward reads the
live window only and stages ``window + rows`` for ``ConvWindowVerify.commit``, as the GDN verify does.
The convolution inputs are bf16 activations, so a bf16 window is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

from freetoken.core import Batch, get_global_ctx

from .config import CONV_STATES


@dataclass
class ConvMetadata:
    """Indices every convolution of one forward shares."""

    slots: torch.Tensor  # [B] int64 state slot of each request
    kernel: int
    # prefill: the packed history holds each request's window, then its rows
    x_index: torch.Tensor | None = None  # [T] history row of each input row
    state_index: torch.Tensor | None = None  # [B * (K-1)] history rows the windows fill
    next_index: torch.Tensor | None = None  # [B * (K-1)] history rows that become the windows
    rows_per_req: int = 1
    verify: bool = False

    @property
    def prefill(self) -> bool:
        return self.x_index is not None


def prefill_indices(lens: List[int], kernel: int, device: torch.device):
    """``(x_index, state_index, next_index)`` of the packed history for requests of ``lens`` rows."""
    km1 = kernel - 1
    counts = torch.tensor(lens, dtype=torch.int64)
    cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    pad = torch.arange(len(lens), dtype=torch.int64) * km1
    base = cu[:-1] + pad
    x_index = torch.arange(int(cu[-1])) + torch.repeat_interleave(pad + km1, counts)
    span = torch.arange(km1, dtype=torch.int64)
    state_index = (base.unsqueeze(1) + span).reshape(-1)
    next_index = ((base + counts).unsqueeze(1) + span).reshape(-1)
    packed = torch.cat([x_index, state_index, next_index])
    if torch.cuda.is_available() and device.type == "cuda":
        packed = packed.pin_memory()
    packed = packed.to(device, non_blocking=True)
    n_x, n_s = x_index.numel(), state_index.numel()
    return packed[:n_x], packed[n_x : n_x + n_s], packed[n_x + n_s :]


def build_conv_metadata(batch: Batch, kernel: int, device: torch.device) -> ConvMetadata:
    """From ``batch.fla_metadata`` (state slots, fresh windows); zeroes the windows of fresh requests."""
    ctx = get_global_ctx()
    pool = ctx.linear_state_pool
    assert pool is not None, "Inkling 2 keeps its convolution windows in the linear-state pool"
    fla = batch.fla_metadata
    if fla is None:
        from freetoken.attention.linear import build_fla_metadata

        fla = batch.fla_metadata = build_fla_metadata(batch, device, pool)
    slots = fla.cache_indices.long()
    if batch.is_prefill:
        if fla.fresh_state_indices is not None:
            for name in CONV_STATES:
                pool.slot_states[name][:, fla.fresh_state_indices] = 0
        x_index, state_index, next_index = prefill_indices([r.extend_len for r in batch.padded_reqs], kernel, device)
        return ConvMetadata(slots, kernel, x_index, state_index, next_index)
    return ConvMetadata(slots, kernel, rows_per_req=int(batch.rows_per_req or 1), verify=batch.phase == "verify")


def _windowed_sum(lagged, weight: torch.Tensor, dynamic: torch.Tensor | None, head_dim: int) -> torch.Tensor:
    """``sum_lag (weight[lag] + dynamic[..., lag]) * lagged(lag)``, fp32; ``dynamic`` [T, heads, K] repeats over each head's channels."""
    acc = None
    for lag in range(weight.shape[0]):
        w = weight[lag]
        if dynamic is not None:
            w = w + dynamic[:, :, lag].repeat_interleave(head_dim, dim=1)
        term = lagged(lag).float() * w
        acc = term if acc is None else acc + term
    return acc


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    meta: ConvMetadata,
    *,
    dynamic: torch.Tensor | None = None,
    head_dim: int = 1,
    verify_stash: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(x + conv(window ++ x)) / sqrt(K + 1)`` (``sqrt(2K + 1)`` with ``dynamic`` taps) in x's dtype.

    ``weight`` [K, C] fp32, tap ``lag`` weighting the input ``lag`` steps back; ``state`` [slots, K-1, C], the
    window of each slot, oldest first, advanced in place (prefill, decode) or staged in ``verify_stash``
    [max_bs, K-1+R, C] (verify)."""
    T, C = x.shape
    K = meta.kernel
    km1 = K - 1
    if meta.prefill:
        hist = x.new_empty(T + meta.slots.numel() * km1, C)
        hist.index_copy_(0, meta.state_index, state.index_select(0, meta.slots).reshape(-1, C).to(x.dtype))
        hist.index_copy_(0, meta.x_index, x)
        state.index_copy_(0, meta.slots, hist.index_select(0, meta.next_index).view(-1, km1, C).to(state.dtype))
        acc = _windowed_sum(lambda lag: hist.index_select(0, meta.x_index - lag), weight, dynamic, head_dim)
    else:
        R = meta.rows_per_req
        B = T // R
        hist = torch.cat([state.index_select(0, meta.slots).to(x.dtype), x.view(B, R, C)], dim=1)
        if meta.verify:
            assert verify_stash is not None, "a verify forward stages its windows for the commit"
            verify_stash[:B].copy_(hist)
        else:
            state.index_copy_(0, meta.slots, hist[:, R:].to(state.dtype))
        acc = _windowed_sum(lambda lag: hist[:, km1 - lag : km1 - lag + R].reshape(T, C), weight, dynamic, head_dim)
    scale = 1.0 / math.sqrt((2 * K + 1) if dynamic is not None else (K + 1))
    return ((acc + x.float()) * scale).to(x.dtype)


class ConvWindowVerify:
    """``SlotVerify`` of one convolution stream: per cache layer and verify row, the live window then every verify input."""

    def __init__(self, spec, max_bs: int, rows_per_req: int, device: torch.device, dtype: torch.dtype) -> None:
        self.name = spec.name
        self.km1, width = spec.shape
        self.history = {
            layer_id: torch.zeros(max_bs, self.km1 + rows_per_req, width, dtype=dtype, device=device)
            for layer_id in spec.layer_ids
        }

    @staticmethod
    def nbytes(spec, max_bs: int, rows_per_req: int, dtype: torch.dtype) -> int:
        km1, width = spec.shape
        return len(spec.layer_ids) * max_bs * (km1 + rows_per_req) * width * dtype.itemsize

    def stash(self, layer_id: int) -> torch.Tensor:
        return self.history[layer_id]

    def commit(self, pool, slots: torch.Tensor, counts: torch.Tensor) -> None:
        # after n accepted rows the window is history rows n .. n + K - 2
        rows = (counts.unsqueeze(1) + torch.arange(self.km1, device=counts.device)).unsqueeze(2)
        for layer_id, history in self.history.items():
            states = pool.slot_state(self.name, layer_id)
            window = history[: slots.numel()].gather(1, rows.expand(-1, -1, history.shape[2]))
            states.index_copy_(0, slots, window.to(states.dtype))


__all__ = ["ConvMetadata", "ConvWindowVerify", "build_conv_metadata", "prefill_indices", "short_conv"]
