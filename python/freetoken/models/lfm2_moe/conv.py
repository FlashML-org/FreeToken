"""Depthwise causal short conv (LFM2-MoE ``conv`` layers).

HF reference (``Lfm2MoeShortConv``): ``B, C, x = in_proj(h).chunk(3)`` -- always in that
order -- ``Bx = B * x``, a depthwise causal conv of kernel ``conv_L_cache``/left padding
``L_cache - 1`` over the ``Bx`` stream, ``y = C * conv(y)``, then ``out_proj``. No activation
anywhere (HF passes ``activation=None``), so the fused sgl ``causal_conv1d_*`` wrappers (which
silu unconditionally) do not apply: the conv is plain torch, like the qwen4_exp PLE conv.

The conv writes ``Bx``'s last ``L_cache - 1`` timesteps into the layer's ``conv_states`` pool
slab, so a continuation resumes from its left context and a fresh sequence starts from zeros.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.attention.linear import FLAMetadata
    from freetoken.models.config import ModelConfig


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise weight ``[conv_dim, 1, K]`` (checkpoint key ``conv.conv.weight``)."""

    def __init__(self, conv_dim: int, kernel: int) -> None:
        self.weight = torch.empty(conv_dim, 1, kernel)


def _packed_layout(batch, fla: "FLAMetadata", state_len: int):
    """Column indices of the packed prefill conv: one block per request,
    ``[state (state_len) | its contiguous chunk]``, returned as
    ``(out_index, state_index, next_state_index)``.

    The layout is a pure function of the forward's metadata and is identical for every conv
    layer (all state layers share the per-request slot), so it is derived once per forward
    instead of once per conv layer -- the derivation needs one ``cu_seqlens`` host sync. The
    cached copy is tagged with the metadata object it came from, so a reused batch (or a
    direct-op caller with a stale batch) can never read another forward's layout.
    """
    cached = getattr(batch, "_lfm2_conv_layout", None)
    if cached is not None and cached[0] is fla:
        return cached[1]

    lens = torch.diff(fla.cu_seqlens).tolist()
    counts = torch.tensor(lens, dtype=torch.int64)
    cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    pad = torch.arange(len(lens), dtype=torch.int64) * state_len
    base = cu[:-1] + pad
    out_index = torch.arange(int(cu[-1])) + torch.repeat_interleave(pad, counts)
    span = torch.arange(state_len, dtype=torch.int64)
    packed = torch.cat(
        [
            out_index,
            (base.unsqueeze(1) + span).reshape(-1),
            ((base + counts).unsqueeze(1) + span).reshape(-1),
        ]
    )
    if torch.cuda.is_available():
        packed = packed.pin_memory()
    packed = packed.to(fla.cu_seqlens.device, non_blocking=True)
    n_out, n_state = out_index.numel(), len(lens) * state_len
    layout = packed[:n_out], packed[n_out : n_out + n_state], packed[n_out + n_state :]
    setattr(batch, "_lfm2_conv_layout", (fla, layout))
    return layout


class Lfm2ShortConv(BaseOP):
    """``C * conv(B * x)`` over the depthwise kernel, with the ``Bx`` history in the state pool.

    Every projection is replicated (the TP=1 correctness milestone, see qwen3_5_moe): the
    reader refuses TP > 1, so a column/row-parallel declaration would misdescribe the module.
    """

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "") -> None:
        group = config.short_conv_group()
        assert group is not None, "Lfm2ShortConv needs a ShortConvGroupConfig"
        self._layer_id = layer_id
        self.conv_dim = config.hidden_size
        self.kernel = group.conv_kernel_dim
        self.state_len = group.conv_kernel_dim - 1
        self.in_proj = LinearReplicated(
            config.hidden_size, 3 * config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.in_proj",
        )
        self.conv = _DepthwiseConv1d(config.hidden_size, group.conv_kernel_dim)
        self.out_proj = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.out_proj",
        )
        # every released checkpoint sets conv_bias false; None keeps the key out of the state dict
        self.bias = torch.empty(config.hidden_size) if group.conv_bias else None

    @nvtx_annotate("Lfm2ShortConv")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        assert pool is not None, "LFM2 short conv needs ctx.linear_state_pool"
        li = pool.local_index(self._layer_id)
        conv_states = pool.conv_states[li]  # [num_slots, conv_dim, state_len]
        proj = self.in_proj.forward(x)
        b, c, xv = proj.split(self.conv_dim, dim=-1)
        bx = b * xv  # [T, conv_dim]: the conv input stream (and the state)

        # per-forward GDN metadata (built once for all state layers; the scheduler/graph sets it)
        fla = batch.fla_metadata
        if fla is None:  # direct-op callers (tests)
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, x.device)
            batch.fla_metadata = fla

        weight = self.conv.weight.squeeze(1).float()  # [conv_dim, K]
        # index_select/index_copy_ take int64; the metadata carries the slots as int32
        slots = fla.cache_indices.to(torch.int64)
        if batch.is_decode:
            out = self._decode_conv(bx, slots, conv_states, weight)
        else:
            layout = _packed_layout(batch, fla, self.state_len)
            out = self._prefill_conv(bx, slots, fla, conv_states, weight, layout)
        if self.bias is not None:
            out = out + self.bias
        return self.out_proj.forward(c * out)

    def _decode_conv(
        self,
        x: torch.Tensor,
        slots: torch.Tensor,
        conv_states: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """One token per padded request: ``[state_left | x]`` taps the full kernel, then the
        state rolls by one. fp32 like the prefill conv, so both paths agree exactly."""
        state = conv_states.index_select(0, slots).float()
        window = torch.cat([state, x.unsqueeze(-1)], dim=-1)
        out = (window * weight).sum(-1).to(x.dtype)
        conv_states.index_copy_(
            0, slots, torch.cat([state[..., 1:], x.unsqueeze(-1)], dim=-1).to(conv_states.dtype)
        )
        return out

    def _prefill_conv(
        self,
        bx: torch.Tensor,
        slots: torch.Tensor,
        fla: "FLAMetadata",
        conv_states: torch.Tensor,
        weight: torch.Tensor,
        layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """One conv over the packed blocks ``[state_0 | chunk_0 | state_1 | chunk_1 | ...]``.

        The blocks abut exactly, so every output window stays inside its own request: request
        i's first token reads history columns base_i .. base_i + state_len, which is its own
        state. A fresh request (cached_len == 0) must start from a zeroed slot: the pool can
        hand back a recycled one.
        """
        if fla.fresh_state_indices is not None:
            conv_states.index_fill_(0, fla.fresh_state_indices, 0.0)
        if getattr(fla, "track_dst", None) is not None:
            # Snapshot the state after exactly the track boundary's tokens (hybrid-radix
            # prefix reuse): the last state_len conv inputs ending at the boundary. The track
            # slots never alias the live slots this forward advances, so this write and the
            # state write below are order-independent.
            win = bx[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
            conv_states.index_copy_(0, fla.track_dst, win.to(conv_states.dtype))

        out_index, state_index, next_state_index = layout
        num_reqs = state_index.numel() // self.state_len
        state = conv_states.index_select(0, slots)

        history = torch.empty(
            self.conv_dim,
            bx.shape[0] + num_reqs * self.state_len,
            dtype=torch.float32,
            device=bx.device,
        )
        history.index_copy_(1, state_index, state.permute(1, 0, 2).reshape(self.conv_dim, -1).float())
        history.index_copy_(1, out_index + self.state_len, bx.transpose(0, 1).float())
        out = F.conv1d(history.unsqueeze(0), weight.unsqueeze(1), groups=self.conv_dim).squeeze(0)
        new_state = history.index_select(1, next_state_index).view(
            self.conv_dim, num_reqs, self.state_len
        )
        conv_states.index_copy_(
            0, slots, new_state.permute(1, 0, 2).to(conv_states.dtype).contiguous()
        )
        return out.index_select(1, out_index).transpose(0, 1).to(bx.dtype)


__all__ = ["Lfm2ShortConv"]
