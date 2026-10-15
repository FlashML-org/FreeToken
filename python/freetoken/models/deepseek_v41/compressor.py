"""DSV41 KV compressor: ``compress_ratio`` consecutive tokens -> one latent (reference ``Compressor``).

* ratio 1 (the CED decoder source): a plain bf16 projection + RMSNorm, one latent per token.
* ratio > 1 (the encoder sources): ``wkv`` / ``wgate`` promoted to fp32, a softmax over the group's
  gate scores pools the group's ``wkv`` rows; the trailing partial group is carried across chunks
  and decode steps in the pool's compress-state ring (per window page, so a page-aligned radix hit
  resumes it by value -- see ``DSV41SparseAttnBackend`` for the addressing).

Returns the latent BEFORE RoPE: the indexer key is projected from the unrotated latent, the
attention layer rotates and quantizes it afterwards.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from .args import DeepseekV41Args


class Compressor(BaseOP):
    def __init__(self, args: DeepseekV41Args, layer_id: int, ratio: int, *, quant_config=None, prefix: str = ""):
        assert ratio >= 1
        self.layer_id = layer_id
        self.ratio = ratio
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.window = args.window_size
        if ratio > 1:
            # fp32 like the reference (the softmax pooling runs in fp32); the checkpoint ships bf16
            self.wkv = torch.empty(args.head_dim, args.dim, dtype=torch.float32)
            self.wgate = torch.empty(args.head_dim, args.dim, dtype=torch.float32)
        else:
            self.wkv = LinearReplicated(args.dim, args.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        self.norm = RMSNorm(args.head_dim, args.norm_eps)

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 ``wkv`` / ``wgate`` projections."""
        xf = x.float()
        return F.linear(xf, self.wkv), F.linear(xf, self.wgate)

    # ----- prefill -----------------------------------------------------------------------
    def needs_tail_carry(self, start_pos: int) -> bool:
        """Whether a prefill from ``start_pos`` resumes a partial group (never at a page-aligned start)."""
        return self.ratio > 1 and start_pos % self.ratio != 0

    def forward_prefill(
        self, x: torch.Tensor, start_pos: int, window_slots: torch.Tensor, tail_window_slot: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress the tokens ``[start_pos, start_pos + n)`` of one request.

        ``window_slots`` are the tokens' window slots (the ring block of each window page carries the
        partial group); ``tail_window_slot`` is the previous token's slot, needed only when
        ``needs_tail_carry(start_pos)``. Returns ``(latent [G, head_dim] bf16 pre-RoPE, group_starts [G]
        int64)`` for every group that completed in this call.
        """
        n = x.shape[0]
        device = x.device
        if self.ratio == 1:
            latent = self.norm.forward(self.wkv.forward(x))
            return latent, torch.arange(start_pos, start_pos + n, device=device)

        ratio, d = self.ratio, self.head_dim
        kv, score = self._project(x)
        # the partial group in progress at start_pos: its rows live in the previous token's ring block
        offset = start_pos % ratio
        if offset:
            assert tail_window_slot is not None, "resuming mid-group needs the previous token's window slot"
            carry = self.attn.read_carry(self.layer_id, tail_window_slot)  # [ratio, 2d]
            kv = torch.cat([carry[:offset, :d], kv], dim=0)
            score = torch.cat([carry[:offset, d:], score], dim=0)
        total = kv.shape[0]
        groups = total // ratio
        remainder = total % ratio
        cut = groups * ratio
        if groups:
            kv_g = kv[:cut].view(groups, ratio, d)
            score_g = score[:cut].view(groups, ratio, d)
            pooled = (kv_g * score_g.softmax(dim=1)).sum(dim=1)
            latent = self.norm.forward(pooled.to(x.dtype))
        else:
            latent = x.new_empty(0, d)
        group_starts = (start_pos - offset) + ratio * torch.arange(groups, device=device)
        # ring: the reset block at every page boundary crossed (a page holds whole groups), and the
        # in-progress tail (the remainder rows) on the last page so decode / the next chunk resume
        end = start_pos + n
        self.attn.write_boundary_carries(self.layer_id, lo=start_pos, hi=end, window_slots=window_slots)
        if end % self.window != 0:
            block = self._carry_block(kv[cut:], score[cut:], remainder)
            self.attn.write_carry(self.layer_id, int(window_slots[-1].item()), block)
        return latent, group_starts

    def _carry_block(self, kv_tail: torch.Tensor, score_tail: torch.Tensor, filled: int) -> torch.Tensor:
        ratio, d = self.ratio, self.head_dim
        ks = kv_tail.new_zeros(ratio, d)
        ss = kv_tail.new_full((ratio, d), float("-inf"))
        if filled:
            ks[:filled] = kv_tail
            ss[:filled] = score_tail
        return torch.cat([ks, ss], dim=-1)

    # ----- decode ------------------------------------------------------------------------
    def forward_decode(self, x: torch.Tensor, pos: torch.Tensor, prev_window_slots: torch.Tensor, window_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One token per row. Returns ``(latent [B, head_dim] bf16 pre-RoPE, completed [B] bool)``;
        a row whose group did not complete carries a meaningless latent the caller discards."""
        B = x.shape[0]
        if self.ratio == 1:
            latent = self.norm.forward(self.wkv.forward(x))
            return latent, torch.ones(B, dtype=torch.bool, device=x.device)

        from freetoken.kernel.triton.dsv4.compress import gated_pool

        ratio, d = self.ratio, self.head_dim
        kv, score = self._project(x)
        idx = (pos % ratio).view(B, 1, 1).expand(B, 1, d)
        block = self.attn.read_carry_blocks(self.layer_id, prev_window_slots)  # [B, ratio, 2d]
        ks = block[..., :d].clone().scatter_(1, idx, kv.unsqueeze(1))
        ss = block[..., d:].clone().scatter_(1, idx, score.unsqueeze(1))
        self.attn.write_carry_blocks(self.layer_id, window_slots, torch.cat([ks, ss], dim=-1))
        pooled = gated_pool(ks, ss, x.dtype).view(B, d)
        completed = (pos + 1) % ratio == 0
        return self.norm.forward(pooled), completed


__all__ = ["Compressor"]
