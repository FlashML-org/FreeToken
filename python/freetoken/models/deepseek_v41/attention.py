"""CSA2 attention, compressed-source sharing, and two-stage index selection."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv41.indexer import select_indices
from freetoken.kernel.triton.dsv41.quant import fp4_roundtrip, fp8_roundtrip, pack_fp4, pack_fp8
from .layers import Linear, RMSNorm


def rotary_frequencies(args, compressed, device):
    dim = args.rope_head_dim
    base = args.compress_rope_theta if compressed else args.rope_theta
    freq = 1.0 / base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    if compressed and args.original_seq_len > 0:
        def correction(rotations):
            return dim * math.log(args.original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(correction(args.beta_fast)), 0), min(math.ceil(correction(args.beta_slow)), dim - 1)
        if low == high:
            high += .001
        smooth = 1 - ((torch.arange(dim // 2, device=device) - low) / (high - low)).clamp(0, 1)
        freq = freq / args.rope_factor * (1 - smooth) + freq * smooth
    return freq


def apply_rope(x, positions, inv_freq, inverse=False):
    """Rotate only the tail; derive frequencies for active tokens instead of a 1M table."""
    dim = inv_freq.numel() * 2
    tail = torch.view_as_complex(x[..., -dim:].float().unflatten(-1, (-1, 2)))
    phase = positions.float()[:, None] * inv_freq[None, :]
    frequencies = torch.polar(torch.ones_like(phase), -phase if inverse else phase)
    frequencies = frequencies.view(positions.numel(), *([1] * (tail.ndim - 2)), -1)
    x[..., -dim:] = torch.view_as_real(tail * frequencies).flatten(-2).to(x.dtype)
    return x


class Compressor(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.ratio = args.compress_ratios[layer_id]
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wkv = Linear(args.dim, args.head_dim, kind="fp32" if self.ratio == 2 else "bf16")
        if self.ratio == 2:
            self.wgate = Linear(args.dim, args.head_dim, kind="fp32")

    def project(self, x):
        if self.ratio == 1:
            return self.norm(self.wkv(x)), None
        return self.wkv(x.float()), self.wgate(x.float())

    def pool(self, kv, scores, dtype):
        return self.norm((kv * scores.softmax(-2)).sum(-2).to(dtype))


class Indexer(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.n_heads, self.head_dim = args.index_n_heads, args.index_head_dim
        self.wq_b = Linear(args.q_lora_rank, self.n_heads * self.head_dim, kind="fp8")
        self.weights_proj = Linear(args.dim, self.n_heads, kind="bf16")
        self.scale = (self.head_dim * self.n_heads) ** -.5
        if layer_id in args.kv_source_layers:
            self.wk = Linear(args.head_dim, self.head_dim, kind="bf16")
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)

    def query(self, x, qr, positions, freq):
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = fp4_roundtrip(apply_rope(q, positions, freq), block_size=32, scale_format="e8m0")
        return q, self.weights_proj(x) * self.scale

    def keys(self, latent, positions, freq, *, packed=False):
        k = self.k_norm(self.wk(latent))
        quantize = pack_fp4 if packed else fp4_roundtrip
        return quantize(apply_rope(k, positions, freq), block_size=32, scale_format="e8m0")


class Attention(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.layer_id, self.args = layer_id, args
        self.dim, self.n_heads, self.head_dim = args.dim, args.n_heads, args.head_dim
        self.n_groups, self.o_lora_rank = args.o_groups, args.o_lora_rank
        self.ratio = self.compress_ratio = args.compress_ratios[layer_id]
        self.window_size = args.window_size
        self.is_kv_source = layer_id in args.kv_source_layers
        self.is_index_source = layer_id in args.index_source_layers
        self.wq_a = Linear(args.dim, args.q_lora_rank, kind="fp8")
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = Linear(args.q_lora_rank, args.n_heads * args.head_dim, kind="fp8")
        self.wkv = Linear(args.dim, args.head_dim, kind="fp8")
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wo_a = nn.Parameter(torch.empty(args.o_groups * args.o_lora_rank,
                                            args.n_heads * args.head_dim // args.o_groups,
                                            dtype=torch.bfloat16), requires_grad=False)
        self.wo_b = Linear(args.o_groups * args.o_lora_rank, args.dim, kind="fp8")
        self.attn_sink = nn.Parameter(torch.empty(args.n_heads, dtype=torch.float32), requires_grad=False)
        self.softmax_scale = args.head_dim ** -.5
        self.compressor = Compressor(args, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(args, layer_id) if self.is_index_source else None
        self.inv_freq = None
        self.fp8_fp4 = False

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, pool, device):
        self.inv_freq = rotary_frequencies(self.args, bool(self.ratio), device)
        self.fp8_fp4 = getattr(pool, "kv_quant", "none") == "fp8-fp4"

    def reset(self):
        pass

    def _project(self, x, positions):
        qr = self.q_norm(self.wq_a(x))
        q = apply_rope(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), positions, self.inv_freq)
        kv = apply_rope(self.kv_norm(self.wkv(x)), positions, self.inv_freq)
        quantize = pack_fp8 if self.fp8_fp4 else fp8_roundtrip
        return qr, q, quantize(kv, block_size=32)

    def _output(self, o, positions):
        o = apply_rope(o, positions, self.inv_freq, inverse=True).reshape(-1, self.n_groups,
                                                                        self.n_heads * self.head_dim // self.n_groups)
        projected = torch.einsum("tgd,grd->tgr", o, self.wo_a.view(self.n_groups, self.o_lora_rank, -1))
        return self.wo_b(projected.flatten(1))

    def _prefill_compress(self, x, ti, start, slots):
        compressor, pool, ratio = self.compressor, self.attn.pool, self.ratio
        kv, score = compressor.project(x)
        end = start + x.shape[0]
        if ratio == 1:
            latent = kv
        else:
            if start % ratio:
                ws = self.attn.window_slots_of(ti, start - 1, start)
                state = pool.get_state(self.layer_id, pool.state_loc(ws, 2, pool.P))
                old_kv, old_score = state.split(self.head_dim, -1)
                kv, score = torch.cat((old_kv, kv), 0), torch.cat((old_score, score), 0)
            complete = kv.shape[0] // ratio * ratio
            latent = (compressor.pool(kv[:complete].unflatten(0, (-1, ratio)),
                                      score[:complete].unflatten(0, (-1, ratio)), x.dtype)
                      if complete else x.new_empty((0, self.head_dim)))
            if end % ratio:
                state_loc = pool.state_loc(slots[-1:], 2, pool.P)
                pool.set_state(self.layer_id, state_loc, torch.cat((kv[-1:], score[-1:]), -1))
        group_positions = torch.arange(start // ratio * ratio, end // ratio * ratio, ratio, device=x.device)
        rows = self.attn.compress_rows_of(ti, group_positions, ratio)
        self._publish(latent, group_positions, rows, rows)

    def _publish(self, latent, positions, cmp_rows, idx_rows):
        if not latent.shape[0]:
            return
        keys = self.indexer.keys(latent, positions, self.inv_freq, packed=self.fp8_fp4)
        self.attn.scatter_compressed(self.layer_id, "idx", idx_rows, keys)
        quantize = pack_fp4 if self.fp8_fp4 else fp4_roundtrip
        compressed = quantize(apply_rope(latent, positions, self.inv_freq),
                              block_size=16, scale_format="e4m3")
        self.attn.scatter_compressed(self.layer_id, "attn", cmp_rows, compressed)

    def _select(self, x, qr, positions, mapping, key, width):
        if not self.is_index_source:
            return self.attn.shared_indices[key]
        args, backend = self.args, self.attn
        q, weights = self.indexer.query(x, qr, positions, self.inv_freq)
        source = backend.pool.kv_sources[self.layer_id]
        candidates = backend.shared_candidates[key] if self.layer_id > args.candidate_source_layer >= 0 else None
        publish = self.layer_id == args.candidate_source_layer
        indices, blocks = select_indices(
            q, weights, backend.pool.idx_pool[source], mapping,
            (positions + 1) // self.ratio, width, self.ratio, args.index_topk,
            candidates=candidates, candidate_topk=args.candidate_topk_blocks if publish else 0,
            block_size=args.candidate_block_size or 8,
        )
        backend.shared_indices[key] = indices
        if publish:
            backend.shared_candidates[key] = blocks
        return indices

    def forward_ragged(self, x, segments, flat_positions):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        backend = self.attn
        if self.layer_id == 0:
            backend.begin_forward()
        qr, q, kv = self._project(x, flat_positions)
        windows, compressed = [], []
        for offset, length, ti, start in segments:
            end = start + length
            pos = flat_positions[offset:offset + length]
            slots = backend.window_slots_of(ti, start, end)
            backend.store_window(kv[offset:offset + length], self.layer_id, slots)
            win_positions = pos[:, None] - self.window_size + 1 + torch.arange(self.window_size, device=x.device)
            locs = backend.pool.full_loc_map[ti, win_positions.clamp_min(0)]
            win = backend.pool.translate_full_to_window(locs)
            windows.append(torch.where(win_positions >= 0, win, -1))
            if self.ratio:
                sx, sqr = x[offset:offset + length], qr[offset:offset + length]
                if self.is_kv_source:
                    self._prefill_compress(sx, ti, start, slots)
                key = (offset, length, ti, start)
                mapping = backend.pool.full_loc_map[ti:ti + 1].expand(length, -1)
                picks = self._select(sx, sqr, pos, mapping, key, end // self.ratio)
                compressed.append(backend.blocks_to_global(picks[None], self.ratio, ti=ti)[0])
        win = torch.cat(windows, 0)
        if self.ratio:
            width = max(part.shape[-1] for part in compressed)
            cmp = torch.cat([F.pad(part, (0, width - part.shape[-1]), value=-1) for part in compressed], 0)
            picks = torch.cat((win, cmp), -1)
        else:
            picks = win
        o = backend.attend(q[None], self.layer_id, picks[None], self.window_size,
                           self.attn_sink, self.softmax_scale, has_compression=bool(self.ratio))[0]
        return self._output(o, flat_positions).view(shape)

    def decode_step(self, x, pos, rows, cmp_stage_cap, wctx=None):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        backend, pool = self.attn, self.attn.pool
        if self.layer_id == 0:
            backend.begin_forward()
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        slots, previous_slots, window_picks = wctx
        qr, q, kv = self._project(x, pos)
        backend.store_window(kv, self.layer_id, slots)
        picks = window_picks
        if self.ratio:
            if self.is_kv_source:
                latent, score = self.compressor.project(x)
                if self.ratio == 2:
                    old = pool.get_state(self.layer_id, pool.state_loc(previous_slots, 2, pool.P))
                    previous_kv, previous_score = old.split(self.head_dim, -1)
                    pooled = self.compressor.pool(torch.stack((previous_kv, latent), 1),
                                                  torch.stack((previous_score, score), 1), x.dtype)
                    pool.set_state(self.layer_id, pool.state_loc(slots, 2, pool.P), torch.cat((latent, score), -1))
                    latent = pooled
                complete = (pos + 1) % self.ratio == 0
                cmp_rows = backend.decode_compress_rows(rows, pos, self.ratio, self.layer_id, "attn", complete)
                idx_rows = backend.decode_compress_rows(rows, pos, self.ratio, self.layer_id, "idx", complete)
                self._publish(latent, pos // self.ratio * self.ratio, cmp_rows, idx_rows)
            indices = self._select(x, qr, pos, backend.snapshot()[rows], "decode", (cmp_stage_cap + 1) // self.ratio)
            cmp = backend.blocks_to_global(indices[:, None], self.ratio, rows=rows)
            picks = torch.cat((window_picks, cmp), -1)
        o = backend.attend(q[:, None], self.layer_id, picks, self.window_size, self.attn_sink,
                           self.softmax_scale, has_compression=bool(self.ratio))[:, 0]
        return self._output(o, pos).view(shape)


DeepseekV41Attention = Attention
