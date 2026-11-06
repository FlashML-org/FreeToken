"""DFlash 2 draft (block-diffusion drafter + candidate-path selector), after sglang ``models/dflash.py`` and ``speculative/dflash_worker_v2.py``.

One forward drafts the block ``[bonus, mask x k]`` at positions ``C .. C+k``; the target ``lm_head`` gives top-K candidates per slot and a low-rank selector walks one path through them.
The draft KV holds context rows (target features through ``fc`` and ``hidden_norm``) projected through each layer's K/V weights; the attention is a sliding window, so the KV is a per-request ring.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from freetoken.kernel.backend import is_flashinfer_installed
from freetoken.kernel.triton.dflash import grouped_conv, norm_rope_store, ring_attention, ring_attention_workspace, selector_walk
from freetoken.layers import LinearQKVMerged, LinearReplicated, LinearRowParallel, OPList, RMSNorm, RMSNormFused
from freetoken.layers.base import BaseOP
from freetoken.layers.quantization import NoQuantConfig, QuantConfig, finalize_quant
from freetoken.models.blocks import GatedMLP
from freetoken.utils import cached_load_hf_config, torch_dtype

from .config import DFlashDraftConfig
from .weight import iter_weights


@dataclass
class DFlashRing:
    """The draft K/V of one request per table row, position ``p`` at slot ``p % ring_len``, with the attention workspace and rope table for blocks of ``block`` tokens; built by ``DFlash2DraftModel.make_dflash_ring`` and owned by the proposer."""

    k: torch.Tensor  # [layers, rows, ring_len, kv_heads, head_dim]
    v: torch.Tensor
    block: int
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    cos_sin: torch.Tensor  # [max_position, head_dim] fp32


class DFlashGroupedConv(BaseOP):
    """Grouped dynamic depthwise 2-tap convolution along the block axis (sglang DFlashGroupedConv)."""

    def __init__(self, hidden: int, taps: int, group_size: int, *, quant_config: QuantConfig, prefix: str) -> None:
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden // group_size
        self.base_kernel = torch.empty(2, taps, hidden)
        self.kernel_projection = LinearReplicated(
            hidden, 2 * taps * self.num_groups, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.kernel_projection",
        )

    def prepare(self, x: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        coef = self.kernel_projection.forward(x).view(x.shape[0], 2, self.taps, self.num_groups)
        return grouped_conv(x, coef[:, 0], self.base_kernel[0], block, self.group_size), coef[:, 1]

    def finish(self, x: torch.Tensor, coef: torch.Tensor, block: int) -> torch.Tensor:
        return grouped_conv(x, coef, self.base_kernel[1], block, self.group_size)


class DFlashAttention(BaseOP):
    def __init__(self, config: DFlashDraftConfig, *, quant_config: QuantConfig, prefix: str) -> None:
        self.q_size = config.num_heads * config.head_dim
        self.qkv_proj = LinearQKVMerged(
            config.hidden_size, config.head_dim, config.num_heads, config.num_kv_heads, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = LinearRowParallel(self.q_size, config.hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)


class DFlashDecoderLayer(BaseOP):
    def __init__(self, config: DFlashDraftConfig, *, quant_config: QuantConfig, prefix: str) -> None:
        self.self_attn = DFlashAttention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = GatedMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        conv = dict(taps=config.conv_kernel_size, group_size=config.conv_group_size, quant_config=quant_config)
        self.attention_conv = DFlashGroupedConv(config.hidden_size, prefix=f"{prefix}.attention_conv", **conv)
        self.mlp_conv = DFlashGroupedConv(config.hidden_size, prefix=f"{prefix}.mlp_conv", **conv)


class DFlashCandidateSelector(BaseOP):
    def __init__(self, config: DFlashDraftConfig, *, quant_config: QuantConfig, prefix: str) -> None:
        self.hidden_projection = LinearReplicated(
            config.hidden_size, config.selector_rank, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.hidden_projection",
        )
        self.predecessor_codebook = torch.empty(config.vocab_size, config.selector_rank)
        self.successor_codebook = torch.empty(config.vocab_size, config.selector_rank)

    def sample_path(self, candidate_ids, unary_logits, hidden, anchor, temperature, greedy, uniforms) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk one path over the ``[bs, k, K]`` candidates: argmax on ``greedy`` rows, else drawn with ``uniforms`` from softmax(scores / temperature); returns the tokens ``[bs, k]`` and the q each was drawn from ``[bs, k, K]``."""
        return selector_walk(
            candidate_ids, unary_logits, self.hidden_projection.forward(hidden), anchor,
            self.predecessor_codebook, self.successor_codebook, temperature, greedy, uniforms,
        )


class DFlash2DraftModel(BaseOP):
    """DFlash 2 weights and compute; the proposer owns the ring, batching and the verify."""

    def __init__(self, config: DFlashDraftConfig, target_model, quant_config: QuantConfig) -> None:
        self.config = config
        hidden, taps = config.hidden_size, len(config.target_layer_ids)
        self.fc = LinearReplicated(taps * hidden, hidden, has_bias=False, quant_config=quant_config, prefix="fc")
        self.hidden_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.layers = OPList([DFlashDecoderLayer(config, quant_config=quant_config, prefix=f"layers.{i}") for i in range(config.num_layers)])
        self.norm = RMSNormFused(hidden, config.rms_norm_eps)
        self.candidate_selector = DFlashCandidateSelector(config, quant_config=quant_config, prefix="candidate_selector")
        # the target's own modules; underscore names keep them out of this module's state dict
        self._embed_tokens = target_model.model.embed_tokens
        self._lm_head = target_model.lm_head
        if is_flashinfer_installed():
            from flashinfer import top_k

            self._top_k = functools.partial(top_k, sorted=True, deterministic=True)
        else:
            self._top_k = functools.partial(torch.topk, dim=-1)

    @classmethod
    def load(cls, config: DFlashDraftConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> DFlash2DraftModel:
        # the draft's own quant config, not the target's: the context projection below reads raw weights
        quant = QuantConfig.from_hf(cached_load_hf_config(model_path))
        if not isinstance(quant, NoQuantConfig):
            raise ValueError(f"{model_path}: quantized DFlash drafts are not supported yet")
        with torch.device("meta"), torch_dtype(dtype):
            draft = cls(config, target_model, quant)
        expected = draft.state_dict()
        # all layers' q|k|v weights in one tensor: each layer uses its own slice, the context projection the K/V rows of every layer
        qkv = torch.empty(config.num_layers, *expected["layers.0.self_attn.qkv_proj.weight"].shape, dtype=dtype, device=device)
        qkv_slices = {f"layers.{i}.self_attn.qkv_proj.weight": w for i, w in enumerate(qkv)}
        loaded = {}
        for name, tensor in iter_weights(model_path, device):
            if name not in expected:
                raise ValueError(f"{model_path}: unexpected DFlash tensor {name!r}")
            if tensor.shape != expected[name].shape:
                raise ValueError(f"{model_path}: {name} is {tuple(tensor.shape)}, expected {tuple(expected[name].shape)}")
            dst = qkv_slices.get(name)
            loaded[name] = tensor.to(device=device, dtype=expected[name].dtype) if dst is None else dst.copy_(tensor)
        if missing := sorted(set(expected) - set(loaded)):
            raise ValueError(f"{model_path}: DFlash tensors missing: {missing}")
        draft.load_state_dict(loaded)
        # not the root: it reaches the target's embedding and head, which the target finalized already
        for op in (draft.fc, draft.layers, draft.candidate_selector):
            finalize_quant(op)
        layers = draft.layers.op_list
        draft._kv_all = qkv[:, layers[0].self_attn.q_size :]
        draft._k_norms = torch.stack([layer.self_attn.k_norm.weight for layer in layers]).contiguous()
        draft._mask_embedding = draft._embed_tokens.forward(torch.tensor([config.mask_token_id], device=device))[0]
        return draft

    def make_dflash_ring(self, max_rows: int, block: int, ring_len: int, max_position: int, device: torch.device) -> DFlashRing:
        """A zeroed ring of ``ring_len`` positions for ``max_rows`` table rows, drafting blocks of ``block`` tokens at positions below ``max_position``."""
        c = self.config
        assert ring_len >= c.window + block, f"a ring of {ring_len} cannot hold the window {c.window} plus the block {block}"
        k = torch.zeros(c.num_layers, max_rows, ring_len, c.num_kv_heads, c.head_dim, dtype=self._kv_all.dtype, device=device)
        workspace = ring_attention_workspace(max_rows, block, c.num_heads, c.num_kv_heads, c.head_dim, ring_len, device)
        # not the cached get_rope: max_position follows the page table, so each rebuild would cache one more table
        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float, device=device) / c.head_dim))
        freqs = torch.arange(max_position, dtype=torch.float, device=device)[:, None] * inv_freq[None, :]
        return DFlashRing(k, torch.zeros_like(k), block, workspace, torch.cat((freqs.cos(), freqs.sin()), dim=-1))

    # ---------------------------------------------------------------- context KV

    def context_rows(self, features: torch.Tensor) -> torch.Tensor:
        """Captured target features ``[n, taps*hidden]`` -> the context rows ``[n, hidden]`` the ring and the window hold."""
        return self.hidden_norm.forward(self.fc.forward(features))

    def store_context(self, ctx: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor, dflash_ring: DFlashRing) -> None:
        """Every layer's K/V of the context rows into the ring rows ``rows`` at ``positions``, which must be distinct modulo the ring within one call."""
        c, n = self.config, ctx.shape[0]
        kv = torch.empty(n, c.num_layers, self._kv_all.shape[1], dtype=ctx.dtype, device=ctx.device)
        torch.bmm(ctx.expand(c.num_layers, *ctx.shape), self._kv_all.transpose(1, 2), out=kv.transpose(0, 1))
        norm_rope_store(
            kv.view(n, -1), num_layers=c.num_layers, num_q_heads=0, q_out=None, q_norm=None,
            k_norms=self._k_norms, cos_sin=dflash_ring.cos_sin, positions=positions, rows=rows,
            ring_k=dflash_ring.k, ring_v=dflash_ring.v, layer0=0, eps=c.rms_norm_eps,
        )

    # ---------------------------------------------------------------- block draft

    def forward_block(self, bonus: torch.Tensor, starts: torch.Tensor, rows: torch.Tensor, dflash_ring: DFlashRing) -> torch.Tensor:
        """Final hidden ``[bs*block, hidden]`` of the block ``[bonus, mask x (block-1)]`` at positions ``starts + j`` of ring rows ``rows``."""
        c = self.config
        bs, block = bonus.shape[0], dflash_ring.block
        x = self._mask_embedding.expand(bs, block, -1).clone()
        x[:, 0] = self._embed_tokens.forward(bonus)
        x = x.view(bs * block, -1)
        positions = (starts.view(-1, 1) + torch.arange(block, device=bonus.device)).view(-1).to(torch.int32)
        token_rows = rows.repeat_interleave(block)
        scale = c.head_dim**-0.5
        residual = None
        for i, layer in enumerate(self.layers.op_list):
            attn = layer.self_attn
            h, residual = layer.input_layernorm.forward(x, residual)
            h, conv = layer.attention_conv.prepare(h, block)
            q = torch.empty(bs * block, attn.q_size, dtype=h.dtype, device=h.device)
            # the block's own K/V go into the ring, so the attention reads window and block alike
            norm_rope_store(
                attn.qkv_proj.forward(h), num_layers=1, num_q_heads=c.num_heads, q_out=q, q_norm=attn.q_norm.weight,
                k_norms=self._k_norms, cos_sin=dflash_ring.cos_sin, positions=positions, rows=token_rows,
                ring_k=dflash_ring.k, ring_v=dflash_ring.v, layer0=i, eps=c.rms_norm_eps,
            )
            o = attn.o_proj.forward(ring_attention(q, dflash_ring.k, dflash_ring.v, i, rows, starts, c.window, scale, dflash_ring.workspace))
            o = layer.attention_conv.finish(o, conv, block)
            h, residual = layer.post_attention_layernorm.forward(o, residual)
            h, conv = layer.mlp_conv.prepare(h, block)
            x = layer.mlp_conv.finish(layer.mlp.forward(h), conv, block)
        return self.norm.forward(x, residual)[0]

    # ---------------------------------------------------------------- selector

    def compute_candidates(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The target head's top-K token ids and their fp32 logits over ``hidden [..., H]``, best first, so a tie in the walk takes the first."""
        vals, ids = self._top_k(self._lm_head.logits(hidden.reshape(-1, hidden.shape[-1])), self.config.selector_top_k)
        shape = hidden.shape[:-1] + (-1,)
        return ids.long().view(shape), vals.float().view(shape)
