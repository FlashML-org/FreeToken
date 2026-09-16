"""DeepSeek-V4.1 backbone with shifted mHC mixing and image-aware MoE/Engram."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.models.blocks import BaseLLMModel

from .attention import Attention
from .layers import OutputHead, RMSNorm
from .moe import MoE


def make_identity_pre_mix(x, hc_mult):
    pre = x.new_zeros(*x.shape[:-2], hc_mult, dtype=torch.float32)
    pre[..., 0] = 1.0
    return pre


class Block(nn.Module):
    def __init__(self, layer_id, args, *, strategy="offload", decode_target="gpu", quant_config=None):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args, strategy=strategy, decode_target=decode_target,
                       quant_config=quant_config)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.engram = None
        if layer_id in args.engram_layer_ids:
            from .engram import Engram

            self.engram = Engram(args, layer_id)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        for sublayer in ("attn", "ffn"):
            for name, shape in (("fn", (mix_hc, args.hc_mult * args.dim)),
                                ("base", (mix_hc,)), ("scale", (3,))):
                self.register_parameter(f"hc_{sublayer}_{name}", nn.Parameter(
                    torch.empty(shape, dtype=torch.float32), requires_grad=False))

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        flat = x.flatten(-2).float()
        mixes = F.linear(flat, hc_fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        hc = self.hc_mult
        if x.is_cuda:
            from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

            pre, post, comb = hc_split_sinkhorn(mixes.reshape(-1, mixes.shape[-1]), hc_scale,
                                               hc_base, hc, self.hc_sinkhorn_iters, self.hc_eps)
            return (pre.reshape(*x.shape[:-2], hc), post.reshape(*x.shape[:-2], hc),
                    comb.reshape(*x.shape[:-2], hc, hc))
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + self.hc_eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(*x.shape[:-2], hc, hc)
        comb = comb.softmax(-1) + self.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def hc_pre(self, x, pre):
        if x.is_cuda:
            from freetoken.kernel.triton.dsv4.hc import hc_pre_combine

            return hc_pre_combine(x.reshape(-1, self.hc_mult, self.dim),
                                  pre.reshape(-1, self.hc_mult), x.dtype).reshape(*x.shape[:-2], self.dim)
        return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        if x.is_cuda:
            from freetoken.kernel.triton.dsv4.hc import hc_post_combine

            return hc_post_combine(x.reshape(-1, self.dim), residual.reshape(-1, self.hc_mult, self.dim),
                                   post.reshape(-1, self.hc_mult),
                                   comb.reshape(-1, self.hc_mult, self.hc_mult)).reshape(residual.shape)
        mixed = torch.einsum("...pq,...pd->...qd", comb.float(), residual.float())
        return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)

    def _forward(self, h, pre_mix, image_mask, attention):
        residual = h
        attn_pre, post, comb = self.hc_mixes(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = attention(self.attn_norm(self.hc_pre(h, pre_mix)))
        h = self.hc_post(x, residual, post, comb)
        residual = h
        ffn_pre, post, comb = self.hc_mixes(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn(self.ffn_norm(self.hc_pre(h, attn_pre)), image_mask)
        return self.hc_post(x, residual, post, comb), ffn_pre

    def prefill_batched(self, h, pre_mix, image_mask, segments, positions):
        return self._forward(h, pre_mix, image_mask,
                             lambda x: self.attn.forward_ragged(x, segments, positions))

    def decode_step(self, h, pre_mix, pos, rows, stage_cap, wctx=None):
        return self._forward(h, pre_mix, None,
                             lambda x: self.attn.decode_step(x, pos, rows, stage_cap, wctx))


class Transformer(nn.Module):
    def __init__(self, args, *, strategy="offload", decode_target="gpu", quant_config=None):
        super().__init__()
        if quant_config is None:
            from .config import DeepseekV41QuantConfig

            quant_config = DeepseekV41QuantConfig(args)
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed = nn.Embedding(args.vocab_size, args.dim, dtype=torch.bfloat16)
        self.embed.weight.requires_grad_(False)
        self.layers = nn.ModuleList([Block(i, args, strategy=strategy, decode_target=decode_target,
                                            quant_config=quant_config) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = OutputHead(args.dim, args.vocab_size)
        self.vision = None
        if args.vision_enabled:
            from .vision import Aligner, ViT

            self.vision, self.aligner = ViT(args), Aligner(args)
            for name in ("image_start", "image_end", "image_newline"):
                self.register_parameter(name, nn.Parameter(torch.empty(args.dim, dtype=torch.bfloat16),
                                                          requires_grad=False))

    def bind(self, pool, device):
        for layer in self.layers:
            layer.attn.bind(pool, device)

    def encode_image(self, patches, n_vit_h, n_vit_w):
        if self.vision is None:
            raise ValueError("This DeepSeek-V4.1 checkpoint has no vision tower")
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def prefill_batched(self, input_ids, segments, positions, last_indices, batch=None):
        ids = input_ids.flatten()
        if batch is not None and getattr(batch, "mm_embeds", None) is not None:
            ids = ids.clamp(max=self.args.vocab_size - 1)
        flat = self.embed(ids)
        image_mask = None
        if batch is not None and self.vision is not None:
            from .vision import merge_image_embeddings

            flat, image_mask = merge_image_embeddings(self, batch, flat)
        h = flat.view(1, -1, self.args.dim).unsqueeze(-2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h)
            h, pre_mix = layer.prefill_batched(h, pre_mix, image_mask, segments, positions)
        h = self.norm(self.layers[-1].hc_pre(h, pre_mix))
        return self.head(h[0, last_indices])

    def decode(self, input_ids, pos, stage_cap):
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        h = self.embed(input_ids).unsqueeze(-2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h)
            h, pre_mix = layer.decode_step(h, pre_mix, pos, rows, stage_cap)
        return self.head(self.norm(self.layers[-1].hc_pre(h, pre_mix))[:, -1])


class DeepseekV41ForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self._config = config
        self._args = config.dsv41_args
        self._transformer = Transformer(self._args, strategy=config.moe_strategy,
                                        decode_target=config.decode_target, quant_config=config.quant)
        # The engine walks BaseOP children; resident nn.Module weights use the adapter below.
        self._offload_layers = list(self._iter_offload_moe_layers())
        self._bound = False
        self._engram_runtime = None

    def _ensure_bound(self):
        if not self._bound:
            pool = get_global_ctx().kv_cache
            self._transformer.bind(pool, pool.device)
            self._bound = True

    def mark_for_rebind(self):
        self._bound = False

    def place_encoder_weights(self, mode):
        if self._transformer.vision is not None:
            self._transformer.vision.place_weights(mode)

    def encode(self, item):
        if item.modality != "image" or self._transformer.vision is None:
            raise ValueError("DeepSeek-V4.1 supports image items only when vision is enabled")
        from .vision import image_span_embeddings

        embeddings = image_span_embeddings(self._transformer, item.feature, item.n_vit_h, item.n_vit_w, item.types)
        if embeddings.shape[0] != item.num_tokens:
            raise ValueError("DeepSeek-V4.1 image embedding count does not match its offsets")
        return embeddings

    def _iter_offload_moe_layers(self):
        for layer in self._transformer.layers:
            yield layer.ffn.experts

    def load_host_tables(self, engine_config):
        from .engram import prepare_engram

        return prepare_engram(self, engine_config)

    @contextmanager
    def forward_host_ctx(self, batch, use_graph):
        if self._engram_runtime is None:
            yield
        else:
            with self._engram_runtime.forward_host_ctx(batch, use_graph):
                yield

    def state_dict(self, *, prefix="", result=None):
        result = {} if result is None else result
        for name, param in self._transformer.named_parameters():
            result[f"{prefix}.{name}" if prefix else name] = param
        return result

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        casted = {}
        for name, param in self._transformer.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            if key not in state_dict:
                raise RuntimeError(f"Missing DeepSeek-V4.1 weight: {key}")
            tensor = state_dict.pop(key)
            if tensor.shape != param.shape:
                raise ValueError(f"DeepSeek-V4.1 weight {key} has shape {tuple(tensor.shape)}; "
                                 f"expected {tuple(param.shape)}")
            casted[name] = tensor.to(param.dtype)
        if state_dict and not _internal:
            raise RuntimeError(f"Unexpected DeepSeek-V4.1 weights: {list(state_dict)[:8]}")
        self._transformer.load_state_dict(casted, assign=True, strict=False)

    def forward(self):
        self._ensure_bound()
        batch = get_global_ctx().batch
        md = batch.attn_metadata
        input_ids = batch.input_ids.long()
        if batch.is_prefill:
            return self._transformer.prefill_batched(input_ids, md.segments,
                                                     batch.positions.long(), md.last_indices.long(), batch)
        size = batch.padded_size
        pos = batch.positions.long().view(-1)[:size]
        stage_cap = md.stage_width - 1 if torch.cuda.is_current_stream_capturing() else int(pos.max().item())
        return self._transformer.decode(input_ids.view(size, 1), pos, stage_cap)


__all__ = ["Block", "Transformer", "DeepseekV41ForCausalLM", "make_identity_pre_mix"]
