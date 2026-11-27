from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.inkling2 import EPI_RESIDUAL, attn_res, gate_combine, gate_inputs, short_conv
from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding, make_moe_layer
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.utils import nvtx_annotate

from .attention import Inkling2Attention
from .state import ConvState
from .vision import Inkling2VisionModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from freetoken.message import MMItem
    from freetoken.models.config import ModelConfig

    from .config import Inkling2Args

# loop-pass prefill chunks of at least this many tokens mix message rows and run the attention gates one at a time, to bound their fp32 temporaries
_SPLIT_TOKENS = 2048


class Inkling2MessageGates(BaseOP):
    """``num`` message gates applied side by side: ``(1 - tanh(F z)) * x + tanh(G z) * m`` with
    ``z = down([rmsnorm(x), rmsnorm(m)])`` -- how much of a looped layer's input a later pass
    forgets and how much of the previous pass's message it adds. ``up`` stacks ``[F; G]``."""

    def __init__(self, args: Inkling2Args, num: int):
        H = args.hidden_size
        self.eps = args.rms_norm_eps
        self.down = torch.empty(num, args.gate_rank, 2 * H)
        self.up = torch.empty(num, 2 * H, args.gate_rank)

    def forward(self, x: torch.Tensor, m: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
        """``x`` / ``m`` ``[num, T, H]``; the gated rows, RMS-normalized with ``norm_weight`` ``[num, H]``, in bf16."""
        z = torch.bmm(gate_inputs(x, m, self.eps), self.down.transpose(1, 2))
        forget_gain = torch.bmm(z, self.up.transpose(1, 2), out_dtype=torch.float32)
        return gate_combine(forget_gain, x, m, norm_weight=norm_weight, norm_eps=norm_eps, out_dtype=self.down.dtype)

    def forward_each(
        self, x: Callable[[int], torch.Tensor], m: Callable[[int], torch.Tensor], T: int, norm_weight: torch.Tensor, norm_eps: float
    ) -> torch.Tensor:
        """``forward`` one gate at a time, so only one gate's ``[1, T, *]`` temporaries are live; ``x(g)`` / ``m(g)`` build gate ``g``'s rows."""
        G, H = self.down.shape[0], self.down.shape[2] // 2
        out = self.down.new_empty((G, T, H))
        for g in range(G):
            xg, mg = x(g), m(g)
            z = torch.bmm(gate_inputs(xg, mg, self.eps), self.down[g : g + 1].transpose(1, 2))
            forget_gain = torch.bmm(z, self.up[g : g + 1].transpose(1, 2), out_dtype=torch.float32)
            del z
            gate_combine(forget_gain, xg, mg, norm_weight=norm_weight[g : g + 1], norm_eps=norm_eps, out=out[g : g + 1])
            del xg, mg, forget_gain
        return out


class Inkling2Message:
    """A loop layer's four message rows (q, k, v input, then MLP input), mixed from the previous pass's blocks and output.

    A split message mixes rows only when asked, so none outlives its gate; otherwise all four are mixed at once.
    """

    def __init__(self, blocks: torch.Tensor, nb: int, x: torch.Tensor, queries: torch.Tensor, split: bool):
        self.split = split
        self._sources = (blocks, nb, x)
        self._queries = queries
        self._rows = None if split else attn_res(blocks, nb, x, queries)

    def rows(self, lo: int, hi: int) -> torch.Tensor:
        """Rows ``lo:hi``, fp32 ``[hi - lo, T, H]``."""
        if self._rows is not None:
            return self._rows[lo:hi]
        return attn_res(*self._sources, self._queries[lo:hi])


class Inkling2MoE(BaseOP):
    """fp32 sigmoid router with a selection-only bias; the chosen scores, renormalized and scaled, weight the experts."""

    def __init__(self, config: ModelConfig, args: Inkling2Args, layer_id: int, *, prefix: str):
        self.top_k = args.top_k
        self.route_scale = args.route_scale
        self.router_weight = torch.empty(args.num_experts, args.hidden_size)
        self.router_bias = torch.empty(args.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            activation="softcap_swiglu",
            renormalize=False,
            alpha=args.gate_softcap,
            limit=args.up_softcap,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )

    @nvtx_annotate("MoE")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.mm(x, self.router_weight.t(), out_dtype=torch.float32)
        weights, ids = fused_topk_sigmoid(logits, self.router_bias, self.top_k, renormalize=True, routed_scale=self.route_scale)
        return self.experts.routed_forward(x, weights, ids)


class Inkling2DecoderLayer(BaseOP):
    """One decoder layer; looped layers run once per pass, each visit with its own KV and conv state.

        a = attn_res(blocks + [x], attention_input)    (layer 0: a = x)
        q/k/v inputs = norm(a), or per input norm(gate(a, message)) on later passes
        h = attn_conv(attn(...));  r = h if the layer opens a block else x + h
        y = mlp_conv(moe(mlp_norm([gate](attn_res(blocks + [r], mlp_input)))));  out = r + y

    A block-opening layer also appends its input ``x`` to the block states its MLP reads.
    Global layers read three attention-residual mixes, one per q/k/v input, each with its own
    norm (``qkv_input_norm`` stacks them).
    """

    def __init__(self, config: ModelConfig, args: Inkling2Args, layer_id: int, *, prefix: str):
        self._layer_id = layer_id
        self.args = args
        H = args.hidden_size
        self.is_swa = args.is_swa(layer_id)
        self.is_loop = args.is_loop(layer_id)
        self.is_boundary = args.is_block_boundary(layer_id)
        self.eps = args.rms_norm_eps
        if layer_id > 0:
            self.attention_input = torch.empty((1 if self.is_swa else 3, H), dtype=torch.float32)
        self.mlp_input = torch.empty(1, H, dtype=torch.float32)
        if self.is_swa:
            self.input_norm = RMSNorm(H, self.eps)
        else:
            self.qkv_input_norm = torch.empty(3, H)
        self.mlp_norm = torch.empty(1, H)
        self.attn_conv = torch.empty(H, args.conv_kernel, dtype=torch.float32)
        self.mlp_conv = torch.empty(H, args.conv_kernel, dtype=torch.float32)
        self.self_attn = Inkling2Attention(args, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Inkling2MoE(config, args, layer_id, prefix=f"{prefix}.mlp")
        if self.is_loop:
            self.attn_gates = Inkling2MessageGates(args, 3)
            self.mlp_gate = Inkling2MessageGates(args, 1)
        self._conv_scale = 1.0 / math.sqrt(args.conv_kernel + 1)

    def _conv(self, x, weight, state: ConvState, visit: int, name: str, **kw) -> torch.Tensor:
        batch = get_global_ctx().batch
        md = batch.attn_metadata
        return short_conv(
            x, weight, *state.rows(visit, name), md.indices, md.indptr, md.prefix_lens, self._conv_scale,
            decode=not batch.is_prefill, q_to_req=md.q_to_req, cu_q=md.cu_seqlens_q_gpu, **kw,
        )

    def _attn_inputs(self, x: torch.Tensor, blocks: torch.Tensor, nb: int, msg: Inkling2Message | None):
        if self._layer_id == 0:
            xq = self.input_norm.forward(x)
            return xq, xq, xq
        T, H = x.shape
        norms = self.input_norm.weight.view(1, H).expand(3, H) if self.is_swa else self.qkv_input_norm
        if msg is None:
            mixes = attn_res(blocks, nb, x, self.attention_input, norm_weight=norms[: self.attention_input.shape[0]], norm_eps=self.eps, out_dtype=x.dtype)
            return (mixes[0],) * 3 if self.is_swa else tuple(mixes)
        if not msg.split:
            mixes = attn_res(blocks, nb, x, self.attention_input)
            return tuple(self.attn_gates.forward(mixes.expand(3, T, H), msg.rows(0, 3), norms, self.eps))
        swa_mix = attn_res(blocks, nb, x, self.attention_input) if self.is_swa else None

        def mix(g: int) -> torch.Tensor:
            return swa_mix if swa_mix is not None else attn_res(blocks, nb, x, self.attention_input[g : g + 1])

        return tuple(self.attn_gates.forward_each(mix, lambda g: msg.rows(g, g + 1), T, norms, self.eps))

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, blocks: torch.Tensor, nb: int, msg: Inkling2Message | None, visit: int, state: ConvState):
        xq, xk, xv = self._attn_inputs(x, blocks, nb, msg)
        h = self.self_attn.forward(xq, xk, xv, visit, state)
        del xq, xk, xv
        h = self._conv(h, self.attn_conv, state, visit, "attn")
        if self.is_boundary:
            blocks[:, nb].copy_(x)
            nb += 1
            res, res2 = h, None
        else:
            res, res2 = x, h
        if msg is None:
            m = attn_res(blocks, nb, res, self.mlp_input, partial2=res2, norm_weight=self.mlp_norm, norm_eps=self.eps, out_dtype=x.dtype)[0]
        else:
            m = attn_res(blocks, nb, res, self.mlp_input, partial2=res2)
            m = self.mlp_gate.forward(m, msg.rows(3, 4), self.mlp_norm, self.eps)[0]
        y = self.mlp.forward(m)
        return self._conv(y, self.mlp_conv, state, visit, "mlp", epilogue=EPI_RESIDUAL, residual=res, residual2=res2), nb


class Inkling2Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        args: Inkling2Args = config.inkling2_args
        self.args = args
        H = args.hidden_size
        self.embed_tokens = VocabParallelEmbedding(num_embeddings=args.vocab_size, embedding_dim=H, prefix=f"{prefix}.embed_tokens")
        self.embed_norm = RMSNorm(H, args.rms_norm_eps)
        self._normed_embedding = _NormedEmbedding(self.embed_tokens, self.embed_norm)
        self.layers = OPList([Inkling2DecoderLayer(config, args, i, prefix=f"{prefix}.layers.{i}") for i in range(args.num_layers)])
        self.message_queries = torch.empty(args.num_loop_layers, 4, H, dtype=torch.float32)
        self.final_query = torch.empty(1, H, dtype=torch.float32)
        self.norm = torch.empty(1, H)
        self._state = ConvState(args)

    def _run(self, layers: range, x: torch.Tensor, blocks: torch.Tensor, nb: int, pass_idx: int, prev: tuple | None):
        args = self.args
        split = prev is not None and get_global_ctx().batch.is_prefill and x.shape[0] >= _SPLIT_TOKENS
        for layer in layers:
            msg = None if prev is None else Inkling2Message(*prev, self.message_queries[layer - args.loop_start], split)
            x, nb = self.layers.op_list[layer].forward(x, blocks, nb, msg, args.visit(layer, pass_idx), self._state)
        return x, nb

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        args = self.args
        batch = get_global_ctx().batch
        x = embed_input_ids(self._normed_embedding, input_ids, batch)
        # the attention-residual sources: [embedding, prelude out, loop block, loop out]
        blocks = torch.empty((x.shape[0], 4, x.shape[1]), dtype=x.dtype, device=x.device)
        x, nb = self._run(range(args.loop_start), x, blocks, 0, 0, None)
        pre_x, pre_nb = x, nb
        loop = range(args.loop_start, args.loop_end)
        x, nb = self._run(loop, x, blocks, nb, 0, None)
        prev = None
        for pass_idx in range(1, args.num_passes):
            # every later pass restarts from the prelude and reads the previous pass's message:
            # the blocks it closed after its first one, plus its output
            prev = (blocks[:, pre_nb + 1 : nb].clone(), nb - pre_nb - 1, x)
            x, nb = self._run(loop, pre_x, blocks, pre_nb, pass_idx, prev)
        del pre_x, prev
        x, nb = self._run(range(args.loop_end, args.num_layers), x, blocks, nb, 0, None)
        return attn_res(blocks, nb, x, self.final_query, norm_weight=self.norm, norm_eps=args.rms_norm_eps, out_dtype=x.dtype)[0]


class _NormedEmbedding:
    """embed_tokens followed by embed_norm as one lookup for embed_input_ids, so image rows scatter in after the norm as in the reference."""

    def __init__(self, embed_tokens: VocabParallelEmbedding, embed_norm: RMSNorm):
        self.embed_tokens = embed_tokens
        self.embed_norm = embed_norm

    @property
    def num_embeddings(self) -> int:
        return self.embed_tokens.num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm.forward(self.embed_tokens.forward(input_ids))


class Inkling2LoopForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Inkling2Model(config)
        args: Inkling2Args = config.inkling2_args
        self.lm_head = ParallelLMHead(num_embeddings=args.unpadded_vocab_size, embedding_dim=args.hidden_size, prefix="lm_head")
        self._softcap = args.final_logit_softcap
        super().__init__()

    def forward(self) -> torch.Tensor:
        logits = self.lm_head.forward(self.model.forward(get_global_ctx().batch.input_ids)).float()
        return torch.tanh(logits / self._softcap) * self._softcap


class Inkling2LoopForConditionalGeneration(Inkling2LoopForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            self.visual = Inkling2VisionModel(config.vision_config)

    def place_encoder_weights(self, mode: str) -> None:
        self.visual.place_weights(mode)

    def encode(self, item: MMItem) -> torch.Tensor:
        return self.visual.forward(item.feature, [item.grid_thw])


__all__ = ["Inkling2LoopForCausalLM", "Inkling2LoopForConditionalGeneration"]
