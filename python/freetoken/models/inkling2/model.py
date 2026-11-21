from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
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
    from freetoken.core import Batch
    from freetoken.message import MMItem
    from freetoken.models.config import ModelConfig

    from .config import Inkling2Args


class Inkling2MessageGates(BaseOP):
    """``num`` message gates applied side by side: ``(1 - tanh(F z)) * x + tanh(G z) * m`` with
    ``z = down([rmsnorm(x), rmsnorm(m)])`` -- how much of a looped layer's input a later pass
    forgets and how much of the previous pass's message it adds. ``up`` stacks ``[F; G]``."""

    def __init__(self, args: Inkling2Args, num: int):
        H = args.hidden_size
        self.eps = args.rms_norm_eps
        self.down = torch.empty(num, args.gate_rank, 2 * H)
        self.up = torch.empty(num, 2 * H, args.gate_rank, dtype=torch.float32)

    def forward(self, x: torch.Tensor, m: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
        """``x`` / ``m`` ``[num, T, H]``; the gated rows, RMS-normalized with ``norm_weight`` ``[num, H]``, in bf16."""
        z = torch.bmm(gate_inputs(x, m, self.eps), self.down.transpose(1, 2))
        forget_gain = torch.bmm(z.float(), self.up.transpose(1, 2))
        return gate_combine(forget_gain, x, m, norm_weight=norm_weight, norm_eps=norm_eps, out_dtype=self.down.dtype)


class Inkling2MoE(BaseOP):
    """fp32 sigmoid router with a selection-only bias; the chosen scores, renormalized and scaled, weight the experts."""

    def __init__(self, config: ModelConfig, args: Inkling2Args, layer_id: int, *, prefix: str):
        self.top_k = args.top_k
        self.route_scale = args.route_scale
        self.router_weight = torch.empty(args.num_experts, args.hidden_size, dtype=torch.float32)
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
        logits = F.linear(x.float(), self.router_weight)
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
            x, weight, state.view(visit, name), state.batch_slots(batch.padded_size), self._conv_scale,
            decode=not batch.is_prefill, q_to_req=md.q_to_req, cu_q=md.cu_seqlens_q_gpu, prefix_lens=md.prefix_lens, **kw,
        )

    def _attn_inputs(self, x: torch.Tensor, blocks: torch.Tensor, nb: int, messages: torch.Tensor | None):
        if self._layer_id == 0:
            xq = self.input_norm.forward(x)
            return xq, xq, xq
        T, H = x.shape
        norms = self.input_norm.weight.view(1, H).expand(3, H) if self.is_swa else self.qkv_input_norm
        if messages is None:
            mixes = attn_res(blocks, nb, x, self.attention_input, norm_weight=norms[: self.attention_input.shape[0]], norm_eps=self.eps, out_dtype=x.dtype)
            return (mixes[0],) * 3 if self.is_swa else tuple(mixes)
        mixes = attn_res(blocks, nb, x, self.attention_input)
        gated = self.attn_gates.forward(mixes.expand(3, T, H), messages[:3], norms, self.eps)
        return tuple(gated)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, blocks: torch.Tensor, nb: int, messages: torch.Tensor | None, visit: int, state: ConvState):
        xq, xk, xv = self._attn_inputs(x, blocks, nb, messages)
        h = self.self_attn.forward(xq, xk, xv, visit, state)
        h = self._conv(h, self.attn_conv, state, visit, "attn")
        if self.is_boundary:
            blocks[:, nb].copy_(x)
            nb += 1
            res, res2 = h, None
        else:
            res, res2 = x, h
        if messages is None:
            m = attn_res(blocks, nb, res, self.mlp_input, partial2=res2, norm_weight=self.mlp_norm, norm_eps=self.eps, out_dtype=x.dtype)[0]
        else:
            m = attn_res(blocks, nb, res, self.mlp_input, partial2=res2)
            m = self.mlp_gate.forward(m, messages[3:], self.mlp_norm, self.eps)[0]
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
        self._state = ConvState(args, torch.bfloat16)

    def _run(self, layers: range, x: torch.Tensor, blocks: torch.Tensor, nb: int, pass_idx: int, prev: tuple | None):
        args = self.args
        for layer in layers:
            messages = None
            if prev is not None:
                prev_blocks, prev_nb, prev_x = prev
                messages = attn_res(prev_blocks, prev_nb, prev_x, self.message_queries[layer - args.loop_start])
            x, nb = self.layers.op_list[layer].forward(x, blocks, nb, messages, args.visit(layer, pass_idx), self._state)
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
        for pass_idx in range(1, args.num_passes):
            # every later pass restarts from the prelude and reads the previous pass's message:
            # the blocks it closed after its first one, plus its output
            prev = (blocks[:, pre_nb + 1 : nb].clone(), nb - pre_nb - 1, x)
            x, nb = self._run(loop, pre_x, blocks, pre_nb, pass_idx, prev)
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

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        self.model._state.stage(batch)
        yield

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
