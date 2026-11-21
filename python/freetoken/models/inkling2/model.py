from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.inkling2 import attn_res, short_conv
from freetoken.layers import BaseOP, LinearReplicated, OPList, ParallelLMHead, VocabParallelEmbedding, make_moe_layer
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.utils import nvtx_annotate

from .attention import Inkling2Attention, unweighted_rms_norm
from .state import ConvState

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

    from .config import Inkling2Args


class Inkling2RMSNorm(BaseOP):
    """RMSNorm computed in fp32 from an fp32 or bf16 input, scaled by the bf16 weight, stored bf16."""

    def __init__(self, size: int, eps: float):
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (unweighted_rms_norm(x, self.eps) * self.weight.float()).to(self.weight.dtype)


class Inkling2MessageGate(BaseOP):
    """``(1 - tanh(F z)) * x + tanh(G z) * m`` with ``z = down([rmsnorm(x), rmsnorm(m)])``: how much of
    a looped layer's input a later pass forgets and how much of the previous pass's message it adds."""

    def __init__(self, args: Inkling2Args, *, prefix: str):
        H = args.hidden_size
        self.eps = args.rms_norm_eps
        self.down = LinearReplicated(2 * H, args.gate_rank, has_bias=False, prefix=f"{prefix}.down")
        self.up = torch.empty(2 * H, args.gate_rank, dtype=torch.float32)  # [forget; gain]

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        z = torch.cat([unweighted_rms_norm(x, self.eps), unweighted_rms_norm(m, self.eps)], dim=-1)
        z = self.down.forward(z.to(self.down.weight.dtype)).float()
        forget, gain = torch.tanh(z @ self.up.t()).chunk(2, dim=-1)
        return (1.0 - forget) * x + gain * m


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
        scores = torch.sigmoid(F.linear(x.float(), self.router_weight))
        ids = torch.topk(scores + self.router_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.route_scale
        return self.experts.routed_forward(x, weights, ids.to(torch.int32))


class Inkling2DecoderLayer(BaseOP):
    """One decoder layer; looped layers run once per pass, each visit with its own KV and conv state.

        a = attn_res(block_states + [x], attention_input)    (layer 0: a = x)
        q/k/v inputs = norm(a), or per input norm(gate(a, message)) on later passes
        h = attn_conv(attn(...));  r = h if the layer opens a block else x + h
        y = mlp_conv(moe(mlp_norm([gate](attn_res(blocks + [r], mlp_input)))));  out = r + y

    A block-opening layer also appends its input ``x`` to the block states its MLP reads.
    Global layers read three attention-residual mixes, one per q/k/v input, each with its own norm.
    """

    def __init__(self, config: ModelConfig, args: Inkling2Args, layer_id: int, *, prefix: str):
        self._layer_id = layer_id
        self.args = args
        H = args.hidden_size
        self.is_swa = args.is_swa(layer_id)
        self.is_loop = args.is_loop(layer_id)
        self.is_boundary = args.is_block_boundary(layer_id)
        eps = args.rms_norm_eps
        if layer_id > 0:
            self.attention_input = torch.empty((H,) if self.is_swa else (3, H), dtype=torch.float32)
        self.mlp_input = torch.empty(H, dtype=torch.float32)
        if self.is_swa:
            self.input_norm = Inkling2RMSNorm(H, eps)
        else:
            self.q_input_norm = Inkling2RMSNorm(H, eps)
            self.k_input_norm = Inkling2RMSNorm(H, eps)
            self.v_input_norm = Inkling2RMSNorm(H, eps)
        self.mlp_norm = Inkling2RMSNorm(H, eps)
        self.attn_conv = torch.empty(H, args.conv_kernel, dtype=torch.float32)
        self.mlp_conv = torch.empty(H, args.conv_kernel, dtype=torch.float32)
        self.self_attn = Inkling2Attention(args, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Inkling2MoE(config, args, layer_id, prefix=f"{prefix}.mlp")
        if self.is_loop:
            self.q_gate = Inkling2MessageGate(args, prefix=f"{prefix}.q_gate")
            self.k_gate = Inkling2MessageGate(args, prefix=f"{prefix}.k_gate")
            self.v_gate = Inkling2MessageGate(args, prefix=f"{prefix}.v_gate")
            self.mlp_gate = Inkling2MessageGate(args, prefix=f"{prefix}.mlp_gate")
        self._conv_scale = 1.0 / math.sqrt(args.conv_kernel + 1)

    def _conv(self, x: torch.Tensor, weight: torch.Tensor, state: ConvState, visit: int, name: str) -> torch.Tensor:
        batch = get_global_ctx().batch
        md = batch.attn_metadata
        return short_conv(
            x.contiguous(), weight, state.view(visit, name), state.batch_slots(batch.padded_size), self._conv_scale,
            decode=not batch.is_prefill, q_to_req=md.q_to_req, cu_q=md.cu_seqlens_q_gpu, prefix_lens=md.prefix_lens,
        )

    def _attn_inputs(self, x: torch.Tensor, block_states: list[torch.Tensor], messages: torch.Tensor | None):
        if self._layer_id == 0:
            mixes = [x]
        else:
            queries = self.attention_input.view(-1, x.shape[-1])
            mixes = list(attn_res(block_states + [x], queries))
        if self.is_swa:
            a = mixes[0]
            if messages is None:
                xq = self.input_norm.forward(a)
                return xq, xq, xq
            return tuple(self.input_norm.forward(g.forward(a, m)) for g, m in zip((self.q_gate, self.k_gate, self.v_gate), messages[:3]))
        norms = (self.q_input_norm, self.k_input_norm, self.v_input_norm)
        if messages is None:
            return tuple(n.forward(a) for n, a in zip(norms, mixes))
        gates = (self.q_gate, self.k_gate, self.v_gate)
        return tuple(n.forward(g.forward(a, m)) for n, g, a, m in zip(norms, gates, mixes, messages[:3]))

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, block_states: list[torch.Tensor], messages: torch.Tensor | None, visit: int, state: ConvState) -> torch.Tensor:
        xq, xk, xv = self._attn_inputs(x, block_states, messages)
        h = self.self_attn.forward(xq, xk, xv, visit, state)
        h = self._conv(h, self.attn_conv, state, visit, "attn")
        r = h.float() if self.is_boundary else x.float() + h.float()
        blocks = block_states + [x] if self.is_boundary else block_states
        m = attn_res(blocks + [r], self.mlp_input.view(1, -1))[0]
        if messages is not None:
            m = self.mlp_gate.forward(m, messages[3])
        y = self.mlp.forward(self.mlp_norm.forward(m))
        y = self._conv(y, self.mlp_conv, state, visit, "mlp")
        return (r + y.float()).to(x.dtype)


class _NormedEmbedding:
    """embed_tokens followed by embed_norm as one lookup for embed_input_ids, so image rows scatter in after the norm as in the reference."""

    def __init__(self, embed_tokens: VocabParallelEmbedding, embed_norm: Inkling2RMSNorm):
        self.embed_tokens = embed_tokens
        self.embed_norm = embed_norm

    @property
    def num_embeddings(self) -> int:
        return self.embed_tokens.num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm.forward(self.embed_tokens.forward(input_ids))


class Inkling2Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        args: Inkling2Args = config.inkling2_args
        self.args = args
        H = args.hidden_size
        self.embed_tokens = VocabParallelEmbedding(num_embeddings=args.vocab_size, embedding_dim=H, prefix=f"{prefix}.embed_tokens")
        self.embed_norm = Inkling2RMSNorm(H, args.rms_norm_eps)
        self._normed_embedding = _NormedEmbedding(self.embed_tokens, self.embed_norm)
        self.layers = OPList([Inkling2DecoderLayer(config, args, i, prefix=f"{prefix}.layers.{i}") for i in range(args.num_layers)])
        self.message_queries = torch.empty(args.num_loop_layers, 4, H, dtype=torch.float32)
        self.final_query = torch.empty(H, dtype=torch.float32)
        self.norm = Inkling2RMSNorm(H, args.rms_norm_eps)
        self._state = ConvState(args, torch.bfloat16)

    def _run(self, layers: range, x: torch.Tensor, block_states: list[torch.Tensor], pass_idx: int, prev: list[torch.Tensor] | None):
        args = self.args
        for layer in layers:
            messages = None
            if prev is not None:
                messages = attn_res(prev, self.message_queries[layer - args.loop_start])
            out = self.layers.op_list[layer].forward(x, block_states, messages, args.visit(layer, pass_idx), self._state)
            if args.is_block_boundary(layer):
                block_states = block_states + [x]
            x = out
        return x, block_states

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        args = self.args
        batch = get_global_ctx().batch
        x = embed_input_ids(self._normed_embedding, input_ids, batch)
        x, blocks = self._run(range(args.loop_start), x, [], 0, None)
        pre_x, pre_blocks = x, blocks
        loop = range(args.loop_start, args.loop_end)
        x, blocks = self._run(loop, x, blocks, 0, None)
        for pass_idx in range(1, args.num_passes):
            # every later pass restarts from the prelude and reads the previous pass's message:
            # the blocks it closed after its first one, plus its output
            prev = blocks[len(pre_blocks) + 1 :] + [x]
            x, blocks = self._run(loop, pre_x, pre_blocks, pass_idx, prev)
        x, blocks = self._run(range(args.loop_end, args.num_layers), x, blocks, 0, None)
        return attn_res(blocks + [x], self.final_query.view(1, -1), norm_weight=self.norm.weight.view(1, -1), norm_eps=args.rms_norm_eps, out_dtype=x.dtype)[0]


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
    pass


__all__ = ["Inkling2LoopForCausalLM", "Inkling2LoopForConditionalGeneration"]
