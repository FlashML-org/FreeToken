from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
    make_moe_layer,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from .config import NemotronHArgs


def _linear(args: "NemotronHArgs", name: str, in_f: int, out_f: int):
    if args.module_quant(name) == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_f, out_f, has_bias=False)
    return LinearReplicated(in_f, out_f, has_bias=False)


class _DepthwiseConv1d(BaseOP):
    def __init__(self, dim: int, kernel: int):
        self.weight = torch.empty(dim, 1, kernel)
        self.bias = torch.empty(dim)


class _MambaGatedRMSNorm(BaseOP):
    def __init__(self, size: int, groups: int, eps: float):
        self.weight = torch.empty(size)
        self.groups = groups
        self.group_size = size // groups
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float() * F.silu(gate.float())
        shape = x.shape
        grouped = x.view(*shape[:-1], self.groups, self.group_size)
        grouped = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)
        return grouped.view(shape).to(dtype) * self.weight


class NemotronHMamba2Mixer(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        args = config.nemotron_h_args
        assert args is not None
        self.layer_id = layer_id
        self.num_heads = args.mamba_num_heads
        self.head_dim = args.mamba_head_dim
        self.state_size = args.ssm_state_size
        self.n_groups = args.n_groups
        self.intermediate_size = args.mamba_intermediate_size
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
        self.chunk_size = args.chunk_size
        prefix = f"backbone.layers.{layer_id}.mixer"
        self.in_proj = _linear(
            args, f"{prefix}.in_proj", config.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
        )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, args.conv_kernel)
        self.dt_bias = torch.empty(self.num_heads, dtype=torch.float32)
        self.A_log = torch.empty(self.num_heads, dtype=torch.float32)
        self.norm = _MambaGatedRMSNorm(
            self.intermediate_size, self.n_groups, config.rms_norm_eps
        )
        self.D = torch.empty(self.num_heads, dtype=torch.float32)
        self.out_proj = _linear(
            args, f"{prefix}.out_proj", self.intermediate_size, config.hidden_size
        )

    def _conv(self, conv_in: torch.Tensor, fla, pool, decode: bool) -> torch.Tensor:
        li = pool.local_index(self.layer_id)
        weight = self.conv1d.weight.squeeze(1)
        if decode:
            return causal_conv1d_decode(
                conv_in, pool.conv_states[li], weight, fla.cache_indices,
                bias=self.conv1d.bias,
            )
        return causal_conv1d_varlen(
            conv_in.transpose(0, 1).contiguous(), weight, pool.conv_states[li],
            fla.cu_seqlens, fla.cache_indices, fla.has_initial_state,
            bias=self.conv1d.bias,
        ).transpose(0, 1)

    def _scan(self, x, dt, B, C, initial):
        # Transformers' implementation has a pure-Torch chunk fallback when mamba_ssm is
        # absent. It is the reference recurrence and avoids an O(sequence) Python loop.
        from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan

        return mamba2_chunk_scan(
            x.unsqueeze(0), dt.unsqueeze(0), -torch.exp(self.A_log.float()),
            B.unsqueeze(0), C.unsqueeze(0), chunk_size=self.chunk_size,
            D=self.D, dt_bias=self.dt_bias, initial_states=initial.unsqueeze(0),
            dt_softplus=True, return_final_states=True,
        )

    def _prefill_scan(self, x, dt, B, C, fla, pool) -> torch.Tensor:
        li = pool.local_index(self.layer_id)
        if fla.fresh_state_indices is not None:
            pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
        outputs = []
        offset = 0
        for req in get_global_ctx().batch.padded_reqs:
            length = req.extend_len
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            initial = pool.recurrent_states[li, slot].transpose(-1, -2).contiguous()
            sx, sdt, sB, sC = (v[offset:offset + length] for v in (x, dt, B, C))

            # Hybrid-radix asks for at most one mid-chunk snapshot per request. Split the
            # reference scan at that boundary so the donated state is exact.
            boundary = None
            if req.mamba_last_track_seqlen is not None:
                candidate = req.mamba_last_track_seqlen - req.cached_len
                if 0 < candidate < length:
                    boundary = candidate
            if boundary is not None:
                out1, state1 = self._scan(
                    sx[:boundary], sdt[:boundary], sB[:boundary], sC[:boundary], initial
                )
                assert req.mamba_ping_pong is not None
                dst = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
                pool.recurrent_states[li, dst].copy_(state1[0].transpose(-1, -2))
                out2, final = self._scan(
                    sx[boundary:], sdt[boundary:], sB[boundary:], sC[boundary:], state1[0]
                )
                out = torch.cat((out1[0], out2[0]), dim=0)
            else:
                scanned, final = self._scan(sx, sdt, sB, sC, initial)
                out = scanned[0]
            pool.recurrent_states[li, slot].copy_(final[0].transpose(-1, -2))
            outputs.append(out)
            offset += length
        return torch.cat(outputs, dim=0)

    def _decode_scan(self, x, dt, B, C, fla, pool) -> torch.Tensor:
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            mamba2_selective_state_update,
        )

        li = pool.local_index(self.layer_id)
        indices = fla.cache_indices.long()
        state = pool.recurrent_states[li].index_select(0, indices).transpose(-1, -2).contiguous()
        A = -torch.exp(self.A_log.float())[:, None, None].expand(
            -1, self.head_dim, self.state_size
        )
        out = mamba2_selective_state_update(
            state, x, dt[:, :, None].expand(-1, -1, self.head_dim), A, B, C,
            self.D[:, None].expand(-1, self.head_dim),
            dt_bias=self.dt_bias[:, None].expand(-1, self.head_dim), dt_softplus=True,
        )
        pool.recurrent_states[li].index_copy_(0, indices, state.transpose(-1, -2))
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch, pool = ctx.batch, ctx.linear_state_pool
        assert pool is not None
        fla = batch.fla_metadata
        if fla is None:
            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla
        proj = self.in_proj.forward(hidden_states)
        gate, conv_in, dt = torch.split(
            proj, [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )
        mixed = self._conv(conv_in, fla, pool, batch.is_decode)
        if not batch.is_decode and fla.track_dst is not None:
            li = pool.local_index(self.layer_id)
            # The causal-conv pool itself is updated to the end of each extend. Preserve
            # the raw conv-input window at the earlier radix snapshot boundary separately.
            conv_window = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()
            pool.conv_states[li].index_copy_(
                0, fla.track_dst, conv_window.to(pool.conv_states.dtype)
            )
        x, B, C = torch.split(
            mixed,
            [self.intermediate_size, self.n_groups * self.state_size,
             self.n_groups * self.state_size],
            dim=-1,
        )
        x = x.view(-1, self.num_heads, self.head_dim)
        B = B.view(-1, self.n_groups, self.state_size)
        C = C.view(-1, self.n_groups, self.state_size)
        if batch.is_decode:
            scanned = self._decode_scan(x, dt, B, C, fla, pool)
        else:
            scanned = self._prefill_scan(x, dt, B, C, fla, pool)
        # The pure-Torch chunk scan accumulates/returns fp32; the reference casts back
        # to the input dtype before the gated norm and output projection.
        out = self.norm.forward(
            scanned.reshape(-1, self.intermediate_size).to(gate.dtype), gate
        )
        return self.out_proj.forward(out)


class NemotronHAttention(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        args = config.nemotron_h_args
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_dim = self.num_q * self.head_dim
        self.kv_dim = self.num_kv * self.head_dim
        # Nemotron-H attention is deliberately position-free: unlike most transformer
        # blocks, it applies no RoPE. q/k/v are BF16 in this release; two o projections
        # are FP8 and follow the checkpoint's per-module quant map.
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, [self.q_dim, self.kv_dim, self.kv_dim], has_bias=False
        )
        self.o_proj = _linear(
            args, f"backbone.layers.{layer_id}.mixer.o_proj",
            self.q_dim, config.hidden_size,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = torch.split(
            self.qkv_proj.forward(x), [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
        )
        q = q.contiguous().view(-1, self.num_q, self.head_dim)
        out = ctx.attn_backend.forward(q, k.contiguous(), v.contiguous(), self.layer_id, ctx.batch)
        return self.o_proj.forward(out.reshape(-1, self.q_dim))


class NemotronHMLP(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int, name: str, width: int):
        args = config.nemotron_h_args
        prefix = f"backbone.layers.{layer_id}.mixer.{name}"
        self.up_proj = _linear(args, f"{prefix}.up_proj", config.hidden_size, width)
        self.down_proj = _linear(args, f"{prefix}.down_proj", width, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(F.relu(self.up_proj.forward(x)).square())


class _NemotronRouter(BaseOP):
    def __init__(self, hidden_size: int, num_experts: int):
        self.weight = torch.empty(num_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(num_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(F.linear(x.float(), self.weight.float()))
        return scores, scores + self.e_score_correction_bias


class NemotronHMoE(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int, bank_id: int):
        args = config.nemotron_h_args
        assert args is not None
        self.gate = _NemotronRouter(config.hidden_size, config.num_experts)
        self.fc1_latent_proj = _linear(
            args, f"backbone.layers.{layer_id}.mixer.fc1_latent_proj",
            config.hidden_size, args.moe_latent_size,
        )
        self.experts = make_moe_layer(
            config, layer_id=bank_id, activation="relu2", renormalize=False,
            hidden_size=args.moe_latent_size, intermediate_size=config.moe_intermediate_size,
        )
        self.fc2_latent_proj = _linear(
            args, f"backbone.layers.{layer_id}.mixer.fc2_latent_proj",
            args.moe_latent_size, config.hidden_size,
        )
        self.shared_experts = NemotronHMLP(
            config, layer_id, "shared_experts", args.shared_intermediate_size
        )
        self.top_k = config.num_experts_per_tok
        self.scale = config.routed_scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores, choice = self.gate.forward(x)
        ids = torch.topk(choice, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = (weights * self.scale).float()
        latent = self.fc1_latent_proj.forward(x)
        routed = self.experts.routed_forward(latent, weights, ids.to(torch.int32)).to(x.dtype)
        return self.fc2_latent_proj.forward(routed) + self.shared_experts.forward(x)


class NemotronHBlock(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int, moe_banks: dict[int, int]):
        self._layer_id = layer_id
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        kind = config.nemotron_h_args.layer_types[layer_id]
        if kind == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_id)
        elif kind == "attention":
            self.mixer = NemotronHAttention(config, layer_id)
        elif kind == "moe":
            self.mixer = NemotronHMoE(config, layer_id, moe_banks[layer_id])
        else:
            raise ValueError(f"unsupported Nemotron-H block type {kind!r}")

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer.forward(self.norm.forward(x))


class NemotronHBackbone(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        moe_banks = {layer: bank for bank, layer in enumerate(config.moe_layer_ids or ())}
        self.layers = OPList([
            NemotronHBlock(config, layer, moe_banks) for layer in range(config.num_layers)
        ])
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings.forward(input_ids)
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm_f.forward(x)


class NemotronHForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.backbone = NemotronHBackbone(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.backbone.embeddings if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        hidden = self.backbone.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden)


__all__ = ["NemotronHForCausalLM"]
