from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import safetensors
import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    make_moe_layer,
    silu_and_mul,
    StateLessOP,
    get_rope,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.utils import download_hf_weight, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

    from .args import Qwen4ExpArgs


class _Qwen4MRoPE(StateLessOP):
    """Partial, interleaved temporal/height/width RoPE for Qwen4-Exp."""

    def __init__(self, config: ModelConfig):
        rotary = config.rotary_config
        self._base = get_rope(
            head_dim=rotary.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
        )
        self.mrope_section = tuple(config.qwen4_args.mrope_section)

    @property
    def head_size(self) -> int:
        return self._base.head_size

    @property
    def rotary_dim(self) -> int:
        return self._base.rotary_dim

    @property
    def is_neox(self) -> bool:
        return self._base.is_neox

    @property
    def _cos_sin_cache(self) -> torch.Tensor:
        return self._base._cos_sin_cache

    @_cos_sin_cache.setter
    def _cos_sin_cache(self, value: torch.Tensor) -> None:
        self._base._cos_sin_cache = value

    def apply_inplace(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int | None = None,
    ) -> None:
        head_size = self.head_size if head_size is None else int(head_size)
        if positions.ndim == 1:
            self._base.apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=head_size,
                cos_sin_cache=self._cos_sin_cache,
                is_neox=self.is_neox,
            )
            return
        if query.is_cuda:
            from freetoken.kernel.triton.rope import (
                apply_mrope_with_cos_sin_cache_inplace,
            )

            apply_mrope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=head_size,
                cos_sin_cache=self._cos_sin_cache,
                mrope_section=self.mrope_section,
                is_neox=self.is_neox,
            )
            return

        # CPU reference path for exact unit tests and configuration checks.
        if positions.ndim != 2 or positions.shape != (3, query.shape[0]):
            raise ValueError(
                f"MRoPE positions must have shape (3, {query.shape[0]}), got "
                f"{tuple(positions.shape)}"
            )
        half = self.rotary_dim // 2
        pair = torch.arange(half, device=positions.device)
        axis = torch.zeros(half, dtype=torch.long, device=positions.device)
        axis[(pair % 3 == 1) & (pair < self.mrope_section[1] * 3)] = 1
        axis[(pair % 3 == 2) & (pair < self.mrope_section[2] * 3)] = 2
        selected = positions.long().transpose(0, 1)[:, axis]
        dim = pair.view(1, -1).expand_as(selected)
        cos = self._cos_sin_cache[:, :half][selected, dim]
        sin = self._cos_sin_cache[:, half:][selected, dim]
        for tensor in (query, key):
            heads = tensor.shape[1] // head_size
            view = tensor.view(tensor.shape[0], heads, head_size)
            first = view[..., :half].float().clone()
            second = view[..., half : self.rotary_dim].float().clone()
            view[..., :half].copy_((first * cos[:, None] - second * sin[:, None]).to(view.dtype))
            view[..., half : self.rotary_dim].copy_(
                (second * cos[:, None] + first * sin[:, None]).to(view.dtype)
            )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.apply_inplace(positions, query, key)
        return query, key


class _GroupedRMSNorm(BaseOP):
    def __init__(self, size: int, group_size: int, eps: float):
        if size % group_size:
            raise ValueError(f"RMSNorm size {size} is not divisible by group size {group_size}")
        self.weight = torch.empty(size)
        self.group_size = group_size
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=hidden,
            weight=self.weight,
            bias=None,
            eps=self.eps,
            group_size=self.group_size,
            is_rms_norm=True,
            weight_plus_one=True,
        )


class _GatedRMSNorm(BaseOP):
    def __init__(self, size: int, eps: float, activation: str):
        self.weight = torch.empty(size)
        self.eps = eps
        self.activation = activation

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=hidden,
            weight=self.weight,
            bias=None,
            z=gate,
            eps=self.eps,
            is_rms_norm=True,
            norm_before_gate=True,
            activation=self.activation,
        )


class _GatedResidual(BaseOP):
    def __init__(self, config: ModelConfig, combine: bool = True):
        args: Qwen4ExpArgs = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        hc_size = self.hc_count * self.hidden_size
        self.hc_norm = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.input_mix_weight_down = LinearReplicated(hc_size, args.hc_lowrank, has_bias=False)
        self.input_mix_weight_up = LinearReplicated(args.hc_lowrank, hc_size, has_bias=False)
        self.block_inject_weight = (
            LinearReplicated(hc_size, self.hc_count, has_bias=False) if combine else None
        )

    def forward(self, hyper_input: torch.Tensor):
        normalized = self.hc_norm.forward(hyper_input)
        mix = F.silu(self.input_mix_weight_down.forward(normalized) / self.hc_count)
        mix = torch.sigmoid(self.input_mix_weight_up.forward(mix))
        mix = mix.view(-1, self.hc_count, self.hidden_size)
        mixed = (mix * normalized.view(-1, self.hc_count, self.hidden_size)).mean(dim=1)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * torch.sigmoid(self.block_inject_weight.forward(normalized) / self.hc_count)
        return mixed, hyper_input, inject


class _SharedExpert(BaseOP):
    def __init__(self, config: ModelConfig):
        width = config.shared_expert_intermediate_size
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [width, width], has_bias=False
        )
        self.down_proj = LinearRowParallel(width, config.hidden_size, has_bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden)))


class _SparseMoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        weight_format = "fp8_block" if config.expert_quant == "fp8_block" else "bf16"
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=bool(config.norm_topk_prob),
            weight_format=weight_format,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(config)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate.forward(hidden)
        shared = self.shared_expert.forward(hidden)
        shared *= torch.sigmoid(self.shared_expert_gate.forward(hidden))
        return self.experts.forward(hidden_states=hidden, router_logits=router_logits) + shared


def _shift_right_ignore_eos(tokens: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    if shift == 0:
        return tokens
    positions = torch.arange(tokens.numel(), dtype=torch.long)
    eos_positions = torch.where(tokens == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=0).values
    previous_eos = torch.cat([eos_positions.new_full((1,), -1), previous_eos_inclusive[:-1]])
    segment_start = previous_eos + 1
    source_positions = positions - shift
    shifted = tokens[source_positions.clamp_min(0)]
    valid = (positions - segment_start >= shift) & (source_positions >= 0)
    return torch.where(valid, shifted, tokens.new_full((), eos_token_id))


def build_ngram_ids(
    tokens: torch.Tensor,
    *,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    tokens = tokens.to(dtype=torch.long, device="cpu")
    shifted = [
        _shift_right_ignore_eos(tokens, shift, eos_token_id) for shift in range(ngram_size)
    ]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        stop = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        sizes = vocab_sizes[start:stop]
        heads = torch.remainder(mixed.unsqueeze(-1), sizes)
        blocks.append(heads + offsets[start:stop])
    return torch.cat(blocks, dim=-1)


def _ple_request_tokens(req, forwarded_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Return the complete host token history visible to this forward.

    The overlap scheduler advances ``device_len`` before it drains the prior
    sampled token to ``req.input_ids``. During decode, that one current token is
    already present in ``batch.input_ids``. Join it to the committed host prefix
    so PLE hashes the same history as a non-overlapped forward.
    """
    host_len = req.input_ids.numel()
    if host_len >= req.device_len:
        return req.input_ids[: req.device_len]
    if host_len != req.cached_len:
        raise RuntimeError(
            "Qwen4-Exp PLE host history has an unexpected gap: "
            f"host={host_len}, cached={req.cached_len}, device={req.device_len}"
        )
    if forwarded_ids is None or forwarded_ids.numel() != req.extend_len:
        actual = 0 if forwarded_ids is None else forwarded_ids.numel()
        raise RuntimeError(
            "Qwen4-Exp PLE needs the current forwarded tokens: "
            f"got {actual}, expected {req.extend_len}"
        )
    return torch.cat((req.input_ids[: req.cached_len], forwarded_ids.to(device="cpu")))


class _HostNGramEmbedding(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: Qwen4ExpArgs = config.qwen4_args
        self.layer_id = layer_id
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.eos_token_id = args.eos_token_id
        self.embedding_dim = args.ple_embed_dim
        self.split_ngram_parts = args.split_ngram_parts
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.head_dim = self.embedding_dim // self.ngram_heads
        self.layer_multipliers = torch.empty(args.ngram_size, dtype=torch.long)
        self.ngram_heads_vocab_sizes = torch.empty(self.ngram_heads, dtype=torch.long)
        self.ngram_heads_offsets = torch.empty(self.ngram_heads, dtype=torch.long)
        self._handles = []
        self._shards: list[torch.Tensor] = []
        self._shard_ends = torch.empty(0, dtype=torch.long)
        self._scale = torch.tensor(1.0, dtype=torch.bfloat16)
        self._host_constants: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._dummy = False

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        if dummy:
            self._dummy = True
            return
        folder = download_hf_weight(model_path)
        index_path = os.path.join(folder, "model.safetensors.index.json")
        with open(index_path) as index_file:
            weight_map = json.load(index_file)["weight_map"]
        prefix = (
            f"model.language_model.layers.{self.layer_id}.ple.ple_embedding."
            "ngram_embedding"
        )
        shard_count = len([key for key in weight_map if key.startswith(prefix + ".shard_")])
        if shard_count != self.split_ngram_parts:
            raise RuntimeError(
                f"Qwen4-Exp PLE has {shard_count} shards, expected {self.split_ngram_parts}"
            )
        shard_keys = [f"{prefix}.shard_{shard_id}.weight" for shard_id in range(shard_count)]
        if not shard_keys or any(key not in weight_map for key in shard_keys):
            raise RuntimeError(f"Incomplete Qwen4-Exp PLE shards under {prefix}")

        handles = {}
        shards = []
        for key in shard_keys:
            filename = weight_map[key]
            handle = handles.get(filename)
            if handle is None:
                handle = safetensors.safe_open(
                    os.path.join(folder, filename), framework="pt", device="cpu"
                ).__enter__()
                handles[filename] = handle
            shard = handle.get_tensor(key)
            if shard.dtype != torch.float8_e4m3fn or shard.shape[1] != self.head_dim:
                raise RuntimeError(f"Unexpected PLE shard {key}: {shard.dtype} {tuple(shard.shape)}")
            shards.append(shard.view(torch.uint8))
        scale_key = prefix + ".weight_scale"
        scale_handle = handles.get(weight_map[scale_key])
        if scale_handle is None:
            scale_handle = safetensors.safe_open(
                os.path.join(folder, weight_map[scale_key]), framework="pt", device="cpu"
            ).__enter__()
            handles[weight_map[scale_key]] = scale_handle

        self._handles = list(handles.values())
        self._shards = shards
        self._shard_ends = torch.tensor([shard.shape[0] for shard in shards]).cumsum(0)
        self._scale = scale_handle.get_tensor(scale_key).reshape(())
        self._host_constants = (
            self.layer_multipliers.cpu(),
            self.ngram_heads_vocab_sizes.cpu(),
            self.ngram_heads_offsets.cpu(),
        )
        expected_rows = int(self._host_constants[1][-1] + self._host_constants[2][-1])
        if int(self._shard_ends[-1]) < expected_rows:
            raise RuntimeError(
                f"PLE table has {int(self._shard_ends[-1])} rows, needs {expected_rows}"
            )

    def _current_ngram_ids(self) -> torch.Tensor:
        if self._host_constants is None:
            raise RuntimeError("Qwen4-Exp PLE host weights are not loaded")
        batch = get_global_ctx().batch
        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        multipliers, vocab_sizes, offsets = self._host_constants
        pieces = []
        forwarded_host = None
        forwarded_offset = 0
        for req in reqs:
            extend_len = req.extend_len
            forwarded = None
            if req.input_ids.numel() < req.device_len:
                if forwarded_host is None:
                    forwarded_host = batch.input_ids.detach().to(device="cpu")
                forwarded = forwarded_host[
                    forwarded_offset : forwarded_offset + extend_len
                ]
            tokens = _ple_request_tokens(req, forwarded)
            all_ids = build_ngram_ids(
                tokens,
                ngram_size=self.ngram_size,
                heads_per_ngram=self.heads_per_ngram,
                eos_token_id=self.eos_token_id,
                multipliers=multipliers,
                vocab_sizes=vocab_sizes,
                offsets=offsets,
            )
            pieces.append(all_ids[req.cached_len : req.device_len])
            forwarded_offset += extend_len
        result = torch.cat(pieces, dim=0)
        if result.shape[0] != batch.input_ids.numel():
            raise RuntimeError(
                f"PLE token count {result.shape[0]} does not match batch {batch.input_ids.numel()}"
            )
        return result

    def forward(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._dummy:
            token_count = get_global_ctx().batch.input_ids.numel()
            return torch.zeros(token_count, self.embedding_dim, device=device, dtype=dtype)
        ngram_ids = self._current_ngram_ids().reshape(-1)
        shard_ids = torch.bucketize(ngram_ids, self._shard_ends, right=True)
        output = torch.empty(
            ngram_ids.numel(),
            self.head_dim,
            dtype=torch.uint8,
            pin_memory=torch.cuda.is_available(),
        )
        starts = torch.cat([self._shard_ends.new_zeros(1), self._shard_ends[:-1]])
        for shard_id in shard_ids.unique().tolist():
            positions = torch.nonzero(shard_ids == shard_id, as_tuple=False).flatten()
            local_ids = ngram_ids.index_select(0, positions) - starts[shard_id]
            rows = self._shards[shard_id].index_select(0, local_ids)
            output.index_copy_(0, positions, rows)
        fp8 = output.to(device=device, non_blocking=True).view(torch.float8_e4m3fn)
        embedded = fp8.to(dtype) * self._scale.to(device=device, dtype=dtype)
        return embedded.view(-1, self.embedding_dim)


class _DepthwiseConv(BaseOP):
    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)


class _PLELayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: Qwen4ExpArgs = config.qwen4_args
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_count = args.hc_count
        hc_size = self.hidden_size * self.hc_count
        self.ple_embedding = _HostNGramEmbedding(config, layer_id)
        self.key_proj = LinearReplicated(args.ple_embed_dim, hc_size, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, self.hidden_size, has_bias=False)
        self.norm_key = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.norm_query = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.norm_conv = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.conv1d = _DepthwiseConv(hc_size, args.ple_conv_kernel_size)
        self.dilation = args.ngram_size
        self.state_len = (args.ple_conv_kernel_size - 1) * self.dilation
        self._conv_states: dict[int, torch.Tensor] = {}

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        self.ple_embedding.load_host_weights(model_path, dummy=dummy)

    def _short_conv(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        outputs = []
        offset = 0
        weight = self.conv1d.weight
        for req in reqs:
            length = req.extend_len
            current = hidden[offset : offset + length].transpose(0, 1).unsqueeze(0)
            state = self._conv_states.get(req.table_idx)
            if req.cached_len == 0:
                state = current.new_zeros(1, current.shape[1], self.state_len)
            elif state is None:
                raise RuntimeError(
                    "Qwen4-Exp PLE state cannot resume a radix prefix; serve with --cache-type naive"
                )
            combined = torch.cat([state, current], dim=-1)
            convolved = F.conv1d(
                combined,
                weight,
                groups=weight.shape[0],
                dilation=self.dilation,
            )
            outputs.append(F.silu(convolved).squeeze(0).transpose(0, 1))
            self._conv_states[req.table_idx] = combined[..., -self.state_len :].detach()
            offset += length
        return torch.cat(outputs, dim=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        embeddings = self.ple_embedding.forward(hidden.device, hidden.dtype)
        key = self.norm_key.forward(self.key_proj.forward(embeddings))
        key = key.view(-1, self.hc_count, self.hidden_size)
        value = self.value_proj.forward(embeddings)
        query = self.norm_query.forward(hidden).view(-1, self.hc_count, self.hidden_size)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(1)).flatten(1)
        normalized = self.norm_conv.forward(gated)
        return gated + self._short_conv(normalized)


class _QSAIndexer(BaseOP):
    """Qwen4-Exp's weight-free four-head compressed-key indexer."""

    def __init__(self, config: ModelConfig, rotary):
        args: Qwen4ExpArgs = config.qwen4_args
        self.num_q_heads = args.indexer_n_heads
        self.num_kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.q_dim = self.num_q_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.index_qk_proj = LinearReplicated(
            config.hidden_size, self.q_dim + self.k_dim, has_bias=False
        )
        self.q_layernorm = GemmaPlusOneRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_layernorm = GemmaPlusOneRMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary = rotary
        if self.rotary.rotary_dim > self.head_dim:
            raise ValueError(
                f"QSA index head {self.head_dim} is smaller than rotary dim "
                f"{self.rotary.rotary_dim}"
            )

    def _apply_rope(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        shape = tensor.shape
        flat = tensor.reshape(shape[0], -1).contiguous()
        # The shared RoPE object was built for 256-wide main heads.  Call its
        # kernel with the 128-wide QSA head size instead of using forward(),
        # which would interpret the fused index-query row with the wrong stride.
        dummy_key = torch.zeros(
            shape[0], self.head_dim, dtype=tensor.dtype, device=tensor.device
        )
        self.rotary.apply_inplace(
            positions=positions,
            query=flat,
            key=dummy_key,
            head_size=self.head_dim,
        )
        return flat.view(shape)

    def project(self, hidden: torch.Tensor, positions: torch.Tensor):
        qk = self.index_qk_proj.forward(hidden)
        q_raw, k_raw = torch.split(qk, (self.q_dim, self.k_dim), dim=-1)
        q = q_raw.view(-1, self.num_q_heads, self.head_dim).contiguous()
        k = k_raw.view(-1, self.num_kv_heads, self.head_dim).contiguous()
        q = self.q_layernorm.forward(q)
        q = self._apply_rope(q, positions)
        return q, k

    def normalize_compressed_keys(
        self, keys: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        keys = self.k_layernorm.forward(keys.contiguous())
        return self._apply_rope(keys, positions)


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.rotary = _Qwen4MRoPE(config)
        # Qwen4 stores centered q/k norm weights (effective scale is 1 + w).
        self.q_norm = GemmaPlusOneRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(config.head_dim, config.rms_norm_eps)
        self.indexer = _QSAIndexer(config, self.rotary)

    @nvtx_annotate("QSA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        rope_positions = ctx.batch.rope_positions
        if rope_positions is None:
            rope_positions = ctx.batch.positions
        q, k, v, gate = self._project(x, rope_positions)
        index_q, index_k = self.indexer.project(x, rope_positions)
        output = ctx.attn_backend.qsa_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            self.indexer,
            self.layer_id,
            ctx.batch,
        )
        return self._combine(output, gate)


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        dense_config = replace(config, expert_quant="none", attn_quant="none")
        if self._is_linear:
            group = config.linear_attention_group()
            assert group is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=group.num_key_heads,
                num_v_heads=group.num_value_heads,
                head_k_dim=group.key_head_dim,
                head_v_dim=group.value_head_dim,
                conv_kernel_size=group.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant="none",
                attn_quant="none",
            )
            self.linear_attn.norm = _GatedRMSNorm(
                group.value_head_dim,
                config.rms_norm_eps,
                config.qwen4_args.output_gate_type,
            )
        else:
            self.self_attn = Qwen4ExpAttention(dense_config, layer_id)
        self.mlp = _SparseMoE(config, layer_id)
        self.ple = (
            _PLELayer(config, layer_id)
            if layer_id in config.qwen4_args.ple_layer_ids
            else None
        )
        self.attn_hyper_connection = _GatedResidual(config)
        self.mlp_hyper_connection = _GatedResidual(config)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden)
        mixed, residual, weights = self.attn_hyper_connection.forward(hidden)
        mixed = (
            self.linear_attn.forward(mixed)
            if self._is_linear
            else self.self_attn.forward(mixed)
        )
        hidden = residual + (mixed.unsqueeze(1) * weights.unsqueeze(-1)).flatten(1)
        mixed, residual, weights = self.mlp_hyper_connection.forward(hidden)
        mixed = self.mlp.forward(mixed)
        return residual + (mixed.unsqueeze(1) * weights.unsqueeze(-1)).flatten(1)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = _GatedResidual(config, combine=False)
        self.hc_count = config.qwen4_args.hc_count
        self._image_token_id = config.image_token_id

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.load_host_weights(model_path, dummy=dummy)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens.forward(input_ids)
        mm_embeds = getattr(get_global_ctx().batch, "mm_embeds", None)
        if mm_embeds is not None and self._image_token_id is not None:
            mask = input_ids == self._image_token_id
            slots = int(mask.sum().item())
            if slots != mm_embeds.shape[0]:
                raise ValueError(
                    f"image-token slots ({slots}) do not match vision features "
                    f"({mm_embeds.shape[0]})"
                )
            hidden = hidden.masked_scatter(mask.unsqueeze(-1), mm_embeds.to(hidden.dtype))
        hidden = hidden.repeat(1, self.hc_count)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        return self.hyper_connection_mixer.forward(hidden)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        if config.is_multimodal:
            from .vision import Qwen4VisionModel

            self.visual = Qwen4VisionModel(config.vision_config)
        super().__init__()

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "visual"):
            raise RuntimeError("Qwen4-Exp vision weights are not loaded")
        return self.visual.forward(pixel_values, image_grid_thw)

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        self.model.load_host_weights(model_path, dummy=dummy)

    def forward(self) -> torch.Tensor:
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden)


__all__ = ["Qwen4ExpForCausalLM", "build_ngram_ids"]
