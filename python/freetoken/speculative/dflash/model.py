from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, OPList, RMSNorm, LinearReplicated
from freetoken.layers.rotary import get_rope

from .config import DFlashConfig


def _dflash_causal_context_block_mask(
    context_len: int,
    block_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones((block_len, context_len + block_len), dtype=torch.bool, device=device)
    mask[:, context_len:] = torch.tril(
        torch.ones((block_len, block_len), dtype=torch.bool, device=device)
    )
    return mask


def _dflash_context_block_mask(
    context_len: int,
    block_len: int,
    device: torch.device,
    *,
    layer_type: str,
    is_causal: bool | None,
    sliding_window: int | None,
) -> torch.Tensor | None:
    causal = layer_type == "sliding_attention" if is_causal is None else is_causal
    window = sliding_window if layer_type == "sliding_attention" else None
    if not causal and window is None:
        return None

    query_position = context_len + torch.arange(block_len, device=device)[:, None]
    key_position = torch.arange(context_len + block_len, device=device)[None, :]
    visible = torch.ones((block_len, context_len + block_len), dtype=torch.bool, device=device)
    if causal:
        visible &= key_position <= query_position
    if window is not None:
        visible &= query_position - key_position < window
        if not causal:
            visible &= key_position - query_position < window
    return visible


class _DFlashAttention(BaseOP):
    """Draft attention over committed target context plus a causal draft block."""

    def __init__(self, config: DFlashConfig, layer_id: int):
        self.layer_id = layer_id
        self.head_dim = config.head_dim
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.qo_attn_dim = self.num_qo_heads * self.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim
        self.layer_type = (
            config.layer_types[layer_id]
            if layer_id < len(config.layer_types)
            else "full_attention"
        )
        self.is_causal = config.is_causal
        self.sliding_window = config.sliding_window
        dtype = torch.bfloat16

        self.q_proj = LinearReplicated(config.hidden_size, self.qo_attn_dim, has_bias=False)
        self.k_proj = LinearReplicated(config.hidden_size, self.kv_attn_dim, has_bias=False)
        self.v_proj = LinearReplicated(config.hidden_size, self.kv_attn_dim, has_bias=False)
        self.o_proj = LinearReplicated(self.qo_attn_dim, config.hidden_size, has_bias=False)
        self.q_proj.weight = self.q_proj.weight.to(dtype)
        self.k_proj.weight = self.k_proj.weight.to(dtype)
        self.v_proj.weight = self.v_proj.weight.to(dtype)
        self.o_proj.weight = self.o_proj.weight.to(dtype)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm.weight = self.q_norm.weight.to(dtype)
        self.k_norm.weight = self.k_norm.weight.to(dtype)

        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        # Scratch for the in-place rope kernel's unused counterpart argument;
        # avoids two torch.empty_like allocations per layer per draft call.
        self._rope_scratch_q: torch.Tensor | None = None
        self._rope_scratch_k: torch.Tensor | None = None

    @staticmethod
    def _rope_scratch_for(scratch: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if (
            scratch is None
            or scratch.shape != ref.shape
            or scratch.dtype != ref.dtype
            or scratch.device != ref.device
        ):
            scratch = torch.empty_like(ref)
        return scratch

    def to(self, device):
        """Move all weights and buffers (including rotary cos_sin_cache) to device."""
        for name, param in self.__dict__.items():
            if isinstance(param, torch.Tensor) and param.device.type != device.type:
                setattr(self, name, param.to(device))
            elif hasattr(param, 'weight') and isinstance(getattr(param, 'weight', None), torch.Tensor):
                if param.weight.device.type != device.type:
                    param.weight = param.weight.to(device)
                if getattr(param, 'bias', None) is not None and param.bias.device.type != device.type:
                    param.bias = param.bias.to(device)
            # Move _cos_sin_cache in RotaryEmbedding (private attr, starts with _)
            if isinstance(param, BaseOP):
                for attr_name, attr_val in param.__dict__.items():
                    if isinstance(attr_val, torch.Tensor) and attr_val.device.type != device.type:
                        setattr(param, attr_name, attr_val.to(device))
        return self

    def _apply_rope_inplace(self, positions, query, key):
        if self.rotary._cos_sin_cache.device.type != query.device.type:
            self.rotary._cos_sin_cache = self.rotary._cos_sin_cache.to(query.device)
        from freetoken.kernel.backend import is_flashinfer_installed
        if is_flashinfer_installed():
            from flashinfer import apply_rope_with_cos_sin_cache_inplace as _apply_rope
        else:
            from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace as _apply_rope
        _apply_rope(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_dim,
            cos_sin_cache=self.rotary._cos_sin_cache,
            is_neox=self.rotary.is_neox,
        )

    def project_context_kv(self, context, positions):
        context_k = self.k_proj.forward(context).view(-1, self.num_kv_heads, self.head_dim)
        context_v = self.v_proj.forward(context).view(-1, self.num_kv_heads, self.head_dim)
        self.k_norm.forward_inplace(context_k)
        context_k_flat = context_k.reshape(-1, self.kv_attn_dim)
        self._apply_rope_inplace(
            positions,
            torch.empty_like(context_k_flat),
            context_k_flat,
        )
        return context_k_flat.view(-1, self.num_kv_heads, self.head_dim), context_v

    def forward(self, hidden_states, context, positions, context_kv=None, attn_mask=None):
        q = self.q_proj.forward(hidden_states)
        if context_kv is None:
            context_positions = torch.arange(
                context.shape[0], dtype=positions.dtype, device=positions.device
            )
            context_k, context_v = self.project_context_kv(context, context_positions)
        else:
            context_k, context_v = context_kv
        block_k = self.k_proj.forward(hidden_states)
        block_v = self.v_proj.forward(hidden_states)

        q = q.view(-1, self.num_qo_heads, self.head_dim)
        block_k = block_k.view(-1, self.num_kv_heads, self.head_dim)
        block_v = block_v.view(-1, self.num_kv_heads, self.head_dim)

        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(block_k)

        q_flat = q.reshape(-1, self.qo_attn_dim)
        self._rope_scratch_q = self._rope_scratch_for(self._rope_scratch_q, q_flat)
        self._apply_rope_inplace(positions, q_flat, self._rope_scratch_q)
        q = q_flat.view(-1, self.num_qo_heads, self.head_dim)

        block_k_flat = block_k.reshape(-1, self.kv_attn_dim)
        self._rope_scratch_k = self._rope_scratch_for(self._rope_scratch_k, block_k_flat)
        self._apply_rope_inplace(
            positions,
            self._rope_scratch_k,
            block_k_flat,
        )
        block_k = block_k_flat.view(-1, self.num_kv_heads, self.head_dim)

        group_size = self.num_qo_heads // self.num_kv_heads
        k_expanded = torch.cat([context_k, block_k], dim=0).repeat_interleave(group_size, dim=1)
        v_expanded = torch.cat([context_v, block_v], dim=0).repeat_interleave(group_size, dim=1)

        # SDPA expects [batch, num_heads, seq, head_dim]
        q_t = q.transpose(0, 1).unsqueeze(0)  # [1, num_qo, block_size, head_dim]
        k_t = k_expanded.transpose(0, 1).unsqueeze(0)  # [1, num_qo, ctx_len + block_size, head_dim]
        v_t = v_expanded.transpose(0, 1).unsqueeze(0)
        if attn_mask is None:
            attn_mask = _dflash_context_block_mask(
                context_len=context_k.shape[0],
                block_len=hidden_states.shape[0],
                device=hidden_states.device,
                layer_type=self.layer_type,
                is_causal=self.is_causal,
                sliding_window=self.sliding_window,
            )
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        scale = self.head_dim ** -0.5
        attn = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, scale=scale)
        attn = attn.squeeze(0).transpose(0, 1)

        out = attn.reshape(-1, self.qo_attn_dim)
        return self.o_proj.forward(out)


class _DFlashMLP(BaseOP):
    """Standard SwiGLU MLP for draft model."""

    def __init__(self, config: DFlashConfig):
        dtype = torch.bfloat16
        self.gate_proj = LinearReplicated(config.hidden_size, config.intermediate_size, has_bias=False)
        self.up_proj = LinearReplicated(config.hidden_size, config.intermediate_size, has_bias=False)
        self.down_proj = LinearReplicated(config.intermediate_size, config.hidden_size, has_bias=False)
        self.gate_proj.weight = self.gate_proj.weight.to(dtype)
        self.up_proj.weight = self.up_proj.weight.to(dtype)
        self.down_proj.weight = self.down_proj.weight.to(dtype)

    def to(self, device):
        """Move all weights and buffers to device."""
        for name, param in self.__dict__.items():
            if isinstance(param, torch.Tensor) and param.device.type != device.type:
                setattr(self, name, param.to(device))
            elif isinstance(param, BaseOP):
                param.to(device)
        return self

    def forward(self, x):
        return self.down_proj.forward(F.silu(self.gate_proj.forward(x)) * self.up_proj.forward(x))


class _DFlashDecoderLayer(BaseOP):
    """One draft decoder layer: cross-attention + MLP with pre-norm."""

    def __init__(self, config: DFlashConfig, layer_id: int):
        self.self_attn = _DFlashAttention(config, layer_id)
        self.mlp = _DFlashMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm.weight = self.input_layernorm.weight.to(torch.bfloat16)
        self.post_attention_layernorm.weight = self.post_attention_layernorm.weight.to(torch.bfloat16)

    def to(self, device):
        """Move all weights and buffers to device."""
        for name, param in self.__dict__.items():
            if isinstance(param, torch.Tensor) and param.device.type != device.type:
                setattr(self, name, param.to(device))
            elif isinstance(param, BaseOP):
                param.to(device)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        positions: torch.Tensor,
        context_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm cross-attention
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(
            hidden_states, context, positions, context_kv=context_kv, attn_mask=attn_mask
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashDraftModel(BaseOP):
    """DFlash draft model: lightweight block-diffusion model for speculative decoding.

    Predicts an entire block of tokens in parallel using cross-attention to the
    target model's intermediate hidden states. Borrows the target model's
    embedding and LM head.
    """

    def __init__(self, config: DFlashConfig):
        self.config = config
        dtype = torch.bfloat16
        # Project concatenated target hidden states -> hidden_size
        self.fc = LinearReplicated(config.context_dim, config.hidden_size, has_bias=False)
        self.fc.weight = self.fc.weight.to(dtype)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm.weight = self.hidden_norm.weight.to(dtype)
        # Draft layers
        self.layers = OPList(
            [_DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm.weight = self.norm.weight.to(dtype)

    def to(self, device):
        """Move all weights and buffers (including rotary cos_sin_cache) to device."""
        def _move(obj):
            if isinstance(obj, torch.Tensor) and obj.device.type != device.type:
                return obj.to(device)
            if isinstance(obj, list):
                return [_move(x) for x in obj]
            if isinstance(obj, BaseOP):
                for attr_name, attr_val in obj.__dict__.items():
                    new_val = _move(attr_val)
                    if new_val is not attr_val:
                        setattr(obj, attr_name, new_val)
            return obj

        for name, param in self.__dict__.items():
            new_val = _move(param)
            if new_val is not param:
                setattr(self, name, new_val)
        return self

    def project_context_features(self, context_features: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm.forward(self.fc.forward(context_features))

    def forward(
        self,
        mask_embeds: torch.Tensor,    # [block_size, hidden] — embedded mask tokens (from target's embedding)
        context_features: torch.Tensor,  # [ctx_len, num_target_layers * hidden] — target hidden states
        positions: torch.Tensor,      # [block_size] — positions for RoPE
        context_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> torch.Tensor:
        """Run draft model forward, return hidden states [block_size, hidden].

        The caller applies the target model's LM head to get logits.
        """
        use_context_kv_cache = (
            context_kv_cache is not None
            and len(context_kv_cache) == len(self.layers.op_list)
            and all(kv is not None for kv in context_kv_cache)
        )
        context = None if use_context_kv_cache else self.project_context_features(context_features)

        # The attention mask depends only on (context_len, block_len, layer kind);
        # build it once per kind instead of once per layer.
        if use_context_kv_cache:
            context_len = context_kv_cache[0][0].shape[0]
        else:
            context_len = context.shape[0]
        masks: dict[tuple, torch.Tensor | None] = {}
        for layer in self.layers.op_list:
            attn = layer.self_attn
            key = (attn.layer_type, attn.is_causal, attn.sliding_window)
            if key not in masks:
                masks[key] = _dflash_context_block_mask(
                    context_len,
                    mask_embeds.shape[0],
                    mask_embeds.device,
                    layer_type=attn.layer_type,
                    is_causal=attn.is_causal,
                    sliding_window=attn.sliding_window,
                )

        # Run through layers
        h = mask_embeds
        for i, layer in enumerate(self.layers.op_list):
            context_kv = context_kv_cache[i] if use_context_kv_cache else None
            attn = layer.self_attn
            key = (attn.layer_type, attn.is_causal, attn.sliding_window)
            h = layer.forward(h, context, positions, context_kv=context_kv, attn_mask=masks[key])

        return self.norm.forward(h)


__all__ = ["DFlashDraftModel", "_dflash_causal_context_block_mask"]
