from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearOProj
from freetoken.layers.rotary import get_rope
from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# V heads carry the GROUP's v_head_dim (config.v_head_dim / swa_v_head_dim,
# modeling_mimo_v2.py:240,244,249 - 128 in supplied release metadata, 64 in test fixtures).
# o_proj input is num_qo_heads * v_head_dim, NOT num_qo_heads * head_dim.
def pad_value_heads(
    v: torch.Tensor,
    num_kv_heads: int,
    v_head_dim: int,
    head_dim: int,
    value_scale: float | None = None,
) -> torch.Tensor:
    """Scale V (modeling_mimo_v2.py:302-303) then zero-pad each head to ``head_dim``.

    Returns rank-2 ``[T, num_kv_heads * head_dim]``: the KV pool store requires the
    flat ``{L, D}`` layout (kernel/csrc/jit/store.cu:79 ``TensorMatcher({L, D})``,
    reached via kvcache/mha_pool.py:130 and hybrid_swa_pool.py:217), and the backend
    re-views it as ``[T, kv_heads, head_dim]`` (attention/triton.py:204).
    """
    T = v.shape[0]
    v = v.view(T, num_kv_heads, v_head_dim)
    if value_scale is not None:
        v = v * value_scale
    return (
        F.pad(v, (0, head_dim - v_head_dim))
        .contiguous()
        .view(T, -1)
    )


def unpad_attention_output(
    o: torch.Tensor,
    num_qo_heads: int,
    head_dim: int,
    v_head_dim: int,
) -> torch.Tensor:
    """Slice the padded tail off the attention output: ``[T, q, head_dim] -> [T, q * v_head_dim]``.

    The backend returns ``torch.empty_like(q)``, i.e. rank-3 ``[T, num_qo_heads, head_dim]``
    (kernel/triton/attention.py:391, 817, 945); the padded lanes are exactly zero
    (p @ [k128|0]) and the result is head-major flat for o_proj.
    """
    T = o.shape[0]
    return o.reshape(T, num_qo_heads, head_dim)[..., :v_head_dim].reshape(
        T, num_qo_heads * v_head_dim
    )


class MiMoV2Attention(BaseOP):
    """MiMo v2 attention for one full-context or SWA layer (single-device / TP=1).

    Fused ``qkv_proj`` splits into ``[Q(kv*head_dim) | K | V]`` with V heads 128-dim;
    V is scaled by ``attention_value_scale`` then zero-padded to the group ``head_dim``
    192 in the supplied release metadata so both groups share the 192-wide KV pool, and the
    attention output is sliced back to ``[..., :128]`` before ``o_proj``. Partial NeoX
    rope (first 64 of 192) runs with the group's own theta; SWA layers add fp32 softmax
    ``sinks`` and the 128-token sliding window. These describe the port's layout,
    not verified equivalence to a complete upstream model.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        if not isinstance(group, (FullAttentionGroupConfig, SWAAttentionGroupConfig)):
            raise ValueError(f"MiMo v2 attention does not support {group.kind!r} layers")
        self.is_swa = isinstance(group, SWAAttentionGroupConfig)
        self.head_dim = group.head_dim
        # 0 (unset) -> V is head_dim-wide, i.e. no pad (the reference's own default).
        self.v_head_dim = group.v_head_dim or self.head_dim
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = group.num_kv_heads
        self.qo_attn_dim = self.num_qo_heads * self.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim
        self.v_attn_dim = self.num_kv_heads * self.v_head_dim

        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size,
            [self.qo_attn_dim, self.kv_attn_dim, self.v_attn_dim],
            has_bias=config.has_attn_bias,
            quant_config=config.quant,
            prefix=f"{prefix}.qkv_proj",
        )
        # o_proj eats the 128-dim V heads, NOT num_qo * head_dim (modeling_mimo_v2.py:266,288).
        self.o_proj = LinearOProj(
            input_size=self.num_qo_heads * self.v_head_dim,
            output_size=config.hidden_size,
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.o_proj",
        )
        # SWA-only (add_full_attention_sink_bias=False); the engine name is `sinks`, fp32.
        # Declaration-only placeholder (gpt_oss attention.py:48): the loader yields device-placed
        # fp32 sinks and BaseOP.load_state_dict SETATTRS them over this attr (layers/base.py:46),
        # so forward below must read the live `self.sinks` at call/capture time, never a spec
        # cached at construction (that pinned the CPU placeholder -> triton sinks.is_cuda assert).
        self.sinks = torch.empty(self.num_qo_heads, dtype=torch.float32) if self.is_swa else None
        self.sliding_window = group.sliding_window if self.is_swa else None
        self.sm_scale = self.head_dim**-0.5
        self.value_scale = config.attention_value_scale

        rotary_config = group.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=(
                tuple(rotary_config.scaling.items())
                if rotary_config.scaling
                else None
            ),
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        T = x.shape[0]

        qkv = self.qkv_proj.forward(x)
        # split on the layer's TP-local segment sizes (the construction list is pre-shard)
        q, k, v = qkv.split(self.qkv_proj.output_sizes, dim=-1)
        del x, qkv

        # scale FIRST (HF scales before the cache update, modeling_mimo_v2.py:302-314),
        # then pad V to the group head_dim; the pool and kernels stay 192-wide.
        # Rank-2 [T, kv*192]: the store kernel matches {L, D} (store.cu:79).
        v = pad_value_heads(
            v, self.num_kv_heads, self.v_head_dim, self.head_dim, self.value_scale
        )

        q = q.view(T, -1).contiguous()
        k = k.view(T, -1).contiguous()
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(T, self.num_qo_heads, self.head_dim)

        o = ctx.attn_backend.forward(
            q,
            k,
            v,
            self.layer_id,
            ctx.batch,
            # built per call (gpt_oss attention.py:88-92): reads the LOAD-TIME sinks attr, so the
            # CUDA-graph capture and every eager call carry the device tensor load_state_dict set.
            attn_spec=AttentionSpec(
                sliding_window=self.sliding_window,
                sm_scale=self.sm_scale,
                sinks=self.sinks,
            ),
        )
        # drop the padded tail (exactly zero: p @ [k128|0]) before o_proj
        o = unpad_attention_output(o, self.num_qo_heads, self.head_dim, self.v_head_dim)
        return self.o_proj.forward(o)


__all__ = ["MiMoV2Attention", "pad_value_heads", "unpad_attention_output"]
