"""Validated Kimi-K3 text-tower geometry.

The public checkpoint numbers its ``linear_attn_config`` layer lists from one.
FreeToken numbers decoder layers from zero, so conversion happens once here and
the normalized tuples are shared by config, model, cache, and weight loading.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass(frozen=True)
class KimiK3Args:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    kda_gate_lower_bound: float | None
    kda_full_rank_gate: bool
    kda_layer_ids: tuple[int, ...]
    mla_layer_ids: tuple[int, ...]
    routed_expert_hidden_size: int
    latent_moe_use_norm: bool
    situ_beta: float
    situ_linear_beta: float
    attn_res_block_size: int
    mla_use_nope: bool
    mla_use_output_gate: bool

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mla_latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


def _zero_based_layers(raw: Any, *, key: str, num_layers: int) -> tuple[int, ...]:
    values = tuple(int(x) for x in (raw or ()))
    if not values:
        raise ValueError(f"Kimi-K3 linear_attn_config.{key} is empty")
    if min(values) < 1 or max(values) > num_layers:
        raise ValueError(
            f"Kimi-K3 {key} must use one-based layer ids in [1, {num_layers}], "
            f"got [{min(values)}, {max(values)}]"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"Kimi-K3 {key} contains duplicate layer ids")
    return tuple(x - 1 for x in values)


def load_args(text: Any) -> KimiK3Args:
    num_layers = int(_value(text, "num_hidden_layers"))
    linear = _value(text, "linear_attn_config") or {}
    kda_ids = _zero_based_layers(
        _value(linear, "kda_layers"), key="kda_layers", num_layers=num_layers
    )
    mla_ids = _zero_based_layers(
        _value(linear, "full_attn_layers"),
        key="full_attn_layers",
        num_layers=num_layers,
    )
    if set(kda_ids) & set(mla_ids):
        raise ValueError("Kimi-K3 KDA and MLA layer sets overlap")
    if set(kda_ids) | set(mla_ids) != set(range(num_layers)):
        missing = sorted(set(range(num_layers)) - set(kda_ids) - set(mla_ids))
        raise ValueError(
            f"Kimi-K3 attention layer map is incomplete; missing {missing}"
        )

    args = KimiK3Args(
        hidden_size=int(_value(text, "hidden_size")),
        num_heads=int(_value(text, "num_attention_heads")),
        num_kv_heads=int(_value(text, "num_key_value_heads")),
        num_layers=num_layers,
        q_lora_rank=int(_value(text, "q_lora_rank")),
        kv_lora_rank=int(_value(text, "kv_lora_rank")),
        qk_nope_head_dim=int(_value(text, "qk_nope_head_dim")),
        qk_rope_head_dim=int(_value(text, "qk_rope_head_dim")),
        v_head_dim=int(_value(text, "v_head_dim")),
        kda_num_heads=int(_value(linear, "num_heads")),
        kda_head_dim=int(_value(linear, "head_dim")),
        kda_conv_kernel=int(_value(linear, "short_conv_kernel_size")),
        kda_gate_lower_bound=(
            None
            if _value(linear, "gate_lower_bound") is None
            else float(_value(linear, "gate_lower_bound"))
        ),
        kda_full_rank_gate=bool(_value(linear, "use_full_rank_gate")),
        kda_layer_ids=kda_ids,
        mla_layer_ids=mla_ids,
        routed_expert_hidden_size=int(_value(text, "routed_expert_hidden_size")),
        latent_moe_use_norm=bool(_value(text, "latent_moe_use_norm")),
        situ_beta=float(_value(text, "activation_situ_beta")),
        situ_linear_beta=float(_value(text, "activation_situ_linear_beta")),
        attn_res_block_size=int(_value(text, "attn_res_block_size")),
        mla_use_nope=bool(_value(text, "mla_use_nope")),
        mla_use_output_gate=bool(_value(text, "mla_use_output_gate")),
    )

    # These are kernel contracts, not merely model-card metadata. Fail a future
    # checkpoint variant before allocating its weights.  The second geometry is
    # the small Kimi-K3 development checkpoint published by
    # inference-optimization; it exercises the same KDA/MLA and latent-MoE paths
    # without requiring the full model's multi-terabyte host.
    if args.kda_num_heads != args.num_heads or args.num_kv_heads != args.num_heads:
        raise ValueError("Kimi-K3 support requires independent KDA/attention heads")
    supported_geometries = {
        (93, 7168, 96, 128, 12, 3584),
        (8, 1024, 8, 32, 4, 512),
    }
    geometry = (
        args.num_layers,
        args.hidden_size,
        args.num_heads,
        args.kda_head_dim,
        args.attn_res_block_size,
        args.routed_expert_hidden_size,
    )
    if geometry not in supported_geometries:
        raise ValueError(f"unsupported Kimi-K3 kernel geometry: {geometry}")
    if args.kda_conv_kernel != 4:
        raise ValueError("Kimi-K3 KDA kernels require conv kernel=4")
    if not args.kda_full_rank_gate:
        raise ValueError("Kimi-K3 support requires the full-rank KDA gate")
    if not args.mla_use_nope or not args.mla_use_output_gate:
        raise ValueError("Kimi-K3 support requires NoPE MLA with its output gate")
    if (args.situ_beta, args.situ_linear_beta) != (4.0, 25.0):
        raise ValueError("Kimi-K3 SiTU kernels require beta=4 and linear_beta=25")
    return args


__all__ = ["KimiK3Args", "load_args"]
