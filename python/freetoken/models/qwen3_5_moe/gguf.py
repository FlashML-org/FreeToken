"""Native-GGUF adapter for Qwen3.5-MoE (GGUF architecture ``qwen35moe``).

Ornith-1.5-35B-A3B uses the standard Qwen3.5 hybrid decoder: three Gated
DeltaNet layers followed by one full-attention layer, 256 top-8 routed experts,
and one gated shared expert.  Dense matrices remain in their per-tensor GGUF
Q4_K/Q6_K representation; routed experts are streamed through the mixed-GGUF
offload cache.  The final GGUF block is the optional MTP predictor and is not
part of autoregressive serving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _g(shim: "GgufConfigShim", key: str):
    full = f"qwen35moe.{key}"
    value = shim.metadata.get(full)
    if value is None:
        raise KeyError(f"missing GGUF metadata key {full}")
    return value


def _main_layer_count(shim: "GgufConfigShim") -> int:
    # llama.cpp includes MTP predictor blocks in block_count and records how many
    # there are separately.  Transformers' num_hidden_layers excludes them.
    total = int(_g(shim, "block_count"))
    mtp = int(shim.metadata.get("qwen35moe.nextn_predict_layers", 0))
    if mtp < 0 or mtp >= total:
        raise ValueError(f"invalid qwen35moe nextn_predict_layers={mtp} for {total} blocks")
    return total - mtp


def _expert_types(shim: "GgufConfigShim") -> tuple[tuple[int, int], ...] | None:
    from freetoken.models.gguf.reader import gguf_tensor_names

    if not gguf_tensor_names(shim.model_path):
        return None
    types = gguf_tensor_types(shim.model_path)
    out = []
    for layer in range(_main_layer_count(shim)):
        gate = types[f"blk.{layer}.ffn_gate_exps.weight"]
        up = types[f"blk.{layer}.ffn_up_exps.weight"]
        down = types[f"blk.{layer}.ffn_down_exps.weight"]
        if gate != up:
            raise ValueError(f"blk.{layer}: gate/up expert banks have different GGUF types")
        out.append((gate, down))
    return tuple(out)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    layers = _main_layer_count(shim)
    hidden = int(_g(shim, "embedding_length"))
    context = int(_g(shim, "context_length"))
    num_q = int(_g(shim, "attention.head_count"))
    num_kv = int(_g(shim, "attention.head_count_kv"))
    head_dim = int(_g(shim, "attention.key_length"))
    interval = int(_g(shim, "full_attention_interval"))
    full_ids = tuple(i for i in range(layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(layers) if i not in full_ids)

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(_g(shim, "rope.dimension_count")),
        max_position=context,
        base=float(_g(shim, "rope.freq_base")),
        scaling=None,
    )
    state_dim = int(_g(shim, "ssm.state_size"))
    inner = int(_g(shim, "ssm.inner_size"))
    value_heads = inner // state_dim
    if inner % state_dim:
        raise ValueError(f"qwen35moe ssm.inner_size {inner} is not divisible by {state_dim}")

    from freetoken.models.gguf.reader import gguf_tensor_type

    return ModelConfig(
        num_layers=layers,
        num_qo_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(_g(shim, "attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(_g(shim, "expert_count")),
        num_experts_per_tok=int(_g(shim, "expert_used_count")),
        moe_intermediate_size=int(_g(shim, "expert_feed_forward_length")),
        shared_expert_intermediate_size=int(_g(shim, "expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        moe_enabled=True,
        use_qk_norm=True,
        model_type="qwen3_5_moe",
        architectures=list(shim.architectures),
        attention_groups=(
            LinearGatedDeltaGroupConfig(
                name="linear",
                layer_ids=linear_ids,
                num_key_heads=int(_g(shim, "ssm.group_count")),
                num_value_heads=value_heads,
                key_head_dim=state_dim,
                value_head_dim=state_dim,
                conv_kernel_dim=int(_g(shim, "ssm.conv_kernel")),
                output_gate=True,
            ),
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=num_kv,
                head_dim=head_dim,
                rotary_config=rotary,
            ),
        ),
        expert_quant="gguf",
        moe_weight_format="gguf",
        gguf_embed_quant=gguf_tensor_type(shim.model_path, "token_embd.weight"),
        gguf_expert_types=_expert_types(shim),
        gguf_model_path=shim.model_path,
    )


def gguf_tensor_types(model_path: str) -> dict[str, int]:
    from freetoken.models.gguf.reader import iter_gguf_tensors

    return {t.name: t.ggml_type for t in iter_gguf_tensors(model_path)}


def _dense(t, dtype=torch.bfloat16) -> torch.Tensor:
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, dtype).reshape(t.shape)


def _inverse_v_permutation(config: ModelConfig, *, head_dim: int) -> torch.Tensor:
    """Undo llama.cpp's grouped->tiled V-head reorder.

    Qwen's HF/FreeToken GDN layout groups the two V heads belonging to each K
    head. llama.cpp rewrites them to tiled order before GGUF quantization so its
    broadcast can use ``ggml_repeat``. FreeToken's FLA kernels consume the
    original grouped order, therefore every affected GGUF tensor must be mapped
    back (see llama.cpp ``_LinearAttentionVReorderBase``).
    """
    group = config.linear_attention_group()
    assert group is not None
    num_k, num_v = group.num_key_heads, group.num_value_heads
    per_k = num_v // num_k
    perm = torch.arange(num_v * head_dim).reshape(num_k, per_k, head_dim)
    perm = perm.permute(1, 0, 2).reshape(-1)
    return torch.argsort(perm)


def _undo_v_rows(tensor: torch.Tensor, config: ModelConfig, head_dim: int) -> torch.Tensor:
    return tensor.index_select(0, _inverse_v_permutation(config, head_dim=head_dim))


_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
):
    """Yield the 40 serving layers while retaining every matrix in native GGUF form."""
    from freetoken.distributed import get_tp_info
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    if get_tp_info().size > 1:
        raise NotImplementedError("qwen35moe GGUF currently supports TP=1 only")
    assert not include_moe_experts, "qwen35moe GGUF experts require the offload backend"
    assert include_non_moe
    config = parse_gguf_config(cached_load_hf_config(model_path))
    shared_buf: dict[int, dict[str, tuple[torch.Tensor, int]]] = {}

    packed = {
        "attn_q.weight": "self_attn.qkv_proj.q.qweight",
        "attn_k.weight": "self_attn.qkv_proj.k.qweight",
        "attn_v.weight": "self_attn.qkv_proj.v.qweight",
        "attn_output.weight": "self_attn.o_proj.qweight",
        "attn_qkv.weight": "linear_attn.in_proj.qkv.qweight",
        "attn_gate.weight": "linear_attn.in_proj.z.qweight",
        "ssm_beta.weight": "linear_attn.in_proj.b.qweight",
        "ssm_alpha.weight": "linear_attn.in_proj.a.qweight",
        "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.qweight",
    }
    dense = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
    }

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _dense(t)
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if not name.startswith("blk."):
            raise ValueError(f"unmapped qwen35moe GGUF tensor: {name}")
        layer = int(name.split(".")[1])
        if layer >= config.num_layers:
            continue  # MTP predictor block
        suffix = name.split(".", 2)[2]
        if suffix in _EXPERT_SUFFIXES:
            continue
        base = f"model.layers.{layer}"
        if suffix in packed:
            weight = t.packed()
            group = config.linear_attention_group()
            assert group is not None
            if suffix == "attn_qkv.weight":
                qk_rows = 2 * group.num_key_heads * group.key_head_dim
                qk, value = weight[:qk_rows], weight[qk_rows:]
                weight = torch.cat(
                    [qk, _undo_v_rows(value, config, group.value_head_dim)], dim=0
                )
            elif suffix == "attn_gate.weight":
                weight = _undo_v_rows(weight, config, group.value_head_dim)
            elif suffix in ("ssm_beta.weight", "ssm_alpha.weight"):
                weight = _undo_v_rows(weight, config, 1)
            yield f"{base}.{packed[suffix]}", weight
        elif suffix in dense:
            yield f"{base}.{dense[suffix]}", _dense(t)
        elif suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _dense(t).reshape(1, -1)
        elif suffix == "ssm_conv1d.weight":
            conv = _dense(t)
            group = config.linear_attention_group()
            assert group is not None
            qk_rows = 2 * group.num_key_heads * group.key_head_dim
            conv = torch.cat(
                [conv[:qk_rows], _undo_v_rows(conv[qk_rows:], config, group.value_head_dim)],
                dim=0,
            )
            yield f"{base}.linear_attn.conv1d.weight", conv.unsqueeze(1)
        elif suffix == "ssm_dt.bias":
            value = _undo_v_rows(_dense(t, torch.float32).reshape(-1, 1), config, 1)
            yield f"{base}.linear_attn.dt_bias", value.reshape(-1)
        elif suffix == "ssm_a":
            # llama.cpp stores the already transformed continuous-time A=-exp(A_log).
            a = _dense(t, torch.float32)
            if not bool((a < 0).all()):
                raise ValueError(f"{name}: expected negative transformed SSM A values")
            a = _undo_v_rows(a.reshape(-1, 1), config, 1).reshape(-1)
            yield f"{base}.linear_attn.A_log", torch.log(-a)
        elif suffix == "ssm_out.weight":
            # llama.cpp permutes this matrix's input columns before quantizing.
            # K-quants group adjacent input columns, so a lossless packed-byte
            # permutation is impossible after quantization; dequantize this one
            # projection and restore the original columns in bf16.
            out = _dense(t)
            group = config.linear_attention_group()
            assert group is not None
            inv = _inverse_v_permutation(config, head_dim=group.value_head_dim)
            yield f"{base}.linear_attn.out_proj.weight", out.index_select(1, inv)
        elif suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            slot = "gate" if "gate" in suffix else "up"
            shared_buf.setdefault(layer, {})[slot] = (t.packed(), t.ggml_type)
            parts = shared_buf[layer]
            if len(parts) == 2:
                if parts["gate"][1] != parts["up"][1]:
                    raise ValueError(f"blk.{layer}: shared-expert gate/up GGUF types differ")
                yield f"{base}.mlp.shared_expert.gate_up_proj.qweight", torch.cat(
                    [parts["gate"][0], parts["up"][0]], dim=0
                )
                del shared_buf[layer]
        else:
            raise ValueError(f"unmapped qwen35moe GGUF tensor: {name}")
    assert not shared_buf, f"incomplete shared-expert gate/up groups: {sorted(shared_buf)}"


class DeferredGGUFLinear(BaseOP):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type: int | None = None
        self.qweight: torch.Tensor | None = None

    def materialize(self, quant_type: int) -> None:
        from freetoken.models.gguf.dequant import row_bytes

        self._quant_type = quant_type
        self.qweight = torch.empty(
            self.out_features, row_bytes(self.in_features, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.layers.gguf import fused_mul_mat_gguf

        assert self.qweight is not None and self._quant_type is not None
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


class GGUFSplitLinear(BaseOP):
    """One logical fused projection backed by independently quantized GGUF parts."""

    def __init__(self, in_features: int, parts: tuple[tuple[str, int], ...]):
        self._part_names = tuple(name for name, _ in parts)
        for name, out_features in parts:
            setattr(self, name, DeferredGGUFLinear(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([getattr(self, name).forward(x) for name in self._part_names], dim=-1)


class GGUFOutputHead(DeferredGGUFLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return super().forward(x)


def is_gguf_model(config: ModelConfig) -> bool:
    return config.gguf_model_path is not None and config.expert_quant == "gguf"


def convert_qwen3_5_to_gguf(model, config: ModelConfig) -> None:
    from freetoken.layers.gguf import GGUFEmbedding

    if not config.gguf_model_path or config.gguf_embed_quant is None:
        raise NotImplementedError("qwen35moe FTW conversion is not yet supported")
    types = gguf_tensor_types(config.gguf_model_path)

    def materialize(module: DeferredGGUFLinear, name: str) -> None:
        module.materialize(types[name])

    model.model.embed_tokens = GGUFEmbedding(
        config.vocab_size, config.hidden_size, config.gguf_embed_quant
    )
    for i, layer in enumerate(model.model.layers.op_list):
        if layer._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            qkv = 2 * g.num_key_heads * g.key_head_dim + g.num_value_heads * g.value_head_dim
            z = g.num_value_heads * g.value_head_dim
            layer.linear_attn.in_proj = GGUFSplitLinear(
                config.hidden_size,
                (("qkv", qkv), ("z", z), ("b", g.num_value_heads), ("a", g.num_value_heads)),
            )
            for part, suffix in (
                ("qkv", "attn_qkv.weight"),
                ("z", "attn_gate.weight"),
                ("b", "ssm_beta.weight"),
                ("a", "ssm_alpha.weight"),
            ):
                materialize(getattr(layer.linear_attn.in_proj, part), f"blk.{i}.{suffix}")
            # ssm_out was column-permuted before GGUF quantization. The loader
            # dequantizes and restores it to the ordinary bf16 module.
        else:
            q_out = 2 * config.num_qo_heads * config.head_dim
            kv_out = config.num_kv_heads * config.head_dim
            layer.self_attn.qkv_proj = GGUFSplitLinear(
                config.hidden_size, (("q", q_out), ("k", kv_out), ("v", kv_out))
            )
            for part, suffix in (
                ("q", "attn_q.weight"),
                ("k", "attn_k.weight"),
                ("v", "attn_v.weight"),
            ):
                materialize(getattr(layer.self_attn.qkv_proj, part), f"blk.{i}.{suffix}")
            layer.self_attn.o_proj = DeferredGGUFLinear(
                config.num_qo_heads * config.head_dim, config.hidden_size
            )
            materialize(layer.self_attn.o_proj, f"blk.{i}.attn_output.weight")

        shared = layer.mlp.shared_expert
        shared.gate_up_proj = DeferredGGUFLinear(
            config.hidden_size, 2 * config.shared_expert_intermediate_size
        )
        shared.down_proj = DeferredGGUFLinear(
            config.shared_expert_intermediate_size, config.hidden_size
        )
        materialize(shared.gate_up_proj, f"blk.{i}.ffn_gate_shexp.weight")
        materialize(shared.down_proj, f"blk.{i}.ffn_down_shexp.weight")

    model.lm_head = GGUFOutputHead(config.hidden_size, config.vocab_size)
    materialize(model.lm_head, "output.weight")


def _expert_bank_geometry(config: ModelConfig) -> tuple[int, int]:
    from freetoken.models.gguf.dequant import row_bytes

    assert config.gguf_expert_types
    h, inter = config.hidden_size, config.moe_intermediate_size
    gu = max(2 * inter * row_bytes(h, t) for t, _ in config.gguf_expert_types)
    down = max(h * row_bytes(inter, t) for _, t in config.gguf_expert_types)
    align = lambda n: (n + 63) // 64 * 64
    return align(gu), align(down)


def load_gguf_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    from freetoken.distributed import get_tp_info
    from freetoken.models.gguf.dequant import row_bytes
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    if get_tp_info().size > 1:
        raise NotImplementedError("qwen35moe GGUF expert banks currently support TP=1 only")
    types = config.gguf_expert_types
    assert types and len(types) == config.num_layers
    experts, hidden, inter = config.num_experts, config.hidden_size, config.moe_intermediate_size
    gu_stride, down_stride = _expert_bank_geometry(config)
    host = alloc_layer_banks(
        {
            "gate_up": ((experts, gu_stride), torch.uint8),
            "down": ((experts, down_stride), torch.uint8),
        },
        config.num_layers,
    )
    banks = {name: [b.tensor for b in per_layer] for name, per_layer in host.items()}
    seen_gu, seen_down = set(), set()

    def load(sink) -> None:
        tracker = LayerCompletionTracker(2, host, sink) if sink is not None else None
        gate_parts: dict[int, dict[str, torch.Tensor]] = {}
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk.") or not t.name.endswith(_EXPERT_SUFFIXES):
                continue
            layer = int(t.name.split(".")[1])
            if layer >= config.num_layers:
                continue
            gu_type, down_type = types[layer]
            if t.name.endswith("ffn_down_exps.weight"):
                payload = hidden * row_bytes(inter, down_type)
                banks["down"][layer][:, :payload].copy_(t.packed().reshape(experts, payload))
                seen_down.add(layer)
                if tracker is not None:
                    tracker.note(layer)
                continue
            half = inter * row_bytes(hidden, gu_type)
            slot = "gate" if "gate" in t.name else "up"
            parts = gate_parts.setdefault(layer, {})
            parts[slot] = t.packed().reshape(experts, half)
            if len(parts) == 2:
                banks["gate_up"][layer][:, :half].copy_(parts["gate"])
                banks["gate_up"][layer][:, half : 2 * half].copy_(parts["up"])
                del gate_parts[layer]
                seen_gu.add(layer)
                if tracker is not None:
                    tracker.note(layer)
        assert not gate_parts, f"incomplete qwen35moe expert gate/up layers: {sorted(gate_parts)}"

    if layer_sink is not None:
        load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            load(pins)
    else:
        load(None)
    wanted = set(range(config.num_layers))
    assert seen_gu == wanted and seen_down == wanted, (
        f"missing qwen35moe expert layers: gate_up {sorted(wanted - seen_gu)}, "
        f"down {sorted(wanted - seen_down)}"
    )
    return banks


def dummy_gguf_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    gu_stride, down_stride = _expert_bank_geometry(config)
    host = alloc_layer_banks(
        {
            "gate_up": ((config.num_experts, gu_stride), torch.uint8),
            "down": ((config.num_experts, down_stride), torch.uint8),
        },
        config.num_layers,
    )
    banks = {name: [b.tensor for b in per_layer] for name, per_layer in host.items()}
    for tensor in banks["gate_up"] + banks["down"]:
        tensor.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(host)
    return banks


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_qwen3_5_to_gguf",
    "is_gguf_model",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
]
