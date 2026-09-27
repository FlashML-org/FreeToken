"""Native-GGUF MoE expert banks, generic over the block-quant type.

Generalizes the original gemma4 Q4_0 path: the routed experts stay as packed block
bytes per output row and are dequantized *inside* the borrowed ggml MoE kernels
(``ggml_moe_a8_vec``), with the quant type a parameter instead of a hardcoded Q4_0.
A GGUF architecture routes its experts through the offload cache with whatever type
the checkpoint shipped, as long as the kernel set covers it.

Banks are one ``[E, 2I, row_bytes(H)]`` ``gate_up`` and one ``[E, H, row_bytes(I)]``
``down`` per layer (or the same with the fused ``ffn_gate_up_exps`` / separate
``ffn_gate_exps`` + ``ffn_up_exps`` GGUF naming).
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import (
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_Q4_0,
    GGML_Q5_K,
    GGML_NAME,
    row_bytes,
)

# GGUF expert format tag -> ggml type. The tags match the names llama.cpp / gguf-py
# use (and the ``expert_quant`` / ``moe_weight_format`` config fields).
GGUF_EXPERT_QUANTS: dict[str, int] = {
    "q4_0": GGML_Q4_0,
    "q5_K": GGML_Q5_K,
    "iq4_nl": GGML_IQ4_NL,
    "iq4_xs": GGML_IQ4_XS,
}
_TAG_BY_TYPE = {ggml_type: tag for tag, ggml_type in GGUF_EXPERT_QUANTS.items()}


def gguf_expert_role_types(tag: str) -> tuple[int, int]:
    """``(gate_up_type, down_type)`` for an expert format tag.

    A ``'+'``-joined tag carries per-role types (e.g. ``iq4_xs+iq4_nl``: the gate/up rows
    are 256 wide so IQ4_XS fits, but the down rows span only ``moe_intermediate_size`` =
    640 and must use a 32-element block type). A plain tag means both roles match.
    """
    if "+" in tag:
        gate_up_tag, down_tag = tag.split("+", 1)
        return GGUF_EXPERT_QUANTS[gate_up_tag], GGUF_EXPERT_QUANTS[down_tag]
    quant_type = GGUF_EXPERT_QUANTS[tag]
    return quant_type, quant_type


def gguf_expert_format(model_path: str) -> str:
    """The checkpoint's routed-expert format tag, from ``ffn_{gate,up,down}_exps`` types.

    gate and up must agree across all layers (they fuse), as must the type of each role;
    down may differ from gate/up, which yields a ``'+'``-joined tag.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    seen: dict[str, set[int]] = {"gate": set(), "up": set(), "down": set()}
    for t in iter_gguf_tensors(model_path):
        if t.name.endswith("ffn_gate_up_exps.weight"):  # fused tensor covers both roles
            seen["gate"].add(t.ggml_type)
            seen["up"].add(t.ggml_type)
            continue
        for role in seen:
            if t.name.endswith(f"ffn_{role}_exps.weight"):
                seen[role].add(t.ggml_type)
    if not seen["gate"]:
        raise ValueError(
            f"{model_path}: no ffn_gate_exps.weight / ffn_gate_up_exps.weight "
            "to detect the expert quant"
        )
    for role, types in seen.items():
        if len(types) != 1:
            raise ValueError(
                f"{model_path}: ffn_{role}_exps.weight mixes quant types "
                f"{sorted(GGML_NAME.get(x, x) for x in types)}"
            )
    if seen["gate"] != seen["up"]:
        raise ValueError(
            f"{model_path}: gate/up expert quants differ "
            f"({sorted(GGML_NAME.get(x, x) for x in seen['gate'])} vs "
            f"{sorted(GGML_NAME.get(x, x) for x in seen['up'])})"
        )
    gate_up_tag = _TAG_BY_TYPE[next(iter(seen["gate"]))]
    down_tag = _TAG_BY_TYPE[next(iter(seen["down"]))]
    return gate_up_tag if gate_up_tag == down_tag else f"{gate_up_tag}+{down_tag}"

_ACT = None


def _activation(name: str):
    global _ACT
    if _ACT is None:
        from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

        _ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
    fn = _ACT.get(name)
    if fn is None:
        raise ValueError(f"unsupported MoE activation {name!r}")
    return fn


def expert_specs(
    hidden_size: int, intermediate_size: int, num_experts: int,
    gate_up_type: int, down_type: int,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Host-bank shapes for one layer: packed ``gate_up``/``down`` rows in their types."""
    H, I, E = hidden_size, intermediate_size, num_experts
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, gate_up_type)), torch.uint8),
        "down": ((E, H, row_bytes(I, down_type)), torch.uint8),
    }


def _expect_type(t, quant_type: int) -> None:
    if t.ggml_type != quant_type:
        raise ValueError(
            f"expert tensor {t.name}: {GGML_NAME.get(t.ggml_type, t.ggml_type)} does not match "
            f"the configured expert quant {GGML_NAME.get(quant_type, quant_type)}"
        )


def load_gguf_expert_sources(
    model_path: str,
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gate_up_type: int,
    down_type: int,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Per-layer packed expert banks from a GGUF, verbatim (no dequant).

    Accepts either the fused ``blk.N.ffn_gate_up_exps.weight`` or the separate
    ``blk.N.ffn_gate_exps.weight`` + ``blk.N.ffn_up_exps.weight`` (concatenated along
    the output dim), plus ``blk.N.ffn_down_exps.weight``. gate/up must be
    ``gate_up_type`` and down ``down_type`` (a checkpoint can quantize them differently,
    e.g. IQ4_XS gate/up with IQ4_NL down when the intermediate width isn't 256-aligned).
    ``layer_sink`` (converter) streams each completed layer instead of pinning here; on a
    CUDA-less host the banks stay pageable.
    """
    from freetoken.distributed import get_tp_info
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    if get_tp_info().size > 1:
        raise NotImplementedError("GGUF expert banks support TP=1 only")

    E, H, I = num_experts, hidden_size, intermediate_size
    h_bytes = row_bytes(H, gate_up_type)
    i_bytes = row_bytes(I, down_type)
    hb = alloc_layer_banks(expert_specs(H, I, E, gate_up_type, down_type), num_layers)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen = {"gate_up": set(), "down": set()}
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None  # gate_up + down
        for t in iter_gguf_tensors(model_path):
            name = t.name
            if not name.startswith("blk."):
                continue
            layer = int(name.split(".")[1])
            if not 0 <= layer < num_layers:
                continue
            if name.endswith("ffn_gate_up_exps.weight"):
                _expect_type(t, gate_up_type)
                banks["gate_up"][layer].copy_(t.packed().reshape(E, 2 * I, h_bytes))
                seen["gate_up"].add(layer)
                if tracker is not None:
                    tracker.note(layer)
            elif name.endswith("ffn_down_exps.weight"):
                _expect_type(t, down_type)
                banks["down"][layer].copy_(t.packed().reshape(E, H, i_bytes))
                seen["down"].add(layer)
                if tracker is not None:
                    tracker.note(layer)
            elif name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight"):
                _expect_type(t, gate_up_type)
                half = "gate" if name.endswith("ffn_gate_exps.weight") else "up"
                gate_up_buf.setdefault(layer, {})[half] = t.packed().reshape(E, I, h_bytes)
                slots = gate_up_buf[layer]
                if "gate" in slots and "up" in slots:
                    banks["gate_up"][layer][:, :I].copy_(slots["gate"])
                    banks["gate_up"][layer][:, I:].copy_(slots["up"])
                    del gate_up_buf[layer]
                    seen["gate_up"].add(layer)
                    if tracker is not None:
                        tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    want = set(range(num_layers))
    assert not gate_up_buf, f"incomplete gate/up expert groups: {sorted(gate_up_buf)}"
    assert seen["gate_up"] == want and seen["down"] == want, (
        f"missing GGUF expert layers: gate_up {sorted(want - seen['gate_up'])}, "
        f"down {sorted(want - seen['down'])}"
    )
    return banks


def dummy_gguf_expert_sources(
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gate_up_type: int,
    down_type: int,
) -> dict[str, list[torch.Tensor]]:
    """Random finite banks shaped like :func:`load_gguf_expert_sources` output."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    hb = alloc_layer_banks(
        expert_specs(hidden_size, intermediate_size, num_experts, gate_up_type, down_type),
        num_layers,
    )
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    gate_up_type: int,
    down_type: int,
) -> torch.Tensor:
    """Grouped expert GEMM over packed GGUF banks, dequantizing inside the kernel.

    ``gate_up_type`` / ``down_type`` are the ggml types of the respective banks (they
    may differ for mixed quants); both must have an ``ggml_moe_a8_vec`` case.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _activation(activation)
    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, gate_up_type, n2, num_tokens)
    inter = act_fn(gate_up)
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, down_type, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(out.dtype)
    return out.sum(dim=1)


__all__ = [
    "GGUF_EXPERT_QUANTS",
    "gguf_expert_role_types",
    "gguf_expert_format",
    "expert_specs",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
    "fused_experts_gguf",
]
