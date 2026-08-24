"""Routed-expert host bank sources for the qwen35moe GGUF checkpoint.

This module loads the per-expert weight tensors that are stored as GGUF stacks
and allocates them into host banks for the offload cache. The layout is a 3D
expert stack: [num_experts, out_features, in_features] in torch order.

CRITICAL CORRECTNESS NOTE: The MoE kernel (kernel/csrc/gguf/moe_vec.cuh)
computes addressing as `blocks_per_row = ncols / qk` and
`x = vx + expert * nrows * blocks_per_row`, i.e. it assumes a FULLY PACKED
contiguous [E, nrows, blocks_per_row] layout with NO padding. So the bank
tensors must be exactly `row_bytes` wide for their own quant type — never pad
a smaller-type layer up to a larger type's stride, because the kernel would
then read every block at the wrong offset and return plausible-looking garbage.

For qwen35moe specifically:
- ``ffn_gate_exps`` and ``ffn_up_exps`` are always IQ3_S (layers 0-39)
- ``ffn_down_exps`` is Q4_K for layers 0-4, IQ3_S for layers 5-39

The gate_up bank per layer is the per-expert concatenation of gate rows
then up rows along the output dimension, giving [E, 2*I, row_bytes(H, t)],
valid because gate and up share a quant type and therefore a row stride.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.models.gguf.dequant import GGML_IQ3_S, GGML_Q4_K, GGML_NAME, row_bytes

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def gguf_expert_types(model_path: str, num_layers: int) -> dict[str, list[int]]:
    """Scan the GGUF tensor table and return per-layer expert quant types.

    Returns a dict with two keys:
    - ``"gate_up"``: list of ``num_layers`` ggml_type enums for ``ffn_gate_exps``.
      gate and up for each layer must have the same type (they are row-concatenated).
      If they differ for any layer, raises a clear ValueError naming the layer and both types.
    - ``"down"``: list of ``num_layers`` ggml_type enums for ``ffn_down_exps``.

    For qwen35moe: gate_up is always IQ3_S (uniformly), and down varies by layer
    (Q4_K for 0-4, IQ3_S for 5-39).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    gate_types: list[int | None] = [None] * num_layers
    up_types: list[int | None] = [None] * num_layers
    down_types: list[int | None] = [None] * num_layers

    for t in iter_gguf_tensors(model_path):
        if not t.name.startswith("blk."):
            continue
        layer = int(t.name.split(".")[1])
        if layer >= num_layers:
            continue  # skip the trailing NextN/MTP block

        if t.name.endswith("ffn_gate_exps.weight"):
            gate_types[layer] = t.ggml_type
        elif t.name.endswith("ffn_up_exps.weight"):
            up_types[layer] = t.ggml_type
        elif t.name.endswith("ffn_down_exps.weight"):
            down_types[layer] = t.ggml_type

    # Validate that gate and up types agree for each layer (they must be row-concatenated).
    gate_up_types: list[int] = []
    for layer in range(num_layers):
        gate_t = gate_types[layer]
        up_t = up_types[layer]
        if gate_t is None or up_t is None:
            raise ValueError(
                f"missing expert tensors for layer {layer}: "
                f"gate={GGML_NAME.get(gate_t, gate_t)}, up={GGML_NAME.get(up_t, up_t)}"
            )
        if gate_t != up_t:
            raise ValueError(
                f"layer {layer}: ffn_gate_exps type {GGML_NAME.get(gate_t, gate_t)} != "
                f"ffn_up_exps type {GGML_NAME.get(up_t, up_t)}; "
                "cannot row-concatenate tensors with different quant types"
            )
        gate_up_types.append(gate_t)

    # Validate down tensors are present.
    for layer in range(num_layers):
        if down_types[layer] is None:
            raise ValueError(f"missing ffn_down_exps for layer {layer}")

    return {
        "gate_up": gate_up_types,
        "down": down_types,
    }


def gguf_expert_specs(
    config: ModelConfig, types: dict[str, list[int]]
) -> dict[str, list[tuple[tuple[int, ...], torch.dtype]]]:
    """Per-layer expert bank shapes, accounting for per-layer dtype variation.

    The routed experts are stored as 3D stacks [E, out, in] in torch order.
    Returns a list of (shape, dtype) tuples per expert bank:
    - ``gate_up[layer]``: ``((E, 2*I, row_bytes(H, types["gate_up"][layer])), torch.uint8)``
    - ``down[layer]``: ``((E, H, row_bytes(I, types["down"][layer])), torch.uint8)``

    where ``E=num_experts``, ``H=hidden_size``, ``I=moe_intermediate_size``.

    The row_bytes dimension varies by layer because the quant type varies,
    and the MoE kernel needs to read the exact byte width for its quant type.
    """
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = config.num_layers

    gate_up_specs = []
    down_specs = []

    for layer in range(num_layers):
        gate_up_type = types["gate_up"][layer]
        down_type = types["down"][layer]

        gate_up_row_bytes = row_bytes(H, gate_up_type)
        down_row_bytes = row_bytes(I, down_type)

        gate_up_specs.append(((E, 2 * I, gate_up_row_bytes), torch.uint8))
        down_specs.append(((E, H, down_row_bytes), torch.uint8))

    return {
        "gate_up": gate_up_specs,
        "down": down_specs,
    }


def load_gguf_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native packed block bytes.

    Loads the three GGUF expert stacks (gate, up, down) into per-layer host banks
    for the offload cache. The gate_up bank for each layer is the per-expert
    concatenation of that expert's gate rows then its up rows along the output
    dimension, giving [E, 2*I, row_bytes(H, t)] -- valid because gate and up
    share a quant type and therefore a row stride.

    Returns a dict with two keys:
    - ``"gate_up"``: list of ``num_layers`` tensors, each ``[E, 2*I, row_bytes_gate_up]`` uint8
    - ``"down"``: list of ``num_layers`` tensors, each ``[E, H, row_bytes_down]`` uint8

    Parameters:
    - ``layer_sink``: If None (serving mode), pins each completed layer via an
      internally-owned PinPipeline. If given (converter mode), fires the completion
      tracker into it instead -- nothing is pinned, and the sink may release banks,
      so returned tensors are only valid until the sink releases them.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    types = gguf_expert_types(model_path, config.num_layers)
    specs = gguf_expert_specs(config, types)

    L = config.num_layers
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size

    # Allocate the per-layer banks (lazy mmap, unpinned).
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}

    # Per-layer buffers to accumulate gate and up before concatenating.
    gate_buf: dict[int, torch.Tensor] = {}
    up_buf: dict[int, torch.Tensor] = {}
    seen_gate = set()
    seen_up = set()
    seen_down = set()

    def _load(sink) -> None:
        # Track completion: 2 banks per layer (gate_up and down).
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None

        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if layer >= L:
                continue  # skip the trailing NextN/MTP block

            if t.name.endswith("ffn_gate_exps.weight"):
                # Shape from GGUF: [E, I, H] in torch order = [H, I, E] in ggml order
                # t.packed() is [H*I, row_bytes(E, type)]
                gate_buf[layer] = t.packed()
                seen_gate.add(layer)

            elif t.name.endswith("ffn_up_exps.weight"):
                # Shape from GGUF: [E, I, H] in torch order = [H, I, E] in ggml order
                # t.packed() is [H*I, row_bytes(E, type)]
                up_buf[layer] = t.packed()
                seen_up.add(layer)

            elif t.name.endswith("ffn_down_exps.weight"):
                # Shape from GGUF: [E, H, I] in torch order = [I, H, E] in ggml order
                # t.packed() is [I*H, row_bytes(E, type)]
                # Reshape to [E, H, row_bytes(I, type)]
                down_row_bytes = specs["down"][layer][0][2]
                banks["down"][layer].copy_(t.packed().reshape(E, H, down_row_bytes))
                seen_down.add(layer)
                if tracker is not None:
                    tracker.note(layer)

            else:
                continue

            # Emit gate_up bank once both gate and up are present.
            if layer in gate_buf and layer in up_buf:
                gate_up_row_bytes = specs["gate_up"][layer][0][2]
                # Concatenate gate [H*I, row_bytes(E, type)] and up [H*I, row_bytes(E, type)]
                # to get [H*2*I, row_bytes(E, type)], then reshape to [E, 2*I, row_bytes(H, type)]
                combined = torch.cat([gate_buf[layer], up_buf[layer]], dim=0)
                banks["gate_up"][layer].copy_(combined.reshape(E, 2 * I, gate_up_row_bytes))
                del gate_buf[layer], up_buf[layer]
                if tracker is not None:
                    tracker.note(layer)

    # Load with or without pinning.
    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    # Verify all layers were loaded.
    want = set(range(L))
    missing_gate = want - seen_gate
    missing_up = want - seen_up
    missing_down = want - seen_down
    if missing_gate or missing_up or missing_down:
        raise ValueError(
            f"missing expert layers: gate {sorted(missing_gate)}, "
            f"up {sorted(missing_up)}, down {sorted(missing_down)}"
        )

    return banks


def dummy_gguf_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    """Random expert banks shaped like ``load_gguf_expert_sources`` output."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    # Use uniform IQ3_S for all layers (a simplification for the dummy).
    num_layers = config.num_layers
    gate_up_types = [GGML_IQ3_S] * num_layers
    down_types = [GGML_IQ3_S] * num_layers
    types = {"gate_up": gate_up_types, "down": down_types}

    specs = gguf_expert_specs(config, types)
    L = config.num_layers

    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}

    # Fill with random uint8.
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)

    if torch.cuda.is_available():
        pin_banks(hb)  # match the other dummies: pin-after-fill

    return banks


__all__ = [
    "gguf_expert_types",
    "gguf_expert_specs",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
]
