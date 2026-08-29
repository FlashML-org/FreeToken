"""Native GGUF routed-expert sources for Qwen3.6 Q4_K_M checkpoints.

This module owns only the mixed expert-bank portion of the Qwen GGUF path.
The model parser and dense tensor loader are deliberately separate because GGUF
encodes each tensor's quantization independently.  For the validated
Qwen3.6-35B-A3B control, gate and up are Q4_K while down is Q5_K.
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K, row_bytes


def _require_tp1() -> None:
    """Reject unsupported tensor parallelism before allocating unsharded GGUF banks."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen3.5 GGUF expert banks currently support TP=1 only")


def _expert_specs(config) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return host-bank shapes expressed in exact packed GGML row bytes."""
    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    return {
        "gate_up": ((experts, 2 * intermediate, row_bytes(hidden, GGML_Q4_K)), torch.uint8),
        "down": ((experts, hidden, row_bytes(intermediate, GGML_Q5_K)), torch.uint8),
    }


def load_q4_k_q5_k_expert_sources(model_path: str, config, *, layer_sink=None):
    """Load byte-exact Qwen GGUF experts into per-layer host banks.

    The loader fuses separately stored `ffn_gate_exps` and `ffn_up_exps` rows
    along their output dimension, which is safe because both use the same Q4_K
    input-row geometry.  `ffn_down_exps` remains Q5_K.  Completion is reported
    only after all three tensors for a layer are present, so a conversion sink
    can write or release the layer without racing a later tensor.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1()
    layers = int(config.num_layers)
    experts = int(config.num_experts)
    hidden = int(config.hidden_size)
    intermediate = int(config.moe_intermediate_size)
    gate_row_bytes = row_bytes(hidden, GGML_Q4_K)
    down_row_bytes = row_bytes(intermediate, GGML_Q5_K)
    host_banks = alloc_layer_banks(_expert_specs(config), layers)
    banks = {name: [bank.tensor for bank in host_banks[name]] for name in host_banks}
    gate_seen: set[int] = set()
    up_seen: set[int] = set()
    down_seen: set[int] = set()
    completed_gate_up: set[int] = set()

    def load(sink) -> None:
        # A completed layer consists of a fused Q4_K gate/up bank and one Q5_K down bank.
        tracker = LayerCompletionTracker(2, host_banks, sink) if sink is not None else None
        for tensor in iter_gguf_tensors(model_path):
            if not tensor.name.startswith("blk."):
                continue
            parts = tensor.name.split(".")
            layer = int(parts[1])
            suffix = ".".join(parts[2:])
            if suffix == "ffn_gate_exps.weight":
                if tensor.ggml_type != GGML_Q4_K:
                    raise ValueError(f"{tensor.name} expected Q4_K, got {tensor.ggml_type}")
                banks["gate_up"][layer][:, :intermediate].copy_(
                    tensor.packed().reshape(experts, intermediate, gate_row_bytes)
                )
                gate_seen.add(layer)
            elif suffix == "ffn_up_exps.weight":
                if tensor.ggml_type != GGML_Q4_K:
                    raise ValueError(f"{tensor.name} expected Q4_K, got {tensor.ggml_type}")
                banks["gate_up"][layer][:, intermediate:].copy_(
                    tensor.packed().reshape(experts, intermediate, gate_row_bytes)
                )
                up_seen.add(layer)
            elif suffix == "ffn_down_exps.weight":
                if tensor.ggml_type != GGML_Q5_K:
                    raise ValueError(f"{tensor.name} expected Q5_K, got {tensor.ggml_type}")
                banks["down"][layer].copy_(
                    tensor.packed().reshape(experts, hidden, down_row_bytes)
                )
                down_seen.add(layer)
                if tracker is not None:
                    tracker.note(layer)
            else:
                continue
            if layer in gate_seen and layer in up_seen and layer not in completed_gate_up:
                completed_gate_up.add(layer)
                if tracker is not None:
                    tracker.note(layer)

    if layer_sink is not None:
        load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            load(pins)
    else:
        load(None)

    wanted = set(range(layers))
    assert gate_seen == wanted and up_seen == wanted and down_seen == wanted, (
        "incomplete Qwen GGUF expert tensors: "
        f"gate={sorted(wanted - gate_seen)}, up={sorted(wanted - up_seen)}, "
        f"down={sorted(wanted - down_seen)}"
    )
    return banks


def dummy_q4_k_q5_k_expert_sources(config):
    """Build correctly shaped random packed banks for loader and cache tests."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    host_banks = alloc_layer_banks(_expert_specs(config), int(config.num_layers))
    banks = {name: [bank.tensor for bank in host_banks[name]] for name in host_banks}
    for tensor in banks["gate_up"] + banks["down"]:
        tensor.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(host_banks)
    return banks


__all__ = ["load_q4_k_q5_k_expert_sources", "dummy_q4_k_q5_k_expert_sources"]
