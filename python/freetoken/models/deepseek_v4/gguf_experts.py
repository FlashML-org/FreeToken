"""Routed-expert host banks for a deepseek4 GGUF checkpoint.

Ported from ``models/qwen3_5_moe/gguf_experts.py``; the two are structurally the same job
because llama.cpp emits the same three stacked tensors for both architectures
(``ffn_gate_exps`` / ``ffn_up_exps`` / ``ffn_down_exps``). What differs is only the
arithmetic: DeepSeek-V4-Flash is 43 served layers of 256 experts at
``moe_inter_dim`` 2048 over ``dim`` 4096, and the checkpoints worth loading are uniformly
Q4_K on all three banks rather than qwen35moe's per-layer mix.

Only the ROUTED experts come through here. The shared expert
(``ffn_{gate,up,down}_shexp``) is an ordinary quantized Linear that
``convert_deepseek4_to_gguf`` swaps for a ``GGUFLinear``, exactly as qwen3_5_moe handles
its own shared expert -- it is dense per token, so there is nothing to offload.

A note on which checkpoints reach this code at all. The offload slot pool is ONE
allocation per bank shared by every layer, and ``moe_vec.cuh`` addresses it as
``expert * nrows * (ncols / qk)`` with no padding allowance, so a bank whose ggml type
varies by layer cannot be served. Of the thirteen published
``unsloth/DeepSeek-V4-Flash-0731-GGUF`` variants, eleven mix types across layers and the
two that do not are MXFP4 (ggml type 39), which has no entry in ``BLOCK_SHAPE`` and no
vendored kernel. The ``antirez/deepseek-v4-gguf`` builds are the ones that load: their
Q4KExperts variant is uniformly Q4_K across all 43 layers, verified by reading the file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.models.gguf.dequant import GGML_NAME, GGML_Q4_K, row_bytes

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def gguf_expert_types(model_path: str, num_layers: int) -> dict[str, list[int]]:
    """Scan the tensor table and return the per-layer ggml type of each expert bank.

    Returns ``{"gate_up": [...], "down": [...]}``, each a list of ``num_layers`` ggml type
    enums. gate and up must agree per layer because they are row-concatenated into one
    bank and therefore must share a row stride; a mismatch raises here naming both types.

    ``expert_banks._gguf_banks`` consumes this and is what rejects a bank that is
    non-uniform ACROSS layers, with the user-facing message about ``--pure``.
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
            # The trailing NextN/MTP block. DeepSeek-V4-Flash ships nextn_predict_layers=1,
            # so the file carries a block at index num_layers that is not served; counting
            # it here would make a uniform checkpoint look mixed.
            continue

        if t.name.endswith("ffn_gate_exps.weight"):
            gate_types[layer] = t.ggml_type
        elif t.name.endswith("ffn_up_exps.weight"):
            up_types[layer] = t.ggml_type
        elif t.name.endswith("ffn_down_exps.weight"):
            down_types[layer] = t.ggml_type

    gate_up_types: list[int] = []
    for layer in range(num_layers):
        gate_t, up_t = gate_types[layer], up_types[layer]
        if gate_t is None or up_t is None:
            raise ValueError(
                f"deepseek4 GGUF: layer {layer} is missing routed-expert tensors "
                f"(gate={gate_t}, up={up_t}); every layer of this architecture is MoE"
            )
        if gate_t != up_t:
            raise ValueError(
                f"deepseek4 GGUF: layer {layer} has ffn_gate_exps "
                f"{GGML_NAME.get(gate_t, gate_t)} but ffn_up_exps "
                f"{GGML_NAME.get(up_t, up_t)}; they are row-concatenated into one bank and "
                "cannot have different row strides"
            )
        gate_up_types.append(gate_t)

    for layer in range(num_layers):
        if down_types[layer] is None:
            raise ValueError(f"deepseek4 GGUF: layer {layer} is missing ffn_down_exps")

    return {"gate_up": gate_up_types, "down": down_types}


def gguf_expert_specs(
    config: "ModelConfig", types: dict[str, list[int]]
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Expert bank shapes as ``{name: (shape, dtype)}`` -- ``alloc_layer_banks``' contract.

    Packed block bytes, in torch order::

        gate_up  (E, 2*I, row_bytes(H, t_gate_up))   uint8
        down     (E, H,   row_bytes(I, t_down))      uint8

    One spec per bank rather than per layer, for the slot-pool stride reason in the module
    docstring. A non-uniform bank is rejected here instead of being mis-decoded.
    """
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    out: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for name, elems in (("gate_up", H), ("down", I)):
        distinct = sorted(set(types[name]))
        if len(distinct) != 1:
            names = [GGML_NAME.get(t, t) for t in distinct]
            raise ValueError(
                f"deepseek4 expert bank {name!r} mixes ggml types across layers ({names}); "
                "a bank must be uniform because its slot pool is one allocation with one "
                "stride"
            )
        rb = row_bytes(elems, distinct[0])
        shape = (E, 2 * I, rb) if name == "gate_up" else (E, H, rb)
        out[name] = (shape, torch.uint8)
    return out


def load_gguf_expert_sources(
    model_path: str, config: "ModelConfig", *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks holding the routed experts' native packed block bytes.

    Nothing is dequantized: the bytes handed to the offload cache are the same ones the
    kernels decode in the K-loop.

    ``layer_sink`` None (serving) pins each completed layer through an internally owned
    ``PinPipeline``; a supplied sink (converter) receives the completion notifications
    instead and may release banks, so the returned tensors live only as long as it allows.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    types = gguf_expert_types(model_path, config.num_layers)
    specs = gguf_expert_specs(config, types)

    L = config.num_layers
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size

    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}

    gate_buf: dict[int, torch.Tensor] = {}
    up_buf: dict[int, torch.Tensor] = {}
    seen_gate: set[int] = set()
    seen_up: set[int] = set()
    seen_down: set[int] = set()

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None

        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if layer >= L:
                continue  # trailing NextN/MTP block, not served

            if t.name.endswith("ffn_gate_exps.weight"):
                gate_buf[layer] = t.packed()
                seen_gate.add(layer)
            elif t.name.endswith("ffn_up_exps.weight"):
                up_buf[layer] = t.packed()
                seen_up.add(layer)
            elif t.name.endswith("ffn_down_exps.weight"):
                # torch shape [E, H, I] is ggml dims [I, H, E] with I fastest, so the reader
                # returns [E*H, row_bytes(I)] already in expert-major row order. Reshaping
                # to [E, H, row_bytes(I)] is a view, not a copy. Note the row_bytes is over
                # I (the fastest dim), not over E.
                down_rb = specs["down"][0][2]
                banks["down"][layer].copy_(t.packed().reshape(E, H, down_rb))
                seen_down.add(layer)
                if tracker is not None:
                    tracker.note(layer)
            else:
                continue

            if layer in gate_buf and layer in up_buf:
                rb = specs["gate_up"][0][2]
                # gate and up each arrive as [E*I, row_bytes(H)], and ggml's fastest-first
                # dims [H, I, E] make E the slowest axis, so those rows are EXPERT-MAJOR:
                # expert e owns rows [e*I, (e+1)*I).
                #
                # The bank must be [E, 2I, row_bytes(H)] with each expert's own gate rows
                # followed by its OWN up rows, so reshape to [E, I, rb] and concatenate on
                # the row axis within each expert (dim=1).
                #
                # cat(dim=0) then reshape(E, 2I, rb) -- the version that looks obviously
                # right -- lays every expert's gate down before any up, so expert 0 would
                # get its gate rows plus expert 1's gate rows and its up would sit E*I rows
                # away. That loads, runs at full speed, and emits fluent nonsense. This
                # exact bug cost real debugging time on qwen35moe.
                g = gate_buf[layer].reshape(E, I, rb)
                u = up_buf[layer].reshape(E, I, rb)
                banks["gate_up"][layer].copy_(torch.cat([g, u], dim=1))
                del gate_buf[layer], up_buf[layer]
                if tracker is not None:
                    tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    want = set(range(L))
    missing_gate, missing_up, missing_down = (
        want - seen_gate, want - seen_up, want - seen_down)
    if missing_gate or missing_up or missing_down:
        raise ValueError(
            f"deepseek4 GGUF is missing routed experts: gate {sorted(missing_gate)}, "
            f"up {sorted(missing_up)}, down {sorted(missing_down)}"
        )

    return banks


def dummy_gguf_expert_sources(config: "ModelConfig") -> dict[str, list[torch.Tensor]]:
    """Random expert banks shaped like ``load_gguf_expert_sources`` output.

    Q4_K throughout, matching the checkpoints that actually load (see module docstring).
    """
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = config.num_layers
    types = {"gate_up": [GGML_Q4_K] * L, "down": [GGML_Q4_K] * L}
    specs = gguf_expert_specs(config, types)

    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


__all__ = [
    "gguf_expert_types",
    "gguf_expert_specs",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
]
