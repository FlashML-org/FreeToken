"""Routed-expert banks straight from a GGUF's ``blk.N.ffn_{gate,up,down}_exps`` tensors, in their packed ggml rows: pinned host banks for the offload cache, or device banks for resident experts.

Each layer keeps its own ggml types, so each layer gets its own bank shapes: ``gate_up`` [E, 2I, row_bytes] (expert e's gate rows, then its up rows) and ``down`` [E, H, row_bytes]; float experts keep their dtype ([E, 2I, H] and [E, H, I]).
A GGUF that stores ``ffn_gate_up_exps`` (gemma4) already holds the gate_up rows in that order.
When NVFP4 experts carry per-expert ``.scale`` tensors, every layer also gets ``global`` [E, 4] fp32: the (gate, up, down, 1) weight_scale_2, 1 where a tensor has none."""

from __future__ import annotations

import contextlib
import os

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.ggml_types import float_dtype, has_expert_scales, mmq_tail_bytes, pad_rows, type_name
from freetoken.utils import init_logger

from .reader import GgufTensorInfo, gguf_tensor_index
from .tensors import weight_scale

logger = init_logger(__name__)


def _expert_rows(info: GgufTensorInfo) -> torch.Tensor:
    """[E, rows, row_bytes] packed ggml rows, or [E, rows, K] for a float tensor."""
    k, rows, experts = info.ne
    raw = torch.from_numpy(info.data()).view(experts, rows, info.nbytes // (experts * rows))
    dtype = float_dtype(info.ggml_type)
    return raw if dtype is None else raw.view(dtype)


def _drop_page_cache(info: GgufTensorInfo) -> None:
    try:
        fd = os.open(info.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, info.offset, info.nbytes, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _device_bank(shape: tuple[int, ...], ggml_type: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An [E, rows, row] device bank; a zero row follows the last one when ggml's MMQ tiles read past it."""
    experts, rows, row = shape
    flat = torch.empty(experts * rows + pad_rows(ggml_type, k), row, dtype=dtype, device=device)
    flat[experts * rows :].zero_()
    return flat[: experts * rows].view(shape)


def _expert_scales(gate_up: list[GgufTensorInfo], down: GgufTensorInfo, experts: int) -> torch.Tensor:
    """[E, 4] fp32 (gate, up, down, 1) weight_scale_2 of one layer's experts."""
    gate, up = gate_up if len(gate_up) == 2 else (gate_up[0], gate_up[0])
    out = torch.ones(experts, 4, dtype=torch.float32)
    for col, info in enumerate((gate, up, down)):
        s = weight_scale(info)
        if s is None:
            continue
        if s.numel() != experts:
            raise NotImplementedError(f"{info.name}: {s.numel()} scales for {experts} experts")
        out[:, col] = s
    return out


def load_gguf_expert_banks(model_path: str, num_layers: int, *, first_layer: int = 0, dummy: bool = False, device: torch.device | None = None):
    """``ExpertBanks`` with one bank per (layer, role): pinned host banks for the offload cache, filled from the file and pinned while later layers load, or resident banks on ``device``."""
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import HostBank, PinPipeline

    index = gguf_tensor_index(model_path)
    scaled = has_expert_scales({name: info.ggml_type for name, info in index.items()})
    roles = ("gate_up", "down", "global") if scaled else ("gate_up", "down")
    sources: dict[str, list[torch.Tensor]] = {role: [] for role in roles}
    tails: dict[str, list[int]] = {role: [] for role in roles}
    tags = []
    with PinPipeline() if device is None else contextlib.nullcontext() as pins:
        for layer in range(num_layers):
            blk = f"blk.{first_layer + layer}"
            down = index[f"{blk}.ffn_down_exps.weight"]
            if f"{blk}.ffn_gate_up_exps.weight" in index:
                gate = index[f"{blk}.ffn_gate_up_exps.weight"]
                parts = [gate]
            else:
                gate, up = index[f"{blk}.ffn_gate_exps.weight"], index[f"{blk}.ffn_up_exps.weight"]
                if gate.ggml_type != up.ggml_type:
                    raise NotImplementedError(f"{blk}: gate and up experts have different ggml types")
                parts = [gate, up]
            tags.append(("ggml", type_name(gate.ggml_type), type_name(down.ggml_type)))
            tails["gate_up"].append(mmq_tail_bytes(gate.ggml_type, gate.ne[0]))
            tails["down"].append(mmq_tail_bytes(down.ggml_type, down.ne[0]))
            if scaled:
                tails["global"].append(0)
            rows = [_expert_rows(info) for info in parts]
            d = _expert_rows(down)
            gate_up_shape = (rows[0].shape[0], sum(r.shape[1] for r in rows), rows[0].shape[2])
            experts = rows[0].shape[0]
            if device is None:
                host = {"gate_up": HostBank(gate_up_shape, rows[0].dtype), "down": HostBank(tuple(d.shape), d.dtype)}
                if scaled:
                    host["global"] = HostBank((experts, 4), torch.float32)
                banks = {role: bank.tensor for role, bank in host.items()}
            else:
                banks = {
                    "gate_up": _device_bank(gate_up_shape, gate.ggml_type, gate.ne[0], rows[0].dtype, device),
                    "down": _device_bank(tuple(d.shape), down.ggml_type, down.ne[0], d.dtype, device),
                }
                if scaled:
                    banks["global"] = torch.empty(experts, 4, dtype=torch.float32, device=device)
            if scaled:
                banks["global"].copy_(_expert_scales(parts, down, experts))
            if dummy:
                banks["gate_up"].zero_()
                banks["down"].zero_()
            else:
                start = 0
                for r in rows:
                    banks["gate_up"][:, start : start + r.shape[1]].copy_(r)
                    start += r.shape[1]
                banks["down"].copy_(d)
                for info in (*parts, down):
                    _drop_page_cache(info)
            if device is None:
                pins(layer, host)
            for role, bank in banks.items():
                sources[role].append(bank)
    total = sum(t.numel() * t.element_size() for ts in sources.values() for t in ts)
    logger.info_rank0(f"GGUF expert banks: {num_layers} layers, {total / 2**30:.1f} GiB {'on ' + str(device) if device is not None else 'in host RAM'}")
    return ExpertBanks("gguf_scaled" if scaled else "gguf", sources, kind=QuantKind.GGUF, kernel="ggml", layout=None, layer_tags=tags, row_tail_bytes=tails)


__all__ = ["load_gguf_expert_banks"]
