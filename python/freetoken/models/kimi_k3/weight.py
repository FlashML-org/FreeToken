"""Kimi-K3 checkpoint and routed-expert weight loading.

The public checkpoint wraps the text tower in ``language_model`` and stores each
routed expert as compressed-tensors MXFP4 rows.  The offload kernels consume the
same values transposed into their split-K layout.  Kimi's expert input is the
3584-wide latent MoE representation, not the decoder's 7168-wide residual.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import safetensors
import torch
from tqdm import tqdm

from freetoken.distributed import get_tp_info
from freetoken.models.loader import ShardReader, iter_weight_files
from freetoken.moe.fused_mxfp4 import dequant_mxfp4_blocks
from freetoken.utils import cached_load_hf_config

from .config import parse_config

_EXPERT_FRAGMENT = ".block_sparse_moe.experts."
_EXPERT_KEY_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\."
    r"experts\.(?P<expert>\d+)\.(?P<proj>w1|w2|w3)\."
    r"(?P<kind>weight_packed|weight_scale)$"
)
_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_DENSE_MLP_PART_RE = re.compile(
    r"^(?P<prefix>model\.layers\.0\.mlp)\.(?P<part>gate|up)_proj\.weight$"
)
_SHARED_MLP_PART_RE = re.compile(
    r"^(?P<prefix>model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.shared_experts)\."
    r"(?P<part>gate|up)_proj\.weight$"
)


def _resident_name(raw_name: str, num_layers: int) -> str | None:
    """Return the text-tower state key, dropping vision, MTP and routed experts."""
    lower_parts = raw_name.lower().split(".")
    if any(part.startswith(("vision", "mtp", "mm_projector")) for part in lower_parts):
        return None
    if _EXPERT_FRAGMENT in raw_name:
        return None
    name = raw_name.removeprefix("language_model.")
    match = _LAYER_RE.match(name)
    if match is not None and int(match.group("layer")) >= num_layers:
        return None  # appended multi-token-prediction layer
    correction = ".block_sparse_moe.gate.e_score_correction_bias"
    if name.endswith(correction):
        return name[: -len(correction)] + ".block_sparse_moe.e_score_correction_bias"
    return name


def _fuse_resident_mlp(
    name: str,
    tensor: torch.Tensor,
    fuse_buf: dict[str, dict[str, torch.Tensor]],
    num_layers: int,
) -> tuple[str, torch.Tensor] | None:
    """Buffer one BF16 gate/up part and emit a gate-then-up fusion."""
    match = _DENSE_MLP_PART_RE.fullmatch(name) or _SHARED_MLP_PART_RE.fullmatch(name)
    if match is None:
        return name, tensor
    layer = match.groupdict().get("layer")
    if layer is not None and not 1 <= int(layer) < num_layers:
        raise ValueError(f"unexpected Kimi-K3 shared-expert MLP layer: {name}")
    target = f"{match.group('prefix')}.gate_up_proj.weight"
    part = match.group("part")
    slots = fuse_buf.setdefault(target, {})
    if part in slots:
        raise ValueError(f"duplicate Kimi-K3 resident MLP tensor: {name}")
    slots[part] = tensor
    if len(slots) != 2:
        return None
    gate, up = slots["gate"], slots["up"]
    if gate.ndim != 2 or gate.shape != up.shape:
        raise ValueError(
            f"incompatible Kimi-K3 gate/up fusion for {target}: "
            f"{tuple(gate.shape)} and {tuple(up.shape)}"
        )
    if gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError(
            f"dtype/device mismatch in Kimi-K3 gate/up fusion for {target}"
        )
    del fuse_buf[target]
    return target, torch.cat((gate, up), dim=0)


def _dequant_resident_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(f"invalid Kimi-K3 resident MXFP4 dtypes for {name}")
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError(f"invalid Kimi-K3 resident MXFP4 ranks for {name}")
    if packed.shape[0] != scales.shape[0] or packed.shape[1] != scales.shape[1] * 16:
        raise ValueError(
            f"invalid Kimi-K3 resident MXFP4 shapes for {name}: "
            f"{tuple(packed.shape)} and {tuple(scales.shape)}"
        )
    blocks = packed.reshape(packed.shape[0], scales.shape[1], 16)
    return dequant_mxfp4_blocks(blocks, scales, out_dtype=torch.bfloat16)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield resident text weights, preserving official projection names.

    Routed experts are only valid in the host offload banks and are never yielded
    into ``load_state_dict``.  Kimi TP loading is deliberately rejected until the
    model's latent-MoE projections and expert banks have a single agreed TP ABI.
    """
    if include_moe_experts:
        raise ValueError("Kimi-K3 routed experts must be loaded into MXFP4 host banks")
    if not include_non_moe:
        return
    tp_info = get_tp_info()
    if tp_info.size != 1:
        raise NotImplementedError("Kimi-K3 weight loading currently supports TP=1 only")
    config = parse_config(cached_load_hf_config(model_path))
    fuse_buf: dict[str, dict[str, torch.Tensor]] = {}
    reader = ShardReader(model_path, device)
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading Kimi-K3 resident weights",
            disable=not tp_info.is_primary(),
        ):
            for raw_name in reader.names_in(file):
                if raw_name.endswith(".weight_scale"):
                    continue  # consumed with its packed sibling, potentially cross-shard
                name = _resident_name(raw_name, config.num_layers)
                if name is None:
                    continue
                if raw_name.endswith(".weight_packed"):
                    raw_base = raw_name[: -len(".weight_packed")]
                    name = name[: -len(".weight_packed")] + ".weight"
                    tensor = _dequant_resident_mxfp4(
                        reader.get_tensor(raw_name),
                        reader.get_tensor(raw_base + ".weight_scale"),
                        name=name,
                    )
                else:
                    tensor = reader.get_tensor(raw_name)
                fused = _fuse_resident_mlp(name, tensor, fuse_buf, config.num_layers)
                if fused is not None:
                    yield fused
    finally:
        reader.close()
    if fuse_buf:
        raise ValueError(f"incomplete Kimi-K3 resident MLP fusions: {sorted(fuse_buf)}")


def _bank_shapes(
    num_experts: int,
    hidden: int,
    intermediate: int,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        "gate_up_blocks": (
            (num_experts, hidden // 2, 2 * intermediate),
            torch.uint8,
        ),
        "gate_up_scales": (
            (num_experts, hidden // 32, 2 * intermediate),
            torch.uint8,
        ),
        "down_blocks": (
            (num_experts, intermediate // 2, hidden),
            torch.uint8,
        ),
        "down_scales": (
            (num_experts, intermediate // 32, hidden),
            torch.uint8,
        ),
    }


def _allocate_banks(
    num_layers: int,
    num_experts: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
):
    """Allocate bulk banks per layer and one shared immutable zero bias per role."""
    specs = _bank_shapes(num_experts, hidden, intermediate)
    host_banks = {}
    if torch.cuda.is_available():
        from freetoken.moe.host_banks import alloc_layer_banks

        host_banks = alloc_layer_banks(specs, num_layers)
        banks = {
            name: [bank.tensor for bank in values]
            for name, values in host_banks.items()
        }
    else:
        banks = {
            name: [torch.empty(shape, dtype=dt) for _ in range(num_layers)]
            for name, (shape, dt) in specs.items()
        }

    # The checkpoint is bias-free. A single full, contiguous expert bank can be
    # shared safely by every layer; this retains ExpertBanks' [E, N] contract
    # without allocating roughly a gigabyte of identical zeros.
    gate_bias = torch.zeros((num_experts, 2 * intermediate), dtype=dtype)
    down_bias = torch.zeros((num_experts, hidden), dtype=dtype)
    banks["gate_up_bias"] = [gate_bias] * num_layers
    banks["down_bias"] = [down_bias] * num_layers
    return banks, host_banks


def _expected_source_shape(
    proj: str,
    kind: str,
    hidden: int,
    intermediate: int,
) -> tuple[int, int]:
    out_features, in_features = (
        (intermediate, hidden) if proj in {"w1", "w3"} else (hidden, intermediate)
    )
    divisor = 2 if kind == "weight_packed" else 32
    return out_features, in_features // divisor


def _copy_expert_tensor(
    banks: dict[str, list[torch.Tensor]],
    *,
    layer: int,
    expert: int,
    proj: str,
    kind: str,
    value: torch.Tensor,
    hidden: int,
    intermediate: int,
) -> None:
    expected = _expected_source_shape(proj, kind, hidden, intermediate)
    if value.dtype != torch.uint8 or tuple(value.shape) != expected:
        raise ValueError(
            f"invalid Kimi-K3 {proj}.{kind}: expected uint8 {expected}, "
            f"got {value.dtype} {tuple(value.shape)}"
        )
    if proj == "w2":
        bank = "down_blocks" if kind == "weight_packed" else "down_scales"
        banks[bank][layer][expert].copy_(value.t())
        return

    bank = "gate_up_blocks" if kind == "weight_packed" else "gate_up_scales"
    # SiTU consumes gate/up pairs: w1 (gate) is even, w3 (up) is odd.
    parity = 0 if proj == "w1" else 1
    banks[bank][layer][expert, :, parity::2].copy_(value.t())


def _expert_bank_geometry(model_config) -> tuple[int, int, int, int, int]:
    """Validate the Kimi expert-bank ABI and return its dimensions."""
    if model_config.moe_weight_format != "mxfp4":
        raise ValueError("Kimi-K3 expert offload requires MXFP4 weights")
    args = model_config.kimi_k3_args
    hidden = args.routed_expert_hidden_size
    intermediate = model_config.moe_intermediate_size
    if hidden % 32 or intermediate % 32:
        raise ValueError(
            f"Kimi-K3 MXFP4 expert dimensions must be divisible by 32: "
            f"{(hidden, intermediate)}"
        )
    first_moe_layer = model_config.first_k_dense_replace
    num_moe_layers = model_config.num_layers - first_moe_layer
    if first_moe_layer != 1 or num_moe_layers <= 0:
        raise ValueError("Kimi-K3 expert loader requires one dense prefix layer")
    if get_tp_info().size != 1:
        raise NotImplementedError(
            "Kimi-K3 MXFP4 expert banks currently support TP=1 only"
        )
    return (
        first_moe_layer,
        num_moe_layers,
        model_config.num_experts,
        hidden,
        intermediate,
    )


def _expert_key(
    name: str,
    *,
    first_moe_layer: int,
    num_layers: int,
    num_experts: int,
) -> tuple[int, int, str, str]:
    """Parse and range-check an official routed-expert tensor name."""
    match = _EXPERT_KEY_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected Kimi-K3 expert tensor: {name}")
    checkpoint_layer = int(match.group("layer"))
    expert = int(match.group("expert"))
    if (
        not first_moe_layer <= checkpoint_layer < num_layers
        or not 0 <= expert < num_experts
    ):
        raise ValueError(f"out-of-range Kimi-K3 expert tensor: {name}")
    return (
        checkpoint_layer - first_moe_layer,
        expert,
        match.group("proj"),
        match.group("kind"),
    )


def _check_complete_expert_banks(
    seen: set[tuple[int, int, str, str]],
    *,
    num_moe_layers: int,
    num_experts: int,
) -> None:
    expected = {
        (layer, expert, proj, kind)
        for layer in range(num_moe_layers)
        for expert in range(num_experts)
        for proj in ("w1", "w2", "w3")
        for kind in ("weight_packed", "weight_scale")
    }
    missing = expected - seen
    if missing:
        raise ValueError(f"missing Kimi-K3 expert tensors: {sorted(missing)[:8]}")


def _pin_expert_banks(
    banks: dict[str, list[torch.Tensor]],
    host_banks: dict,
    num_moe_layers: int,
) -> None:
    if not host_banks:
        return
    from freetoken.moe.host_banks import pin_banks

    pin_banks(host_banks)
    # Bias banks are small relative to weights and shared across layers. Pin one
    # copy of each so all repeated source pointers are DMA-safe.
    banks["gate_up_bias"] = [banks["gate_up_bias"][0].pin_memory()] * num_moe_layers
    banks["down_bias"] = [banks["down_bias"][0].pin_memory()] * num_moe_layers


def load_mxfp4_expert_banks(
    model_path: str,
    model_config,
    *,
    dtype: torch.dtype,
) -> dict[str, list[torch.Tensor]]:
    """Load official per-expert MXFP4 tensors into transposed group-32 banks."""
    first_moe_layer, num_moe_layers, num_experts, hidden, intermediate = (
        _expert_bank_geometry(model_config)
    )
    banks, host_banks = _allocate_banks(
        num_moe_layers, num_experts, hidden, intermediate, dtype
    )
    seen: set[tuple[int, int, str, str]] = set()

    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            # SafeOpen is intentionally not treated as a mapping: explicit
            # ``keys()`` works across the supported safetensors releases.
            for name in f.keys():  # noqa: SIM118
                if _EXPERT_FRAGMENT not in name:
                    continue
                key = _expert_key(
                    name,
                    first_moe_layer=first_moe_layer,
                    num_layers=model_config.num_layers,
                    num_experts=num_experts,
                )
                if key in seen:
                    raise ValueError(f"duplicate Kimi-K3 expert tensor: {name}")
                seen.add(key)
                _copy_expert_tensor(
                    banks,
                    layer=key[0],
                    expert=key[1],
                    proj=key[2],
                    kind=key[3],
                    value=f.get_tensor(name),
                    hidden=hidden,
                    intermediate=intermediate,
                )

    _check_complete_expert_banks(
        seen, num_moe_layers=num_moe_layers, num_experts=num_experts
    )
    _pin_expert_banks(banks, host_banks, num_moe_layers)
    return banks


def load_mxfp4_expert_banks_parallel(
    model_path: str,
    model_config,
    *,
    dtype: torch.dtype,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> dict[str, list[torch.Tensor]]:
    """Load Kimi MXFP4 experts with shard prefetch and parallel O_DIRECT reads."""
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    first_moe_layer, num_moe_layers, num_experts, hidden, intermediate = (
        _expert_bank_geometry(model_config)
    )
    banks, host_banks = _allocate_banks(
        num_moe_layers, num_experts, hidden, intermediate, dtype
    )
    seen: set[tuple[int, int, str, str]] = set()

    def _load(sink) -> None:
        tracker = (
            LayerCompletionTracker(num_experts * 6, host_banks, sink)
            if host_banks
            else None
        )
        for name, value in iter_expert_tensors_parallel(
            model_path,
            lambda candidate: _EXPERT_FRAGMENT in candidate,
            workers=workers,
            chunk=chunk,
        ):
            key = _expert_key(
                name,
                first_moe_layer=first_moe_layer,
                num_layers=model_config.num_layers,
                num_experts=num_experts,
            )
            if key in seen:
                raise ValueError(f"duplicate Kimi-K3 expert tensor: {name}")
            seen.add(key)
            _copy_expert_tensor(
                banks,
                layer=key[0],
                expert=key[1],
                proj=key[2],
                kind=key[3],
                value=value,
                hidden=hidden,
                intermediate=intermediate,
            )
            if tracker is not None:
                tracker.note(key[0])

    if host_banks:
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    _check_complete_expert_banks(
        seen, num_moe_layers=num_moe_layers, num_experts=num_experts
    )
    # Layer banks were pinned by PinPipeline as each layer completed. Only the
    # small shared zero-bias banks remain.
    if host_banks:
        banks["gate_up_bias"] = [banks["gate_up_bias"][0].pin_memory()] * num_moe_layers
        banks["down_bias"] = [banks["down_bias"][0].pin_memory()] * num_moe_layers
    return banks


def _dummy_banks(model_config, dtype: torch.dtype) -> dict[str, list[torch.Tensor]]:
    args = model_config.kimi_k3_args
    num_moe_layers = model_config.num_layers - model_config.first_k_dense_replace
    banks, host_banks = _allocate_banks(
        num_moe_layers,
        model_config.num_experts,
        args.routed_expert_hidden_size,
        model_config.moe_intermediate_size,
        dtype,
    )
    for name in ("gate_up_blocks", "down_blocks"):
        for tensor in banks[name]:
            tensor.random_(0, 256)
    for name in ("gate_up_scales", "down_scales"):
        for tensor in banks[name]:
            tensor.fill_(127)
    if host_banks:
        from freetoken.moe.host_banks import pin_banks

        pin_banks(host_banks)
        banks["gate_up_bias"] = [banks["gate_up_bias"][0].pin_memory()] * num_moe_layers
        banks["down_bias"] = [banks["down_bias"][0].pin_memory()] * num_moe_layers
    return banks


def setup_offload_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
):
    """Model-specific ExpertBanks provider used by the offload engine."""
    del device  # host banks are CPU-resident by contract
    from freetoken.moe.expert_banks import ExpertBanks

    if model_config.moe_weight_format != "mxfp4":
        raise ValueError("Kimi-K3 offload requires MXFP4 expert weights")
    if dummy:
        sources = _dummy_banks(model_config, dtype)
    elif parallel:
        sources = load_mxfp4_expert_banks_parallel(
            model_path,
            model_config,
            dtype=dtype,
            workers=workers,
            chunk=chunk,
        )
    else:
        sources = load_mxfp4_expert_banks(model_path, model_config, dtype=dtype)
    return ExpertBanks("mxfp4_triton", sources)


__all__ = [
    "iter_weights",
    "load_mxfp4_expert_banks",
    "load_mxfp4_expert_banks_parallel",
    "setup_offload_expert_banks",
]
