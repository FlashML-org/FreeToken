"""LFM2-MoE checkpoint reader.

The dense pass yields every non-expert tensor under its own checkpoint name: the attribute
paths of the model equal the checkpoint keys, so there is nothing to rename. The quantized
projections (``{q,k,v,out}_proj``, dense ``w1``/``w2``/``w3``) are read through the
QuantConfig the model was built from -- its role table maps the checkpoint's tensor suffixes
onto the model buffers (compressed-tensors ``weight_scale`` -> the layer's
``weight_scale_inv``) and its block edge validates the scale shape.

Routed experts are per-expert in every release, so they never ride the dense pass: the bf16
stacked form (``experts.gate_up_proj`` / ``experts.down_proj``) is built per layer here, and
``iter_expert_pieces`` streams the bank pieces straight to the expert kernels.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantConfig, QuantKind, get_quant_config
from freetoken.models.loader import ShardReader, iter_weight_files
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

# per-expert routed experts: model.layers.N.feed_forward.experts.E.{w1,w3,w2}.{weight,weight_scale}
_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.feed_forward\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|weight_scale)$"
)
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
# the Linear leaves the checkpoint may quantize (under self_attn./feed_forward.); conv, gate,
# embed and every norm stay bf16 in every release
_QUANT_PARENTS = ("self_attn", "feed_forward")
_QUANT_LEAVES = frozenset({"q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"})
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)
_E4M3 = torch.float8_e4m3fn


def _require_tp1() -> None:
    if get_tp_info().size > 1:
        raise NotImplementedError("lfm2_moe weight loading supports TP=1 only")


def _is_quant_linear(module: str) -> bool:
    parent, _, leaf = module.rpartition(".")
    return leaf in _QUANT_LEAVES and parent.rpartition(".")[2] in _QUANT_PARENTS


def _scale_suffixes(quant: QuantConfig | None) -> frozenset[str]:
    """Tensor suffixes the checkpoint dialect uses for quant-side tensors (scales, not weights).

    Read off the dialect's own role table so the names follow the checkpoint rather than this
    reader; a tensor with one of these names belongs to a module the checkpoint quantized.
    """
    storage = getattr(quant, "STORAGE", None) or {}
    out = set()
    for roles in storage.values():
        for entry in roles.values():
            name = entry.name if hasattr(entry, "name") else entry
            if name != "weight":
                out.add(name)
    return frozenset(out)


def _check_block_scales(module: str, scheme, part: dict[str, torch.Tensor]) -> None:
    """Validate the block-fp8 weight/scale pair against the scheme's block edge."""
    if scheme.kind is not QuantKind.FP8_BLOCK:
        return
    weight, scale = part["weight"], part["weight_scale_inv"]
    block = scheme.weight.group or (1, 1)
    rows, cols = weight.shape
    expected = (rows // block[0], cols // block[1])
    if rows % block[0] or cols % block[1] or tuple(scale.shape) != expected:
        raise ValueError(
            f"{module}: weight_scale_inv is {tuple(scale.shape)}, expected {expected} for a "
            f"{block[0]}x{block[1]} block-fp8 {tuple(weight.shape)} weight"
        )
    if scale.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{module}: block scales are {scale.dtype}; block-fp8 stores float scales")


def _emit_dense(
    name: str,
    tensor: torch.Tensor,
    quant: QuantConfig | None,
    scale_suffixes: frozenset[str],
    pending: dict[str, dict[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """``[(model-side name, tensor)]`` for one dense checkpoint tensor ([] while its quantized
    module is still incomplete), or a pass-through pair for a tensor the family serves as stored."""
    module, _, suffix = name.rpartition(".")
    scheme = quant.scheme_for(module) if quant is not None and _is_quant_linear(module) else None
    if scheme is None:
        if suffix in scale_suffixes:
            raise ValueError(
                f"{name}: the checkpoint quantized a module lfm2_moe serves in bf16 ({module})"
            )
        if _is_quant_linear(module) and tensor.dtype in _QUANT_DTYPES:
            raise ValueError(
                f"{name} is {tensor.dtype} but the checkpoint's quant config serves {module} in bf16"
            )
        return [(name, tensor)]

    assert quant is not None  # a scheme only ever comes from the config
    roles = {entry.name: role for role, entry in quant.storage(scheme).items()}
    role = roles.get(suffix)
    if role is None:
        raise ValueError(
            f"{name}: the checkpoint's quant config stores {module} as {sorted(roles)}, not {suffix!r}"
        )
    if role == "weight" and tensor.dtype is not _E4M3:
        raise ValueError(
            f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}"
        )
    part = pending.setdefault(module, {})
    part[role] = tensor
    if len(part) < len(roles):
        return []
    del pending[module]
    _check_block_scales(module, scheme, part)
    return [(f"{module}.{r}", part[r]) for r in sorted(part)]


class _ExpertStacker:
    """Whole-layer bf16 expert stacks from the checkpoint's per-expert ``w1``/``w3``/``w2``.

    Yields exactly the names the generic bf16 piece reader expects
    (``model.layers.N.feed_forward.experts.{gate_up_proj,down_proj}``, gate half first). One
    layer's experts are held at a time, so the transient is a single layer's experts.
    """

    def __init__(self, config) -> None:
        self._config = config
        self._dense = config.first_k_dense_replace
        self._pending: dict[int, dict[tuple[str, int], torch.Tensor]] = {}

    def owns(self, name: str) -> bool:
        match = _EXPERT_RE.match(name)
        return match is not None and match["kind"] == "weight"

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        match = _EXPERT_RE.match(name)
        assert match is not None, f"not a routed-expert tensor: {name}"
        layer = int(match["layer"]) - self._dense
        if not 0 <= layer < self._config.num_moe_layers:
            raise ValueError(f"unexpected routed-expert layer in {name}")
        pending = self._pending.setdefault(layer, {})
        key = (match["proj"], int(match["expert"]))
        if key in pending:
            raise ValueError(f"duplicate expert tensor in checkpoint: {name}")
        pending[key] = tensor
        num_experts = self._config.num_experts
        if len(pending) < 3 * num_experts:
            return []
        del self._pending[layer]
        experts = range(num_experts)
        gate = torch.stack([pending[("w1", e)] for e in experts])
        up = torch.stack([pending[("w3", e)] for e in experts])
        down = torch.stack([pending[("w2", e)] for e in experts])
        prefix = f"model.layers.{layer + self._dense}.feed_forward.experts"
        return [
            (f"{prefix}.gate_up_proj", torch.cat([gate, up], dim=1)),
            (f"{prefix}.down_proj", down),
        ]

    def close(self) -> None:
        if self._pending:
            raise ValueError(
                f"incomplete expert tensors in checkpoint (layers {sorted(self._pending)})"
            )


def _iter_weights(model_path: str, device: torch.device, include_moe_experts: bool, include_non_moe: bool):
    config = parse_config(cached_load_hf_config(model_path))
    quant = get_quant_config()
    scale_suffixes = _scale_suffixes(quant)
    experts = _ExpertStacker(config) if include_moe_experts else None
    pending: dict[str, dict[str, torch.Tensor]] = {}

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                # fetch only what this pass needs: a dense load must not read the (mostly
                # quantized) expert tensors at all
                if experts is not None and experts.owns(raw_name):
                    yield from experts.add(raw_name, f.get_tensor(raw_name))
                    continue
                if not include_non_moe or _EXPERT_RE.match(raw_name):
                    continue
                yield from _emit_dense(raw_name, f.get_tensor(raw_name), quant, scale_suffixes, pending)

    if experts is not None:
        experts.close()
    if pending:
        missing = {module: sorted(part) for module, part in sorted(pending.items())}
        raise ValueError(f"checkpoint is missing tensors the quant config declares: {missing}")


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The checkpoint's dense tensors named as the model's own attribute paths, and with
    ``include_moe_experts`` the bf16 stacked experts of every MoE layer."""
    _require_tp1()
    yield from _iter_weights(model_path, device, include_moe_experts, include_non_moe)


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Experts-only parallel read (the chunked multi-threaded O_DIRECT reader), stacking each
    layer's experts like the serial path -- used by ``--expert-load parallel`` on bf16 banks."""
    assert include_moe_experts and not include_non_moe, (
        "lfm2_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    _require_tp1()
    from freetoken.models.weight import iter_expert_tensors_parallel

    experts = _ExpertStacker(parse_config(cached_load_hf_config(model_path)))
    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, experts.owns, workers=workers, chunk=chunk
    ):
        yield from experts.add(raw_name, tensor)
    experts.close()


def iter_expert_pieces(
    model_path: str,
    config,
    kind: QuantKind,
    *,
    parallel: bool | None = False,
    workers: int = 8,
    chunk: int = 8 << 20,
):
    """Routed experts as one piece per expert: ``{gate, up, down}`` (bf16) with their ``_scale``
    block-scale companions (block-fp8). Other kinds use the generic readers."""
    if kind not in (QuantKind.NONE, QuantKind.FP8_BLOCK):
        return None
    _require_tp1()
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    dense = config.first_k_dense_replace
    num_moe_layers = config.num_moe_layers
    if kind is QuantKind.FP8_BLOCK:
        quant = get_quant_config()
        assert quant is not None, "block-fp8 experts need the checkpoint's QuantConfig installed"
        scale = quant.stored_tensors(QuantKind.FP8_BLOCK)["weight_scale_inv"].name
        suffixes = {"weight": "", scale: "_scale"}
    else:
        suffixes = {"weight": ""}

    def locate(raw_name: str):
        match = _EXPERT_RE.match(raw_name)
        if match is None or match["kind"] not in suffixes:
            return None
        layer = int(match["layer"]) - dense
        if not 0 <= layer < num_moe_layers:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        return layer, int(match["expert"]), _PROJ_ROLE[match["proj"]] + suffixes[match["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=3 * len(suffixes))

    def _serial():
        reader = ShardReader(model_path, torch.device("cpu"))
        try:
            for layer in tqdm(
                range(num_moe_layers),
                desc="Loading experts (serial)",
                disable=not get_tp_info().is_primary(),
            ):
                for expert in range(config.num_experts):
                    base = f"model.layers.{dense + layer}.feed_forward.experts.{expert}"
                    for proj in ("w1", "w3", "w2"):
                        for kind_name in suffixes:
                            name = f"{base}.{proj}.{kind_name}"
                            yield name, reader.get_tensor(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=3 * len(suffixes))


__all__ = ["iter_weights", "iter_weights_parallel", "iter_expert_pieces"]
