"""Expert banks for the offload MoE cache: load, pack and pin the routed experts.

The expert kernel (``QuantMethod.kernel``) owns the bank layout and the pack step; the
checkpoint side delivers pieces (``moe.expert_pieces``) and this module fills the pinned host
banks from them (``build_expert_banks``). The GGUF q4_0 experts still
use their own providers until they get a method.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.utils import init_logger

from .host_banks import alloc_layer_banks
from .offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS

logger = init_logger(__name__)

# the parallel expert-bank reader needs POSIX O_DIRECT + preadv; without them the serial (safetensors/mmap) build is the only option
_PARALLEL_READER_SUPPORTED = hasattr(os, "O_DIRECT") and hasattr(os, "preadv")


@dataclass(frozen=True)
class ExpertBanks:
    """Loaded expert banks, normalized for ``OffloadMoeCache`` wiring."""

    quant_format: str  # _BANK_SCHEMAS key
    # Pinned host banks, keyed by the format's schema: one [num_experts, ...]
    # tensor per layer (independent allocations -> per-layer host attributes).
    sources: dict[str, list[torch.Tensor]]
    # marlin/b12x per-expert global scales ([L*E]); None for formats without them
    gate_up_alpha: torch.Tensor | None = field(default=None)
    down_alpha: torch.Tensor | None = field(default=None)
    # per-layer HostResidency values actually applied by the loader; None -> all pinned (also the degrade signal when a request was not honored)
    layer_residency: list[str] | None = field(default=None)
    # True iff the ``layer_sink`` passed to the loader was actually engaged (each layer
    # streamed straight to its sink instead of staying materialized here) -- set by
    # convert.py's per-format streaming gate; ``sources`` may hold released tensors.
    streamed: bool = False
    # the expert (kind, kernel) the banks were packed for; None for the legacy providers
    kind: QuantKind | None = None
    kernel: str | None = None
    layout: dict | None = None


def _dummy_fill(role: str, tensor: torch.Tensor) -> None:
    """Random but finite bank contents for --use-dummy-weight."""
    if role.endswith("_scale"):
        if tensor.dtype is torch.uint8:
            tensor.fill_(127)  # e8m0 exponent code for 1.0
        else:
            tensor.fill_(1.0)
    elif role.endswith("_global"):
        tensor.fill_(0.01)
    elif tensor.dtype in (torch.uint8, torch.int32):
        tensor.view(torch.uint8).random_(0, 256)
    elif tensor.dtype is torch.float8_e4m3fn:
        tensor.view(torch.uint8).random_(0, 16)  # small codes, no NaN / inf
    else:
        tensor.normal_()


def build_expert_banks(
    method,
    num_layers: int,
    pieces,
    *,
    device: torch.device,
    layer_sink=None,
    dummy: bool = False,
) -> ExpertBanks:
    """Fill host banks in the kernel's layout from a stream of expert pieces.

    ``pieces`` yields ``(layer_id, e0, e1, {role: tensor[e1 - e0, ...]})`` in any order;
    each batch is packed in place into rows ``e0:e1`` of that layer's banks. A layer is
    complete once its ``num_experts`` rows have arrived: with ``layer_sink=None`` its banks
    are pinned in the background, otherwise the sink receives them (converter). ``dummy``
    skips the pieces and fills the banks with finite random contents.
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, pin_banks
    from freetoken.moe.legacy_format import legacy_format_for

    kernel = method.kernel
    layout = method.layout()
    E = method.cfg.num_experts
    specs = {role: ((E, *spec.shape), spec.dtype) for role, spec in layout.items() if not spec.resident}
    hb = alloc_layer_banks(specs, num_layers)
    banks = {role: [b.tensor for b in hb[role]] for role in specs}
    alphas = {
        role: torch.empty(num_layers * E, dtype=spec.dtype, device=device)
        for role, spec in layout.items() if spec.resident
    }

    if dummy:
        for role, per_layer in banks.items():
            for tensor in per_layer:
                _dummy_fill(role, tensor)
        for alpha in alphas.values():
            alpha.fill_(1.0)
        if torch.cuda.is_available():
            pin_banks(hb)
        return ExpertBanks(
            legacy_format_for(method.kind, kernel.name), banks,
            gate_up_alpha=alphas.get("gate_up_alpha"), down_alpha=alphas.get("down_alpha"),
            kind=method.kind, kernel=kernel.name, layout=layout,
        )

    def _fill(sink) -> None:
        tracker = LayerCompletionTracker(E, hb, sink) if sink is not None else None
        # a reader that skips a layer or mislabels a piece must fail here, not serve uninitialized rows
        written = torch.zeros(num_layers, E, dtype=torch.int32)
        for layer_id, e0, e1, piece in pieces:
            if not (0 <= layer_id < num_layers and 0 <= e0 < e1 <= E):
                raise ValueError(f"expert piece out of range: layer {layer_id}, experts {e0}:{e1} of {num_layers} x {E}")
            # refuse before writing: a duplicate row would also complete the layer early and hand the sink a half-filled bank
            if written[layer_id, e0:e1].any():
                raise ValueError(f"expert rows written more than once: layer {layer_id}, experts {e0}:{e1}")
            written[layer_id, e0:e1] = 1
            out = {role: banks[role][layer_id][e0:e1] for role in specs}
            got = method.pack(piece, out)
            for role, values in got.items():
                alphas[role][layer_id * E + e0 : layer_id * E + e1] = values.to(alphas[role].dtype)
            if tracker is not None:
                for _ in range(e1 - e0):
                    tracker.note(layer_id)
        missing = (written == 0).nonzero().tolist()
        if missing:
            raise ValueError(f"expert banks were not filled: {len(missing)} (layer, expert) rows missing (first {missing[:4]})")

    if layer_sink is not None:
        _fill(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _fill(pins)
    else:
        _fill(None)

    return ExpertBanks(
        legacy_format_for(method.kind, kernel.name), banks,
        gate_up_alpha=alphas.get("gate_up_alpha"), down_alpha=alphas.get("down_alpha"),
        streamed=layer_sink is not None, kind=method.kind, kernel=kernel.name, layout=layout,
    )


_PARALLEL_CHUNK = 8 << 20  # default O_DIRECT chunk for the parallel reader


def _q4_0_banks(model_path, model_config, device, dtype, dummy, parallel=False, workers=8, chunk=_PARALLEL_CHUNK, decode_target="gpu", layer_sink=None) -> ExpertBanks:
    if parallel:
        raise NotImplementedError(
            "parallel reader not implemented for q4_0: GGUF is a single packed file "
            "(not safetensors), so the common reader doesn't apply -- it needs a GGUF-native "
            "parallel reader (parse the tensor table, chunked O_DIRECT over the one file)"
        )
    from freetoken.models.weight import load_q4_0_moe_expert_sources

    # Native GGUF Q4_0 routed experts: packed block bytes streamed to the GPU and
    # dequantized inside the borrowed ggml MoE kernels (no bf16 expert copy). Banks are
    # per-layer HostBanks (pin-after-fill), so conversion streams each completed layer's
    # gate_up + down straight through the sink (dummy fabricates in one shot -> not streamed).
    sink = None if dummy else layer_sink
    sources = load_q4_0_moe_expert_sources(model_path, model_config, dummy=dummy, layer_sink=sink)
    return ExpertBanks(
        "q4_0", {name: sources[name] for name in _BANK_SCHEMAS["q4_0"]}, streamed=sink is not None
    )


# expert formats that still load through their own provider (GGUF)
_PROVIDERS = {
    "q4_0": _q4_0_banks,
}


def _legacy_expert_banks(model_path, model_config, device, dtype, dummy, parallel, workers, chunk, decode_target="gpu", layer_sink=None) -> ExpertBanks:
    expert_quant = model_config.expert_quant
    if expert_quant not in _PROVIDERS:
        raise ValueError(
            f"{expert_quant!r} experts load through their MoE quant method; "
            f"only {sorted(_PROVIDERS)} still have a format provider"
        )
    return _PROVIDERS[expert_quant](
        model_path, model_config, device, dtype, dummy,
        parallel=parallel, workers=workers, chunk=chunk, decode_target=decode_target,
        layer_sink=layer_sink,
    )


def _method_expert_banks(model_path, model_config, method, device, dummy, parallel, workers, chunk, layer_sink=None) -> ExpertBanks:
    from freetoken.moe.expert_pieces import iter_expert_pieces

    num_layers = model_config.num_moe_layers
    if dummy:
        return build_expert_banks(method, num_layers, None, device=device, dummy=True)
    pieces = iter_expert_pieces(
        model_path, model_config, method.kind, parallel=parallel, workers=workers, chunk=chunk
    )
    return build_expert_banks(method, num_layers, pieces, device=device, layer_sink=layer_sink)


def _host_ram_fits_parallel(model_path: str) -> bool:
    """Best-effort: can free host RAM hold the expert banks plus the parallel reader's one
    extra (non-reclaimable) whole-shard buffer? Unknown (non-local path / no /proc) -> True,
    i.e. keep the fast path. Banks ~= checkpoint size (experts dominate); transient ~= the
    largest shard. Uses MemAvailable (counts reclaimable cache) -- the OOM-relevant figure."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if avail is None:
        return True
    try:  # resolve a hub id to its local cache dir (no-op for a local path) so glob sees the shards
        from freetoken.utils.hf import download_hf_weight

        model_path = download_hf_weight(model_path)
    except Exception:
        return True
    sizes = [os.path.getsize(p) for p in glob.glob(os.path.join(model_path, "*.safetensors"))]
    if not sizes:
        return True
    return avail > sum(sizes) + max(sizes)


def ftw_bank_bytes(model_path: str) -> int | None:
    """Total expert-bank bytes of an FTW checkpoint, from its metadata (no bank IO).
    ``None`` when the checkpoint is not FTW -- callers that size things pre-load (auto split residency) then leave the load unchanged."""
    import json

    meta = os.path.join(model_path, "freetoken_weight.json")
    if not os.path.isfile(meta):
        return None
    with open(meta, encoding="utf-8") as f:
        tensors = json.load(f).get("tensors", [])
    return sum(t["nbytes"] for t in tensors if t.get("kind") == "experts_bank")


def bank_bytes_estimate(model_config, method=None) -> int | None:
    """Estimated total expert-bank bytes of a raw checkpoint before loading it.

    With a bound expert ``method`` the kernel's layout gives the exact host bytes; otherwise the
    format-tag table sizes the GGUF format. ``None`` for unknown formats or missing dims
    (callers then skip the pre-load sizing)."""
    layers = getattr(model_config, "num_moe_layers", None)
    if method is not None and layers:
        per_expert = sum(
            math.prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()
            for spec in method.layout().values() if not spec.resident
        )
        return layers * method.cfg.num_experts * per_expert
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (
        getattr(model_config, "moe_weight_format", None) or "bf16"
    )
    per_expert = _BANK_BYTES_PER_EXPERT.get(fmt)
    layers = getattr(model_config, "num_moe_layers", None)
    experts = getattr(model_config, "num_experts", None)
    hidden = getattr(model_config, "hidden_size", None)
    inter = getattr(model_config, "moe_intermediate_size", None)
    if per_expert is None or not all((layers, experts, hidden, inter)):
        return None
    return layers * experts * per_expert(hidden, inter)


def load_expert_banks(
    model_path: str,
    model_config,
    *,
    method=None,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool | None = None,
    workers: int = 8,
    chunk: int = _PARALLEL_CHUNK,
    decode_target: str = "gpu",
    layer_sink=None,
    layer_residency: list[str] | None = None,
) -> ExpertBanks:
    """Load (or fabricate, with ``dummy=True``) the expert banks. Two paths, both returning
    the same normalized ``ExpertBanks`` and both pinning after fill:

    * **Fast path (FTW)**: if ``model_path`` is a converted FTW checkpoint, read its
      repacked banks directly (contiguous chunked O_DIRECT). No auto-conversion.
    * **Slow path** (the original checkpoint): auto-pick **parallel** (the common parallel chunked
      O_DIRECT reader) when experts are stored as many small tensors -- the serial read is
      slow there -- else the **serial baseline** (packed experts: serial already saturates,
      parallel only adds read amplification). parallel unavailable for a quant falls back to serial.

    ``parallel`` overrides the slow-path auto-pick: ``None`` = auto (production), ``True`` /
    ``False`` = force parallel / serial (used by the loader benchmark and the converter).

    ``layer_sink`` (the converter only): forwarded to whichever provider is picked; a
    provider only engages it (and reports ``ExpertBanks.streamed=True``) for its own
    streamable formats, so callers must check ``streamed`` rather than assume it fired.

    ``method`` (the bound expert quant method of the model's offload layers) selects
    the generic path: the family's pieces packed by the method's kernel. Without it only the
    GGUF q4_0 format loads, through its own provider.

    ``layer_residency``: per-layer ``HostResidency`` labels applied at settle time -- explicitly on the FTW fast path, ambiently (``requested_residency``) in the slow-path providers.
    Applied labels are echoed on ``ExpertBanks.layer_residency``; a loader that settles some other way leaves it ``None`` (CPU-layer decode still works on pinned banks, it just saves no pin quota).
    """
    from freetoken.checkpoint.ftw import is_ftw_checkpoint, load_ftw_banks

    if model_path and is_ftw_checkpoint(model_path) and not dummy:
        banks = load_ftw_banks(
            model_path, num_layers=model_config.num_moe_layers, workers=workers, chunk=chunk,
            layer_residency=layer_residency,
        )
        if banks is not None:
            logger.info_rank0(f"expert banks: FTW fast path (FTW checkpoint {model_path})")
            return banks

    if parallel and not _PARALLEL_READER_SUPPORTED:
        logger.warning_rank0(
            "expert banks: parallel O_DIRECT reader unsupported on this platform "
            "(no os.O_DIRECT/preadv) -> serial build"
        )
        parallel = False

    auto = parallel is None
    if auto:
        from freetoken.models.weight import experts_scattered

        parallel = _PARALLEL_READER_SUPPORTED and not dummy and experts_scattered(model_path)
        # Low-RAM fallback: the parallel reader holds whole-shard ANONYMOUS buffers
        # (non-reclaimable) on top of the ~bank-sized resident set, so on a memory-tight box
        # it OOMs where the serial path (reclaimable file mmap) survives. Drop to serial when
        # free RAM can't cover the banks + one shard's transient. (--expert-load serial/parallel
        # bypass this by forcing ``parallel`` explicitly.)
        if parallel and not _host_ram_fits_parallel(model_path):
            logger.warning_rank0(
                "expert banks: low free RAM -> serial build (avoids parallel-reader OOM; "
                "override with --expert-load parallel)"
            )
            parallel = False
    logger.info_rank0(f"expert banks: slow path ({'parallel' if parallel else 'serial'} build)")
    # parallel's reader resolves hub ids + handles single-file/no-index checkpoints, so it won't
    # OSError on those (which would leak the banks it pre-allocated, since host banks live for
    # the process). Only NotImplementedError (quant has no parallel reader; raised before any
    # allocation) falls back to serial.
    from freetoken.moe.host_banks import requested_residency

    def _build(par: bool) -> ExpertBanks:
        if method is not None:
            return _method_expert_banks(model_path, model_config, method, device, dummy, par, workers, chunk, layer_sink)
        return _legacy_expert_banks(model_path, model_config, device, dtype, dummy, par, workers, chunk, decode_target, layer_sink)

    with requested_residency(layer_residency) as residency_plan:
        try:
            banks = _build(parallel)
        except NotImplementedError as exc:
            if not parallel:
                raise
            logger.warning_rank0(f"parallel reader unavailable ({exc}); falling back to serial build")
            banks = _build(False)
    return _echo_residency(banks, layer_residency, residency_plan)


def _echo_residency(banks: ExpertBanks, requested, plan) -> ExpertBanks:
    """Stamp an honored residency request onto the ExpertBanks; keep None (and warn) when no settle point consulted the plan."""
    if requested is None or banks.layer_residency is not None:
        return banks
    if plan is not None and plan.applied:
        import dataclasses

        labels = [plan.actual.get(i, r) for i, r in enumerate(requested)]
        downgraded = [i for i, r in enumerate(requested) if labels[i] != r]
        if downgraded:
            logger.warning_rank0(
                f"--moe-cpu-layers: layers {downgraded} settled pageable instead of "
                f"OS-locked (lock failed); they still decode on the CPU executor but "
                f"may swap under memory pressure"
            )
        return dataclasses.replace(banks, layer_residency=labels)
    from freetoken.moe.host_banks import HostResidency

    if any(r != HostResidency.PINNED.value for r in requested):
        logger.warning_rank0(
            "--moe-cpu-layers: this checkpoint's bank loader settles banks without "
            "per-layer residency (pre-pins everything); CPU-layer decode still works "
            "but saves no pinned quota"
        )
    return banks
