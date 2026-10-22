"""Expert banks for the offload MoE cache: load, pack and pin the routed experts.

The expert kernel (``QuantMethod.kernel``) owns the bank layout and the pack step; the
checkpoint side delivers pieces (``moe.expert_pieces``) and this module fills the pinned host
banks from them (``build_expert_banks``). GGUF experts keep their packed ggml rows
(``models.gguf.experts``).
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.utils import init_logger

from .host_banks import HostBank
from .offload_cache import _BANK_BYTES_PER_EXPERT, placeholder_bank

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
    # per-layer quant tag (e.g. the gate_up and down GGUF types), for layers whose banks match in dtype and row shape but not in quant; None = one quant
    layer_tags: list | None = None
    # per bank, per layer: bytes past an expert matrix's last row that the kernel reads and that must read as zeros (ggml MMQ); None = no such reads
    row_tail_bytes: dict[str, list[int]] | None = None
    # per-layer quant_format of experts in several formats (``quant_format``/``kind``/``kernel`` then name the most common one); None = every layer is ``quant_format``
    layer_formats: list[str] | None = None
    # the residency plan holding DEFERRED layers' banks, for host_banks.probe_pin; None when nothing was deferred
    deferred_plan: object | None = None


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
    resident: bool = False,
    layers: list[int] | None = None,
) -> ExpertBanks:
    """Fill host banks in the kernel's layout from a stream of expert pieces.

    ``pieces`` yields ``(layer_id, e0, e1, {role: tensor[e1 - e0, ...]})`` in any order;
    each batch is packed in place into rows ``e0:e1`` of that layer's banks. A layer is
    complete once its ``num_experts`` rows have arrived: with ``layer_sink=None`` its banks
    are pinned in the background, otherwise the sink receives them (converter). ``dummy``
    skips the pieces and fills the banks with finite random contents.
    ``resident`` packs into banks on ``device`` instead of host banks (the resident 'fused' experts).
    ``layers`` fills only those bank layers (the ones ``method`` serves, for experts of several
    formats): pieces of other layers are skipped and the other layers' sources are None.
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, pin_banks
    from freetoken.moe.legacy_format import legacy_format_for

    kernel = method.kernel
    layout = method.layout()
    E = method.cfg.num_experts
    ids = list(range(num_layers)) if layers is None else sorted(layers)
    filled = set(ids)
    specs = {role: ((E, *spec.shape), spec.dtype) for role, spec in layout.items() if not spec.resident}
    if resident:
        assert layer_sink is None
        hb = None
        banks = {role: {l: torch.empty(shape, dtype=dtype, device=device) for l in ids} for role, (shape, dtype) in specs.items()}
    else:
        hb = {role: {l: HostBank(shape, dtype) for l in ids} for role, (shape, dtype) in specs.items()}
        banks = {role: {l: b.tensor for l, b in per_layer.items()} for role, per_layer in hb.items()}
    alphas = {
        role: torch.empty(num_layers * E, dtype=spec.dtype, device=device)
        for role, spec in layout.items() if spec.resident
    }

    def _sources() -> dict[str, list]:
        return {role: [per_layer.get(l) for l in range(num_layers)] for role, per_layer in banks.items()}

    if dummy:
        for role, per_layer in banks.items():
            for tensor in per_layer.values():
                _dummy_fill(role, tensor)
        for alpha in alphas.values():
            alpha.fill_(1.0)
        if hb is not None and torch.cuda.is_available():
            pin_banks(hb)
        return ExpertBanks(
            legacy_format_for(method.kind, kernel.name), _sources(),
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
            if layer_id not in filled:
                continue
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
        missing = (written[ids] == 0).nonzero().tolist()
        if missing:
            missing = [[ids[l], e] for l, e in missing]
            raise ValueError(f"expert banks were not filled: {len(missing)} (layer, expert) rows missing (first {missing[:4]})")

    if layer_sink is not None:
        _fill(layer_sink)
    elif torch.cuda.is_available() and not resident:
        with PinPipeline() as pins:
            _fill(pins)
    else:
        _fill(None)

    return ExpertBanks(
        legacy_format_for(method.kind, kernel.name), _sources(),
        gate_up_alpha=alphas.get("gate_up_alpha"), down_alpha=alphas.get("down_alpha"),
        streamed=layer_sink is not None, kind=method.kind, kernel=kernel.name, layout=layout,
    )


def method_groups(layer_methods) -> list[tuple]:
    """``[(method, [bank layers])]`` of the layers' expert methods, one entry per (kind, kernel), most layers first."""
    groups: dict = {}
    for layer, method in enumerate(layer_methods):
        groups.setdefault((method.kind, method.kernel.name), (method, []))[1].append(layer)
    return sorted(groups.values(), key=lambda g: (-len(g[1]), g[1][0]))


def is_mixed(layer_methods) -> bool:
    return layer_methods is not None and len(method_groups(layer_methods)) > 1


def merge_expert_banks(parts: list[ExpertBanks], layer_methods, num_experts: int) -> ExpertBanks:
    """One ``ExpertBanks`` over the per-format ``parts`` (each filled only on its layers): the banks are the
    union of the roles, a layer without a role gets a zero-byte placeholder (see ``OffloadMoeCache.layer_layouts``)."""
    from freetoken.moe.legacy_format import legacy_format_for

    num_layers = len(layer_methods)
    sources: dict[str, list] = {}
    for part in parts:
        if part.gate_up_alpha is not None or part.down_alpha is not None:
            raise ValueError(f"experts of several formats cannot use the {part.kernel} kernel, which keeps GPU-resident alphas")
        for role, per_layer in part.sources.items():
            merged = sources.setdefault(role, [None] * num_layers)
            for layer, tensor in enumerate(per_layer):
                if tensor is not None:
                    assert merged[layer] is None, (role, layer)
                    merged[layer] = tensor
    for per_layer in sources.values():
        for layer, tensor in enumerate(per_layer):
            if tensor is None:
                per_layer[layer] = placeholder_bank(num_experts)
    formats = [legacy_format_for(m.kind, m.kernel.name) for m in layer_methods]
    primary = parts[0]
    return ExpertBanks(
        primary.quant_format, sources, streamed=all(p.streamed for p in parts),
        kind=primary.kind, kernel=primary.kernel, layer_tags=formats, layer_formats=formats,
    )


def _mixed_expert_banks(model_path, model_config, layer_methods, device, dummy, parallel, workers, chunk, layer_sink=None) -> ExpertBanks:
    from freetoken.moe.expert_pieces import iter_expert_pieces

    num_layers = model_config.num_moe_layers
    assert len(layer_methods) == num_layers, (len(layer_methods), num_layers)
    groups = method_groups(layer_methods)
    # resolve every format's reader before any bank exists, so a missing parallel reader falls back to serial with nothing allocated
    readers = [
        None if dummy else iter_expert_pieces(
            model_path, model_config, method.kind, layers=layers, parallel=parallel, workers=workers, chunk=chunk
        )
        for method, layers in groups
    ]
    parts = []
    for (method, layers), pieces in zip(groups, readers):
        parts.append(build_expert_banks(
            method, num_layers, pieces, device=device, layer_sink=layer_sink, dummy=dummy, layers=layers,
        ))
        logger.info_rank0(f"expert banks: {method.kind} / {method.kernel.name} on {len(layers)} of {num_layers} MoE layers")
    return merge_expert_banks(parts, layer_methods, model_config.num_experts)


_PARALLEL_CHUNK = 8 << 20  # default O_DIRECT chunk for the parallel reader


def _method_expert_banks(model_path, model_config, method, device, dummy, parallel, workers, chunk, layer_sink=None, resident=False) -> ExpertBanks:
    from freetoken.moe.expert_pieces import iter_expert_pieces

    num_layers = model_config.num_moe_layers
    if dummy:
        return build_expert_banks(method, num_layers, None, device=device, dummy=True, resident=resident)
    pieces = iter_expert_pieces(
        model_path, model_config, method.kind, parallel=parallel, workers=workers, chunk=chunk
    )
    return build_expert_banks(method, num_layers, pieces, device=device, layer_sink=layer_sink, resident=resident)


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


def _method_expert_bytes(method) -> int:
    return method.cfg.num_experts * sum(
        math.prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()
        for spec in method.layout().values() if not spec.resident
    )


def bank_bytes_estimate(model_config, method=None, layer_methods=None) -> int | None:
    """Estimated total expert-bank bytes of a raw checkpoint before loading it.

    With a bound expert ``method`` the kernel's layout gives the exact host bytes (``layer_methods``,
    one per MoE layer, for experts of several formats); otherwise the format-tag table sizes the
    format. ``None`` for unknown formats or missing dims (callers then skip the pre-load sizing)."""
    layers = getattr(model_config, "num_moe_layers", None)
    if layer_methods is not None:
        return sum(_method_expert_bytes(m) for m in layer_methods)
    if method is not None and layers:
        return layers * _method_expert_bytes(method)
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
    dummy: bool = False,
    parallel: bool | None = None,
    workers: int = 8,
    chunk: int = _PARALLEL_CHUNK,
    layer_sink=None,
    layer_residency: list[str] | None = None,
    resident: bool = False,
    layer_methods: list | None = None,
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
    the generic path: the family's pieces packed by the method's kernel. Without it only an
    FTW checkpoint loads.

    ``layer_residency``: per-layer ``HostResidency`` labels applied at settle time -- explicitly on the FTW fast path, ambiently (``requested_residency``) in the slow-path providers.
    Applied labels are echoed on ``ExpertBanks.layer_residency``; a loader that settles some other way leaves it ``None`` (CPU-layer decode still works on pinned banks, it just saves no pin quota).

    ``resident`` (with ``method``): the banks land on ``device`` for the resident 'fused' experts; no host bank outlives the load.

    ``layer_methods`` (one bound method per MoE layer) serves experts of several formats: each
    format fills its own layers and the banks merge over the union of roles (``merge_expert_banks``).
    """
    from freetoken.checkpoint.ftw import is_ftw_checkpoint, load_ftw_banks, load_ftw_banks_to_device

    mixed = is_mixed(layer_methods)
    if mixed and (resident or layer_residency is not None):
        raise ValueError("experts of several formats are served from the offload slot cache only (no resident experts, no --moe-cpu-layers)")
    if resident:
        assert method is not None and layer_sink is None and layer_residency is None
    if model_path and is_ftw_checkpoint(model_path) and not dummy:
        if resident:
            banks = load_ftw_banks_to_device(model_path, num_layers=model_config.num_moe_layers, device=device, workers=workers, chunk=chunk)
        else:
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
        if mixed:
            return _mixed_expert_banks(model_path, model_config, layer_methods, device, dummy, par, workers, chunk, layer_sink)
        if method is not None and method.kind is QuantKind.GGUF:
            from freetoken.models.gguf.experts import load_gguf_expert_banks

            # each layer keeps its own ggml types, so the banks come from the file, not from one method's layout
            return load_gguf_expert_banks(model_path, model_config.num_moe_layers, first_layer=model_config.first_k_dense_replace, dummy=dummy, device=device if resident else None)
        if method is None:
            raise ValueError("the experts of a raw checkpoint are packed by the offload layers' MoE quant method; pass method=")
        return _method_expert_banks(model_path, model_config, method, device, dummy, par, workers, chunk, layer_sink, resident)

    with requested_residency(layer_residency) as residency_plan:
        try:
            banks = _build(parallel)
        except NotImplementedError as exc:
            if not parallel:
                raise
            logger.warning_rank0(f"parallel reader unavailable ({exc}); falling back to serial build")
            banks = _build(False)
    return _echo_residency(banks, layer_residency, residency_plan)


def pin_probe_enabled(config, mixed_formats: bool) -> bool:
    """Whether ``--moe-cpu-layers auto`` finds the pin limit by pinning until refused.

    Only where pinning is known to be capped; FREETOKEN_PIN_BUDGET_GB fixes the limit instead (then a failed pin is an error)."""
    from freetoken.checkpoint.ftw import is_ftw_checkpoint
    from freetoken.moe.host_banks import pin_capped

    return (
        (config.moe_cpu_layers or "").strip() == "auto"
        and pin_capped()
        and not os.environ.get("FREETOKEN_PIN_BUDGET_GB")
        and config.moe_strategy in ("offload", "hybrid")
        and not mixed_formats
        and not config.use_dummy_weight
        and not (config.model_path and is_ftw_checkpoint(config.model_path))  # FTW settles its banks itself
    )


def probe_banks(banks: ExpertBanks, num_layers: int) -> tuple[ExpertBanks, frozenset[int]]:
    """Settle DEFERRED banks with host_banks.probe_pin; returns the banks and the OS-locked layers."""
    import dataclasses
    import time

    from freetoken.moe.host_banks import HostResidency, probe_pin

    plan = banks.deferred_plan
    if plan is None:
        return banks, frozenset()  # the loader settled the banks itself (pinned)
    t0 = time.perf_counter()
    labels = probe_pin(plan, num_layers)
    locked = frozenset(i for i, r in enumerate(labels) if r != HostResidency.PINNED.value)
    pinned_bytes = sum(b.nbytes for l, bs in plan.deferred.items() if l not in locked for b in bs)
    logger.info_rank0(
        f"--moe-cpu-layers auto (pin probe): pinned {num_layers - len(locked)} MoE layers ({pinned_bytes / 2**30:.2f} GiB) "
        f"in {time.perf_counter() - t0:.1f}s, locked {len(locked)} {sorted(locked)}"
    )
    return dataclasses.replace(banks, layer_residency=labels, deferred_plan=None), locked


def _echo_residency(banks: ExpertBanks, requested, plan) -> ExpertBanks:
    """Stamp an honored residency request onto the ExpertBanks; keep None (and warn) when no settle point consulted the plan."""
    if requested is None or banks.layer_residency is not None:
        return banks
    if plan is not None and plan.applied:
        import dataclasses

        if plan.deferred:
            return dataclasses.replace(banks, layer_residency=list(requested), deferred_plan=plan)
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


def attach_resident_banks(layers, banks: ExpertBanks) -> None:
    """Give each resident MoE layer its rows of the device ``banks``; bank layer ``i`` goes to the layer whose ``layer_id`` is ``i``."""
    ids = sorted(layer.layer_id for layer in layers if layer.layer_id is not None)
    num_layers = len(next(iter(banks.sources.values())))
    if ids != list(range(num_layers)) or len(ids) != len(layers):
        raise ValueError(f"resident MoE layer ids {ids} do not cover the {num_layers} expert bank layers")
    for layer in layers:
        i, e = layer.layer_id, layer.num_experts
        layout = layer.quant_method.layout()
        rows = {role: (e, *spec.shape) for role, spec in layout.items() if not spec.resident}
        if set(banks.sources) != set(rows):
            raise ValueError(f"expert banks hold {sorted(banks.sources)}, the {layer.quant_method.kernel.name} kernel reads {sorted(rows)}")
        for role, shape in rows.items():
            bank = banks.sources[role][i]
            if tuple(bank.shape) != shape or bank.dtype != layout[role].dtype:
                raise ValueError(
                    f"bank {role!r} of layer {i} is {tuple(bank.shape)} {bank.dtype} but the {layer.quant_method.kernel.name} kernel "
                    f"wants {shape} {layout[role].dtype}; the banks were packed for another kernel or model"
                )
        alphas = []
        # apply() reads view.alphas as (gate_up, down)
        for role in (role for role in ("gate_up_alpha", "down_alpha") if role in layout):
            a = getattr(banks, role)
            if a is None or a.numel() != num_layers * e or a.dtype != layout[role].dtype:
                raise ValueError(f"the {layer.quant_method.kernel.name} kernel needs {num_layers * e} {layout[role].dtype} {role} values, the expert banks carry {None if a is None else (a.numel(), a.dtype)}")
            alphas.append(a)
        layer._expert_banks = {role: per_layer[i] for role, per_layer in banks.sources.items()}
        layer._expert_alphas = tuple(a[i * e : (i + 1) * e] for a in alphas) or None
