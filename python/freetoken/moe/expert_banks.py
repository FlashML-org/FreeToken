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

from .host_banks import alloc_layer_banks  # noqa: F401  (converter tooling allocates uniform banks through it)
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
    # 冷压缩行映射 [num_layers, num_experts] int32（CPU 张量）：pinned 专家 -> -1，
    # 冷专家 -> 其冷行号（该层冷专家按 id 升序的序号）。未启用钉住时为 None。
    # 每层 bank 行数为 [num_experts - K_l, *row]，行号 == cold_row 值。
    cold_row: torch.Tensor | None = field(default=None)


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
    pin_sets: dict[int, list[int]] | None = None,
    pin_sink=None,
) -> ExpertBanks:
    """Fill host banks in the kernel's layout from a stream of expert pieces.

    ``pieces`` yields ``(layer_id, e0, e1, {role: tensor[e1 - e0, ...]})`` in any order;
    each batch is packed in place into rows ``e0:e1`` of that layer's banks. A layer is
    complete once its ``num_experts`` rows have arrived: with ``layer_sink=None`` its banks
    are pinned in the background, otherwise the sink receives them (converter). ``dummy``
    skips the pieces and fills the banks with finite random contents.

    ``pin_sets``（显存钉住，v1）: ``{layer_id -> 按热度序的钉住专家 id 列表}``。命中的层
    冷压缩：bank 只有 ``[E - K_l, *row]`` 行，行号 = 冷行号（冷专家按 id 升序压缩）；钉住
    专家的权重改为 pack 进 ``[K_l, *row]`` 的 per-layer host 暂存。该层全部行读完后以
    ``pin_sink(layer_id, {role: 暂存})`` 一次性移交（sink 必须在返回前完成 H2D 或持有一份
    拷贝——暂存随调用结束释放，host 峰值占用 ≤ 单层钉住字节）。钉住 piece 仍会被读盘
    （reader 按范围整段吐出），这是 v1 接受的 IO 浪费，v2 可在读取层过滤。返回的
    ``ExpertBanks.cold_row`` 为 ``[L, E] int32``（pinned -> -1）。``layer_sink`` 与
    ``pin_sets`` 互斥（converter 不参与钉住）。
    """
    from freetoken.moe.host_banks import HostBank, LayerCompletionTracker, PinPipeline, pin_banks
    from freetoken.moe.legacy_format import legacy_format_for

    if layer_sink is not None and pin_sets:
        raise ValueError("expert bank converter (layer_sink) 与显存钉住 (pin_sets) 互斥")

    kernel = method.kernel
    layout = method.layout()
    E = method.cfg.num_experts
    pin_counts = [len((pin_sets or {}).get(l, ())) for l in range(num_layers)]
    for layer_id, experts in (pin_sets or {}).items():
        if not 0 <= layer_id < num_layers:
            raise ValueError(f"pin 层号越界: {layer_id} (num_layers={num_layers})")
        if len(set(experts)) != len(experts) or any(not 0 <= e < E for e in experts):
            raise ValueError(f"layer {layer_id} 的钉住专家 id 非法/重复: {experts}")
        if len(experts) >= E:
            raise ValueError(
                f"layer {layer_id} 的钉住数 {len(experts)} 必须小于专家数 {E}（冷 bank 至少 1 行）"
            )
    # 冷行映射只建一次，两条路径（dummy / pieces）共用
    cold_row = _cold_row_from_pin_sets(pin_sets, num_layers, E)
    specs = {role: ((E, *spec.shape), spec.dtype) for role, spec in layout.items() if not spec.resident}
    hb = {
        role: [
            # 冷压缩：钉住层的 bank 只有 E-K 行（行号 = cold_row），其余层保持全量 E 行
            HostBank((E - pin_counts[l], *shape[1:]), dtype)
            for l in range(num_layers)
        ]
        for role, (shape, dtype) in specs.items()
    }
    banks = {role: [b.tensor for b in hb[role]] for role in specs}
    alphas = {
        role: torch.empty(num_layers * E, dtype=spec.dtype, device=device)
        for role, spec in layout.items() if spec.resident
    }
    # per-layer 钉住暂存：host 峰值 ≤ 单层钉住字节（每层移交 pin_sink 后即释放）
    pin_stage: dict[int, dict[str, torch.Tensor]] = {}
    pin_read: dict[int, int] = {}

    def _pin_stage_for(layer_id: int) -> dict[str, torch.Tensor]:
        """取（惰性建）该层的钉住暂存 {role: [K_l, *row]}，并清零其已读行计数。"""
        stage = pin_stage.get(layer_id)
        if stage is None:
            k = pin_counts[layer_id]
            stage = {
                role: torch.empty((k, *shape[1:]), dtype=dtype)
                for role, (shape, dtype) in specs.items()
            }
            pin_stage[layer_id] = stage
            pin_read[layer_id] = 0
        return stage

    def _release_pin_stage(layer_id: int) -> None:
        """该层全部行读毕：把钉住暂存一次性移交 pin_sink 后丢弃（host 峰值 ≤ 单层钉住字节）。"""
        stage = pin_stage.pop(layer_id)
        if pin_sink is not None:
            pin_sink(layer_id, stage)

    if dummy:
        for role, per_layer in banks.items():
            for tensor in per_layer:
                _dummy_fill(role, tensor)
        for alpha in alphas.values():
            alpha.fill_(1.0)
        for layer_id in range(num_layers):
            if pin_counts[layer_id]:
                stage = _pin_stage_for(layer_id)
                for role, tensor in stage.items():
                    _dummy_fill(role, tensor)
                _release_pin_stage(layer_id)
        if torch.cuda.is_available():
            pin_banks(hb)
        return ExpertBanks(
            legacy_format_for(method.kind, kernel.name), banks,
            gate_up_alpha=alphas.get("gate_up_alpha"), down_alpha=alphas.get("down_alpha"),
            kind=method.kind, kernel=kernel.name, layout=layout, cold_row=cold_row,
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
            layer_pins = set((pin_sets or {}).get(layer_id, ()))
            pin_list = (pin_sets or {}).get(layer_id) or []
            # 冷压缩 pack：把 [e0, e1) 切成极大冷专家连续段，逐段 pack 进对应冷行区间；
            # 钉住专家单独成段 pack 进 per-layer 暂存（cold_row 对钉住专家为 -1，不占冷行）。
            # 暂存行序 == pin list 顺序（slot 分配按同一序号）。
            runs = _split_cold_runs(e0, e1, layer_pins)
            for a, b in runs:
                out, src_piece = _run_views(a, b, e0, piece, banks, layer_id, pin_list, _pin_stage_for)
                got = method.pack(src_piece, out)
                for role, values in got.items():
                    alphas[role][layer_id * E + a : layer_id * E + b] = values.to(alphas[role].dtype)
            if tracker is not None:
                for _ in range(e1 - e0):
                    tracker.note(layer_id)
            if pin_counts[layer_id]:
                pin_read[layer_id] = pin_read.get(layer_id, 0) + (e1 - e0)
                if pin_read[layer_id] == E:
                    _release_pin_stage(layer_id)
        missing = (written == 0).nonzero().tolist()
        if missing:
            raise ValueError(f"expert banks were not filled: {len(missing)} (layer, expert) rows missing (first {missing[:4]})")
        # 暂存未移交 = 该层行没读全却提前结束（上面 missing 已兜底）或 sink 缺失
        if pin_stage and pin_sink is None:
            raise ValueError("pin_sets 给定但 pin_sink 缺失：钉住权重无处移交")

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
        cold_row=cold_row,
    )


def _cold_row_from_pin_sets(pin_sets: dict[int, list[int]] | None, num_layers: int, num_experts: int) -> torch.Tensor:
    """
    Business Logic（为什么需要这个函数）:
        build_expert_banks 的冷压缩行号与 OffloadMoeCache 的 remap/materialize 语义
        必须同源；hot_pin.cold_row_from_pins 是该语义的唯一实现，这里只做薄封装，
        避免两套手写映射漂移。

    Code Logic（这个函数做什么）:
        把 {layer -> 钉住 id 列表} 还原成 [num_layers][K] 矩形输入（缺失层补空），
        调用 freetoken.moe.hot_pin.cold_row_from_pins 得到 [L, E] int32（pinned -> -1）。
        没有任何钉住项（pin_sets 为 None 或全空）时返回 None——ExpertBanks.cold_row
        为 None 即"未启用钉住"的契约标记。
    """
    from freetoken.moe.hot_pin import cold_row_from_pins

    pins = [list((pin_sets or {}).get(l, ())) for l in range(num_layers)]
    if not any(pins):
        return None
    return cold_row_from_pins(pins, num_experts)


def _split_cold_runs(e0: int, e1: int, layer_pins: set[int]) -> list[tuple[int, int]]:
    """
    Business Logic（为什么需要这个函数）:
        pack 契约要求目标行区间连续；冷压缩后钉住专家在专家 id 轴上挖出空洞，
        必须把 piece 切成极大冷专家连续段与单专家钉住段，段内 (源行, 目标行)
        才能保持等宽直线拷贝。

    Code Logic（这个函数做什么）:
        扫描 [e0, e1)：冷专家归入极大连续段 (a, b)；钉住专家自成单元素段
        (e, e+1)（pack 进钉住暂存）。无钉住时返回单一 (e0, e1)，与既有行为完全一致。
    """
    if not layer_pins:
        return [(e0, e1)]
    runs: list[tuple[int, int]] = []
    a = None
    for e in range(e0, e1):
        if e in layer_pins:
            if a is not None:
                runs.append((a, e))
                a = None
            runs.append((e, e + 1))
        elif a is None:
            a = e
    if a is not None:
        runs.append((a, e1))
    return runs


def _run_views(a: int, b: int, e0: int, piece, banks, layer_id: int, pin_list: list[int], pin_stage_for) -> tuple[dict, dict]:
    """
    Business Logic（为什么需要这个函数）:
        一段专家连续段的 pack 目标（bank 冷行区间或钉住暂存行）与源（piece 行区间）
        必须逐 role 对齐；pack 契约要求目标行连续，所以段拆分（_split_cold_runs）
        与这里的目标视图必须配套。

    Code Logic（这个函数做什么）:
        钉住段（a 为钉住专家，单元素段）：返回钉住暂存的第 j 行（j = a 在 pin list
        中的序数——与 slot 顶部区的分配序号一致）。冷段 (a, b)：返回 ({role: bank 冷行
        区间 [cold_before : cold_before + 段长]}, {role: piece 对应行区间})，
        cold_before = a 之前冷专家个数（冷行号按 id 升序压缩，段内连续）。
    """
    if a in set(pin_list):
        stage = pin_stage_for(layer_id)
        # 段内只会有单个钉住专家（_split_cold_runs 的钉住段都是单元素）
        j = pin_list.index(a)
        return ({role: tensor[j : j + (b - a)] for role, tensor in stage.items()},
                {role: values[a - e0 : b - e0] for role, values in piece.items()})
    cold_before = a - sum(1 for e in pin_list if e < a)
    return ({role: per_layer[layer_id][cold_before : cold_before + (b - a)] for role, per_layer in banks.items()},
            {role: values[a - e0 : b - e0] for role, values in piece.items()})


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


def _method_expert_banks(model_path, model_config, method, device, dummy, parallel, workers, chunk, layer_sink=None, pin_sets=None, pin_sink=None) -> ExpertBanks:
    from freetoken.moe.expert_pieces import iter_expert_pieces

    num_layers = model_config.num_moe_layers
    if dummy:
        return build_expert_banks(method, num_layers, None, device=device, dummy=True, pin_sets=pin_sets, pin_sink=pin_sink)
    pieces = iter_expert_pieces(
        model_path, model_config, method.kind, parallel=parallel, workers=workers, chunk=chunk
    )
    return build_expert_banks(method, num_layers, pieces, device=device, layer_sink=layer_sink, pin_sets=pin_sets, pin_sink=pin_sink)


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
    pin_sets: dict[int, list[int]] | None = None,
    pin_sink=None,
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

    ``pin_sets`` / ``pin_sink``（显存钉住，v1）: 冷压缩构建参数，原样转发给
    ``build_expert_banks``（语义见其 docstring：钉住层 bank 冷压缩为 ``[E-K, *row]``，
    钉住权重经 per-layer 暂存移交给 ``pin_sink``，返回 ``ExpertBanks.cold_row``）。
    仅 method 慢路径支持：FTW 快路径遇到钉住时降级为慢路径并告警（否则 host 无法省下
    钉住字节）；GGUF q4_0 提供方不支持，直接报错。
    """
    from freetoken.checkpoint.ftw import is_ftw_checkpoint, load_ftw_banks

    if pin_sets and model_path and is_ftw_checkpoint(model_path) and not dummy:
        logger.warning_rank0(
            "--hot-expert-list: FTW 快路径暂不支持冷压缩钉住，改走慢路径逐 piece 打包 "
            "（加载更慢；host 省下钉住字节的目标不受影响）"
        )
    elif model_path and is_ftw_checkpoint(model_path) and not dummy:
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
            return _method_expert_banks(model_path, model_config, method, device, dummy, par, workers, chunk, layer_sink, pin_sets, pin_sink)
        if pin_sets:
            raise ValueError(
                "--hot-expert-list 需要走 quant method 的慢路径打包（当前 checkpoint 的专家"
                "加载器不支持冷压缩钉住；GGUF q4_0 暂不兼容钉住）"
            )
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
