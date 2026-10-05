"""Pre-load GPU memory forecast for one serve configuration.

Prices what Engine.__init__ will put on the device before any weight is read: the resident
weights come from the meta-device model the engine builds anyway (its parameters are exactly
the tensors the loader materializes), and every pool is sized by the engine's own functions
(``_startup_kv_budget``, the pool family's ``kv_cost`` / ``plan_num_pages``, ``state_pool_bytes``,
``plan_moe_cache_auto``). Only what the engine does not price up
front -- load overhead, prefill activations, CUDA graphs -- is a heuristic here, and the verdict
never refuses on a heuristic alone.
"""

from __future__ import annotations

import copy
import itertools
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

GiB = 1 << 30
MiB = 1 << 20

# CUDA context + distributed setup held before the engine's first free-memory reading: nvidia-smi
# showed 22.79 GiB free where the engine logged 22.39 GiB (RTX 4090, CUDA 13.1).
CUDA_CONTEXT_BYTES = 400 * MiB
# Device memory the load takes beyond the parameters (kernel modules, library handles, finalize
# scratch): measured weights minus meta parameters was 0.31 GiB for Qwen3.8-27B-NVFP4 on an RTX 4090.
LOAD_OVERHEAD_BYTES = 320 * MiB
# Free memory "fits" keeps at the forecast prefill peak; less is reported as tight.
PEAK_MARGIN_BYTES = 256 * MiB
# Below this context (or the model's own limit) a configuration is reported as tight, not fits.
USEFUL_CONTEXT = 8192
# fla's chunked GDN prefill keeps one [K, V] state per value head and 64-token chunk.
_GDN_CHUNK = 64

VERDICTS = ("does not fit", "tight", "fits")

CATEGORY_LABELS = {
    "attention": "attention",
    "linear_attention": "linear attention (GDN)",
    "mlp": "MLP / shared experts / routers",
    "experts": "MoE routed experts",
    "embeddings": "embeddings / lm_head",
    "vision": "vision encoder",
    "mtp": "MTP / draft heads",
    "other": "norms / other",
    "rope": "rope tables",
}

_LANGUAGE_ROOTS = ("model.language_model.", "language_model.model.", "language_model.")


def tensor_category(name: str) -> str:
    """Bucket a checkpoint or model tensor name for the report."""
    from freetoken.models.config import VISION_KEY_PREFIXES

    n = name
    for root in _LANGUAGE_ROOTS:
        if n.startswith(root):
            n = n[len(root):]
            break
    if n.startswith("model.") and n[len("model."):].startswith(VISION_KEY_PREFIXES):
        n = n[len("model."):]
    if n.startswith(VISION_KEY_PREFIXES + ("vision_model.", "multi_modal_projector.", "mm_projector.")):
        return "vision"
    parts = n.split(".")
    if "mtp" in parts or any("nextn" in p for p in parts):
        return "mtp"
    keys = set(parts)
    if keys & {"embed_tokens", "lm_head", "wte", "embed"}:
        return "embeddings"
    if "experts" in keys:
        return "experts"
    if keys & {"linear_attn", "mamba", "mixer"}:
        return "linear_attention"
    if keys & {"self_attn", "attn", "attention"}:
        return "attention"
    if keys & {"mlp", "feed_forward", "ffn", "shared_expert", "shared_experts", "router"}:
        return "mlp"
    return "other"


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _walk(obj: Any, path: str = ""):
    """(path, object) over the model's own object graph (BaseOP trees, lists, dicts)."""
    seen: set[int] = set()
    stack = [(path, obj)]
    while stack:
        p, o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        yield p, o
        if isinstance(o, torch.Tensor):
            continue
        if isinstance(o, (list, tuple)):
            stack.extend((p, x) for x in o)
        elif isinstance(o, dict):
            stack.extend((f"{p}.{k}", x) for k, x in o.items())
        elif type(o).__module__.startswith("freetoken") and hasattr(o, "__dict__"):
            stack.extend((f"{p}.{k}" if p else k, x) for k, x in vars(o).items())


def _eager_tensor_bytes(model) -> int:
    """Device tensors built eagerly during the meta build (rope cos/sin tables): resident too."""
    storages: dict[int, int] = {}
    for _, o in _walk(model):
        if isinstance(o, torch.Tensor) and o.device.type != "meta":
            storages[o.untyped_storage().data_ptr()] = o.untyped_storage().nbytes()
    return sum(storages.values())


_BLOCK_STACK = re.compile(r"^(?P<stack>.+?\.(?:blocks|layers))\.(?P<idx>\d+)\.")


def _streamed_encoder_bytes(model, state: dict[str, torch.Tensor]) -> tuple[int, int]:
    """(block bytes moved to pinned host banks, device staging bytes) under --mm-encoder-weights host:
    the vision block stacks stream through BlockWeightStreamer, the rest of the tower stays resident."""
    from freetoken.layers.base import OPList
    from freetoken.models.weight_stream import BlockWeightStreamer

    stacks: dict[str, int] = {}
    for name, t in state.items():
        if tensor_category(name) != "vision":
            continue
        m = _BLOCK_STACK.match(name)
        if m:
            stacks[m["stack"]] = stacks.get(m["stack"], 0) + _nbytes(t)
    streamed = staging = 0
    for stack, nbytes in stacks.items():
        obj = model
        try:
            for seg in stack.split("."):
                obj = getattr(obj, seg)
        except AttributeError:
            continue
        blocks = obj.op_list if isinstance(obj, OPList) else obj
        if not blocks:
            continue
        streamed += nbytes
        staging += BlockWeightStreamer.staging_bytes(blocks)
    return streamed, staging


@dataclass
class WeightReport:
    gpu: dict[str, int]
    host: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # device bytes of the vision tower under --mm-encoder-weights host (None: nothing streams)
    vision_gpu_if_host: int | None = None

    @property
    def gpu_total(self) -> int:
        return sum(self.gpu.values())


# DF11 (lossless entropy-coded bf16) buffers are sized by the data at load and are empty in the meta
# build; GLM-4.5-Air's attention and embedding measured 10.7-11.2 bits per weight.
_DF11_BITS_PER_WEIGHT = 11.2


def _df11_estimate(model) -> Counter[str]:
    """Per category, the estimated bytes of DF11 weights whose buffers are still placeholders."""
    found: Counter[str] = Counter()
    for path, o in _walk(model):
        low8 = getattr(o, "low8", None) if hasattr(o, "_DF11_KEYS") else None
        if isinstance(low8, torch.Tensor) and low8.numel() == 0:
            found[tensor_category(f"{path}.weight")] += int(o.n * _DF11_BITS_PER_WEIGHT / 8)
    return found


def resident_weights(model, config) -> WeightReport:
    """Per-category bytes the loaded model keeps on the GPU, from its meta-device build: the loader
    casts every checkpoint tensor into these parameters, so skipped tensors (a vision tower under
    --text-model-only, MTP heads the family does not serve, offloaded experts) are already absent
    and quantize/upcast-at-load is already priced."""
    state = model.state_dict()
    gpu: Counter[str] = Counter()
    for name, t in state.items():
        gpu[tensor_category(name)] += _nbytes(t)
    df11 = _df11_estimate(model)
    gpu.update(df11)
    report = WeightReport(gpu=dict(gpu))
    if df11:
        report.notes.append(f"DF11-compressed weights are estimated at {_DF11_BITS_PER_WEIGHT} bits per weight; "
                            "their real size depends on the data.")
    rope = _eager_tensor_bytes(model)
    if rope:
        report.gpu["rope"] = rope
    vision = report.gpu.get("vision", 0)
    if vision and config.active_encoders:
        streamed, staging = _streamed_encoder_bytes(model, state)
        if streamed:
            report.vision_gpu_if_host = vision - streamed + staging
            if config.mm.encoder_weights == "host":
                report.gpu["vision"] = report.vision_gpu_if_host
                report.host["vision"] = streamed
    return report


def quant_by_category(model) -> dict[str, list[str]]:
    """Per category, the storage formats the model's projections were built with ("nvfp4 via triton",
    "bfloat16") -- what the loader converts the checkpoint into."""
    found: dict[str, set[str]] = {}
    for path, o in _walk(model):
        if not isinstance(getattr(o, "in_features", None), int):
            continue
        method = getattr(o, "quant_method", None)
        weight = getattr(o, "weight", None)
        kind = getattr(getattr(method, "kind", None), "value", None)
        if kind not in (None, "none"):
            label = f"{kind} via {method.kernel.name}"
        elif isinstance(weight, torch.Tensor):
            label = str(weight.dtype).removeprefix("torch.")
        else:
            continue
        found.setdefault(tensor_category(f"{path}.weight"), set()).add(label)
    return {cat: sorted(v) for cat, v in found.items()}


def activation_bytes_per_token(model, config) -> int:
    """Heuristic transient bytes per prefill token: a few residual-stream copies plus the widest
    projection's input and output, the GDN chunk states, and the routed-expert intermediates."""
    from freetoken.kvcache.linear_state_pool import _linear_local_dims

    mc = config.model_config
    item = config.dtype.itemsize
    tp = config.tp_info.size
    skip = ("visual", "vision_tower", "embed_vision", "vision_embedder", "lm_head", "embed_tokens", "mtp")
    widest = 0
    for path, o in _walk(model):
        if path.split(".")[0] in skip:
            continue
        i, out = getattr(o, "in_features", None), getattr(o, "out_features", None)
        if isinstance(i, int) and isinstance(out, int):
            widest = max(widest, i + out)
    per_token = item * (4 * mc.hidden_size + widest)
    group = mc.linear_attention_group() if hasattr(mc, "linear_attention_group") else None
    if group is not None:
        _, _, v_heads = _linear_local_dims(group, tp)
        per_token += item * v_heads * group.key_head_dim * group.value_head_dim // _GDN_CHUNK
    if getattr(mc, "is_moe", False):
        per_token += item * mc.num_experts_per_tok * 3 * mc.moe_intermediate_size // tp
    return int(per_token)


def _is_offload(config) -> bool:
    from freetoken.moe import is_offload_moe_strategy

    return bool(getattr(config.model_config, "is_moe", False)) and is_offload_moe_strategy(config.moe_strategy)


@dataclass
class _Plan:
    ok: bool
    reason: str = ""
    moe_slots: int = 0
    num_pages: int = 0
    prefill_overlap: bool = True


def _solve(config, pool_cls, free_before: int, weights_bytes: int, per_expert: int,
           max_slots: int | None, state_bytes: int) -> _Plan:
    """The engine's startup solve (offload cache, then KV pages) against predicted numbers."""
    from freetoken.engine.engine import _startup_kv_budget, plan_moe_cache_auto

    slots, overlap, planned_pages = 0, config.moe_prefill_overlap, None
    if _is_offload(config):
        if config.moe_cache_auto:
            if per_expert <= 0:
                return _Plan(False, "the expert slot size is unknown, cannot plan --moe-cache-auto")
            try:
                slots, pages, overlap = plan_moe_cache_auto(
                    config, pool_cls, baseline_free=free_before, weights_bytes=weights_bytes,
                    per_expert_bytes=per_expert, max_slots=max_slots,
                )
            except AssertionError as exc:
                return _Plan(False, f"--moe-cache-auto cannot split the budget ({exc})")
            if config.num_page_override is None:
                planned_pages = pages
        else:
            slots = config.moe_cache_size
    resident = weights_bytes + slots * per_expert
    available = _startup_kv_budget(config.memory_ratio, free_before, free_before - resident) - state_bytes
    if planned_pages is not None:
        pages = planned_pages
    else:
        try:
            pages = pool_cls.plan_num_pages(config, available)
        except ValueError as exc:
            return _Plan(False, str(exc), slots, 0, overlap)
    if pages <= 1:
        return _Plan(False, f"no room for the KV cache ({_gib(available)} left for it after weights "
                            f"and fixed pools)", slots, pages, overlap)
    if config.num_page_override is not None:
        cache_per_page, fixed, _, _ = pool_cls.kv_cost(config)
        need = pages * cache_per_page + fixed
        physical = free_before - resident - state_bytes
        if need > physical:
            return _Plan(False, f"--num-pages {pages} needs {_gib(need)} of KV cache but only "
                                f"{_gib(physical)} is free after the weights", slots, pages, overlap)
    return _Plan(True, "", slots, pages, overlap)


def _fixed_workspace_bytes(backend: str) -> int:
    """Fixed (non-paged) attention workspace the backend allocates after the pools, summed over
    comma-separated prefill,decode parts: FlashInfer's float workspace floor (256 MiB,
    attention/fi.py) and TRT-LLM's 128 MiB buffer (attention/trtllm.py). The engine does not
    charge it to the pool budget, so it comes out of the memory left after init."""
    sizes = {"fi": 256 * MiB, "trtllm": 128 * MiB}
    return sum(sizes.get(p.strip(), 0) for p in backend.split(","))


@dataclass
class Forecast:
    free_before: int | None
    memory_ratio: float
    weights_bytes: int
    load_overhead: int
    kv_bytes_per_token: int
    kv_fixed_bytes: int
    page_size: int
    state_slots: int
    state_bytes_per_slot: int
    per_expert_bytes: int
    max_running_req: int
    max_seq_len: int
    moe_slots: int = 0
    num_pages: int = 0
    kv_tokens: int = 0
    max_context: int = 0
    attention_workspace: int = 0
    page_table: int = 0
    cuda_graphs: int = 0
    graph_batch_sizes: list[int] = field(default_factory=list)
    prefill_chunk: int = 0
    prefill_activations: int = 0
    free_after_init: int | None = None
    free_at_peak: int | None = None
    verdict: str = "unknown"
    reasons: list[str] = field(default_factory=list)

    @property
    def state_bytes(self) -> int:
        return self.state_slots * self.state_bytes_per_slot

    @property
    def kv_bytes(self) -> int:
        cache_per_page = self.kv_bytes_per_token * self.page_size
        return self.num_pages * cache_per_page + (self.kv_fixed_bytes if self.num_pages else 0)

    @property
    def moe_slot_bytes(self) -> int:
        return self.moe_slots * self.per_expert_bytes

    @property
    def budget(self) -> int | None:
        return None if self.free_before is None else int(self.memory_ratio * self.free_before)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(state_bytes=self.state_bytes, kv_bytes=self.kv_bytes,
                   moe_slot_bytes=self.moe_slot_bytes, budget=self.budget)
        return out


@dataclass
class ForecastInputs:
    """Everything the forecast needs that does not change with the sizing flags."""

    config: Any
    weights: WeightReport
    free_before: int | None
    per_expert_bytes: int = 0
    max_slots: int | None = None
    activation_per_token: int = 0
    load_overhead: int = LOAD_OVERHEAD_BYTES

    def forecast(self, config=None, *, weight_delta: int = 0, per_expert: int | None = None) -> Forecast:
        return forecast_memory(
            self.config if config is None else config,
            self.weights.gpu_total + weight_delta,
            self.free_before,
            per_expert_bytes=self.per_expert_bytes if per_expert is None else per_expert,
            max_slots=self.max_slots,
            activation_per_token=self.activation_per_token,
            load_overhead=self.load_overhead,
        )


def forecast_memory(config, weights_bytes: int, free_before: int | None, *, per_expert_bytes: int = 0,
                    max_slots: int | None = None, activation_per_token: int = 0,
                    load_overhead: int = LOAD_OVERHEAD_BYTES) -> Forecast:
    """Forecast the startup geometry of a resolved ``config`` (after ``_adjust_config``) with
    ``weights_bytes`` resident and ``free_before`` device bytes free before the load.

    The verdict refuses ("does not fit") only when the engine's own solve fails with the
    parameters alone -- the startup assert it would hit after loading. The load overhead and the
    heuristic workspaces can only make it "tight"."""
    from freetoken.engine.engine import _page_table_width
    from freetoken.engine.graph import _determine_cuda_graph_bs
    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots, state_pool_bytes

    mc = config.model_config
    pool_cls = resolve_pool_class(mc)
    cache_per_page, kv_fixed, _, _ = pool_cls.kv_cost(config)
    has_state = mc.linear_attention_group() is not None
    state_slots = _linear_pool_num_slots(config) if has_state else 0
    state_bytes = state_pool_bytes(config)
    fc = Forecast(
        free_before=free_before,
        memory_ratio=config.memory_ratio,
        weights_bytes=weights_bytes,
        load_overhead=load_overhead,
        kv_bytes_per_token=cache_per_page // config.page_size,
        kv_fixed_bytes=kv_fixed,
        page_size=config.page_size,
        state_slots=state_slots,
        state_bytes_per_slot=state_bytes // state_slots if state_slots else 0,
        per_expert_bytes=per_expert_bytes if _is_offload(config) else 0,
        max_running_req=config.max_running_req,
        max_seq_len=config.max_seq_len,
        attention_workspace=_fixed_workspace_bytes(config.attention_backend),
    )
    if free_before is None:
        return fc

    solve = lambda w: _solve(config, pool_cls, free_before, w, fc.per_expert_bytes, max_slots, state_bytes)  # noqa: E731
    lower = solve(weights_bytes)
    if not lower.ok:
        fc.verdict = "does not fit"
        fc.reasons.append(lower.reason)
        fc.moe_slots = lower.moe_slots
        return fc
    plan = solve(weights_bytes + load_overhead)
    tight: list[str] = []
    if not plan.ok:
        tight.append(f"with the estimated {_gib(load_overhead)} load overhead: {plan.reason}")
        plan = lower
    fc.moe_slots, fc.num_pages = plan.moe_slots, plan.num_pages
    fc.kv_tokens = plan.num_pages * config.page_size
    fc.max_context = min(config.max_seq_len, fc.kv_tokens)
    fc.page_table = (config.max_running_req + 1) * _page_table_width(fc.max_context, config.page_size) * 4
    bs_list = _determine_cuda_graph_bs(config.cuda_graph_bs, config.cuda_graph_max_bs, free_before)
    fc.graph_batch_sizes = sorted(bs_list)
    if bs_list:
        width = _page_table_width(fc.max_context, config.page_size)
        fc.cuda_graphs = max(bs_list) * (mc.vocab_size * 4 + activation_per_token + width * 4) + len(bs_list) * 2 * MiB
    fc.prefill_chunk = min(config.max_forward_len, fc.max_context)
    fc.prefill_activations = fc.prefill_chunk * activation_per_token

    resident = weights_bytes + load_overhead + fc.moe_slot_bytes + fc.state_bytes + fc.kv_bytes
    fc.free_after_init = free_before - resident - fc.attention_workspace - fc.page_table
    fc.free_at_peak = fc.free_after_init - fc.cuda_graphs - fc.prefill_activations
    spare = fc.free_at_peak
    if spare < 0:
        tight.append(
            f"prefill workspace not covered: a {fc.prefill_chunk}-token prefill needs ~{_gib(fc.prefill_activations)} and the "
            f"CUDA graphs ~{_gib(fc.cuda_graphs)}, but only {_gib(fc.free_after_init)} is free after init"
        )
    elif spare < PEAK_MARGIN_BYTES:
        tight.append(f"only {_gib(spare)} spare at the prefill peak")
    if fc.max_context < min(USEFUL_CONTEXT, config.max_seq_len):
        tight.append(f"max context is only {fc.max_context} tokens")
    override = getattr(config, "max_seq_len_override", None)
    if override is not None and fc.kv_tokens < override:
        tight.append(f"--max-seq-len-override {override} is above the {fc.kv_tokens} KV tokens; "
                     f"requests are capped at {fc.max_context}")
    fc.reasons.extend(tight)
    fc.verdict = "tight" if tight else "fits"
    return fc


# ---------------------------------------------------------------------------------------------
# Tips: each candidate is one flag change, priced by re-running the forecast on a config copy.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    flag: str
    key: str  # one change per key in a combination
    cost: int  # what the user gives up; the suggested combination minimizes the sum
    apply: Callable[[Any], None] = lambda config: None
    weight_delta: int = 0
    per_expert: int | None = None
    note: str = ""


def _set(**values):
    def apply(config):
        for k, v in values.items():
            object.__setattr__(config, k, v)
    return apply


def _set_running_req(n: int):
    def apply(config):
        if config.cuda_graph_max_bs == config.max_running_req:
            object.__setattr__(config, "cuda_graph_max_bs", n)
        if config.cuda_graph_bs is not None:
            object.__setattr__(config, "cuda_graph_bs", [bs for bs in config.cuda_graph_bs if bs <= n] or [1])
        object.__setattr__(config, "max_running_req", n)
    return apply


def _tips_concurrency(inp: ForecastInputs) -> Iterable[Change]:
    mr = inp.config.max_running_req
    for n in sorted({1, mr // 2} - {0}):
        if n < mr:
            yield Change(f"--max-running-requests {n}", "max_running_req", 3 if n == 1 else 2, _set_running_req(n))


def _tips_memory_ratio(inp: ForecastInputs) -> Iterable[Change]:
    current = inp.config.memory_ratio
    for ratio, cost in ((0.95, 2), (0.97, 3)):
        if ratio > current + 1e-9:
            yield Change(f"--memory-ratio {ratio}", "memory_ratio", cost, _set(memory_ratio=ratio),
                         note="less headroom outside the budget")
    for ratio in [r for r in (0.95, 0.9, 0.85, 0.8) if r < current - 1e-9][:2]:
        yield Change(f"--memory-ratio {ratio}", "memory_ratio", 1, _set(memory_ratio=ratio),
                     note="smaller KV cache, more headroom")


def _tips_prefill(inp: ForecastInputs) -> Iterable[Change]:
    config = inp.config
    current = getattr(config, "max_extend_tokens", None)
    if current is None or getattr(config.model_config, "dsv4_args", None) is not None:
        return
    for length, cost in ((4096, 1), (2048, 2)):
        if length < current:
            yield Change(f"--max-prefill-length {length}", "max_extend_tokens", cost,
                         _set(max_extend_tokens=length), note="longer prompts prefill in more chunks")


def _tips_cache_type(inp: ForecastInputs) -> Iterable[Change]:
    if inp.config.cache_type == "hybrid_radix":
        yield Change("--cache-type naive", "cache_type", 6, _set(cache_type="naive"),
                     note="no cross-request prefix reuse")


def _tips_encoders(inp: ForecastInputs) -> Iterable[Change]:
    vision = inp.weights.gpu.get("vision", 0)
    if vision:
        yield Change("--text-model-only", "encoders", 4, weight_delta=-vision, note="no image input")
        host = inp.weights.vision_gpu_if_host
        if inp.config.mm.encoder_weights == "gpu" and host is not None and host < vision:
            yield Change("--mm-encoder-weights host", "encoder_weights", 1, weight_delta=host - vision,
                         note="vision blocks stream from pinned host RAM")


def _tips_moe(inp: ForecastInputs) -> Iterable[Change]:
    config = inp.config
    mc = config.model_config
    if not getattr(mc, "is_moe", False):
        return
    total = mc.num_moe_layers * mc.num_experts
    if config.moe_strategy == "fused":
        experts = inp.weights.gpu.get("experts", 0)
        if experts and total:
            yield Change("--moe-strategy offload", "moe_strategy", 3,
                         _set(moe_strategy="offload", moe_cache_auto=True, moe_cache_size=0),
                         weight_delta=-experts, per_expert=experts // total,
                         note="experts stream from pinned host RAM into a GPU slot cache")
        return
    if not _is_offload(config):
        return
    if not config.moe_cache_auto:
        yield Change("--moe-cache-auto", "moe_cache", 1, _set(moe_cache_auto=True))
        half = max(2 * mc.num_experts, config.moe_cache_size // 2)
        if half < config.moe_cache_size:
            yield Change(f"--moe-cache-size {half}", "moe_cache", 2, _set(moe_cache_size=half))
    else:
        tokens = max(2 * config.kv_reserve_tokens, 32768)
        yield Change(f"--kv-reserve-tokens {tokens}", "kv_reserve", 1, _set(kv_reserve_tokens=tokens),
                     note="fewer expert slots, more KV")
    if config.moe_strategy != "cpu":
        yield Change("--moe-strategy cpu", "moe_strategy", 5,
                     _set(moe_strategy="cpu", moe_cache_auto=False, moe_cache_size=2 * mc.num_experts,
                          moe_prefill_overlap=True),
                     note="experts decode on the CPU")


# A new sizing option (e.g. a KV-cache dtype flag) adds one generator here; its bytes-per-token
# effect reaches the forecast through the pool family's kv_cost, so nothing else changes.
TIP_CANDIDATES: list[Callable[[ForecastInputs], Iterable[Change]]] = [
    _tips_concurrency, _tips_memory_ratio, _tips_prefill, _tips_cache_type, _tips_encoders, _tips_moe,
]


@dataclass
class Tip:
    flags: list[str]
    forecast: Forecast
    effect: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"flags": self.flags, "verdict": self.forecast.verdict, "kv_tokens": self.forecast.kv_tokens,
                "max_context": self.forecast.max_context, "max_running_req": self.forecast.max_running_req,
                "effect": self.effect, "notes": self.notes}


def _evaluate(inp: ForecastInputs, changes: tuple[Change, ...]) -> Forecast:
    config = copy.copy(inp.config)
    delta, per_expert = 0, None
    for c in changes:
        c.apply(config)
        delta += c.weight_delta
        per_expert = c.per_expert if c.per_expert is not None else per_expert
    return inp.forecast(config, weight_delta=delta, per_expert=per_expert)


def _rank(fc: Forecast) -> tuple:
    return VERDICTS.index(fc.verdict) if fc.verdict in VERDICTS else -1, fc.kv_tokens


def _improves(base: Forecast, new: Forecast) -> bool:
    """A better verdict; at the same verdict, more KV for a fitting config, else more room where
    the config is short (prefill headroom, or a context below USEFUL_CONTEXT)."""
    if _rank(new)[0] != _rank(base)[0]:
        return _rank(new)[0] > _rank(base)[0]
    if base.verdict == "fits":
        return new.kv_tokens > base.kv_tokens
    if base.verdict != "tight":
        return False
    more_peak = (new.free_at_peak or 0) - (base.free_at_peak or 0) >= 64 * MiB
    short_context = base.max_context < min(USEFUL_CONTEXT, base.max_seq_len)
    return more_peak or (short_context and new.max_context > base.max_context)


def _effect(base: Forecast, new: Forecast, changes: tuple[Change, ...]) -> str:
    parts = []
    if new.verdict != base.verdict:
        parts.append(f"{base.verdict} -> {new.verdict}")
    if new.kv_tokens != base.kv_tokens:
        parts.append(f"KV {base.kv_tokens} -> {new.kv_tokens} tokens ({new.kv_tokens - base.kv_tokens:+d})")
    delta = sum(c.weight_delta for c in changes)
    if delta:
        parts.append(f"{'saves' if delta < 0 else 'adds'} {_gib(abs(delta))} of weights")
    if new.state_slots != base.state_slots:
        parts.append(f"GDN state {base.state_slots} -> {new.state_slots} slots "
                     f"({_gib(base.state_bytes - new.state_bytes)} freed)")
    if new.moe_slots != base.moe_slots:
        parts.append(f"expert slots {base.moe_slots} -> {new.moe_slots}")
    if new.prefill_activations != base.prefill_activations:
        parts.append(f"prefill workspace {_gib(base.prefill_activations)} -> {_gib(new.prefill_activations)}")
    if (new.free_at_peak is not None and base.free_at_peak is not None
            and abs(new.free_at_peak - base.free_at_peak) >= 64 * MiB and not parts[1:]):
        parts.append(f"free at prefill peak {_gib(base.free_at_peak)} -> {_gib(new.free_at_peak)}")
    return ", ".join(parts)


def suggest_tips(inp: ForecastInputs, base: Forecast, max_combo: int = 3) -> tuple[list[Tip], Tip | None]:
    """Single-flag tips that improve the forecast, and the cheapest combination (fewest given up)
    that reaches "fits" -- or the best reachable one when nothing fits."""
    if inp.free_before is None:
        return [], None
    changes = [c for gen in TIP_CANDIDATES for c in gen(inp)]
    tips = []
    for c in changes:
        fc = _evaluate(inp, (c,))
        if _improves(base, fc):
            tips.append((c, fc))
    tips.sort(key=lambda t: (-_rank(t[1])[0], t[0].cost, -t[1].kv_tokens))
    single = [Tip([c.flag], fc, _effect(base, fc, (c,)), [c.note] if c.note else []) for c, fc in tips]
    if base.verdict == "fits":
        return single, None
    best = None
    for k in range(1, max_combo + 1):
        for combo in itertools.combinations(changes, k):
            if len({c.key for c in combo}) < k:
                continue
            fc = _evaluate(inp, combo)
            cost = sum(c.cost for c in combo)
            key = (fc.verdict == "fits", -cost if fc.verdict == "fits" else 0, _rank(fc), -cost)
            if best is None or key > best[0]:
                best = (key, combo, fc)
    if best is None or _rank(best[2])[0] <= _rank(base)[0]:
        return single, None
    _, combo, fc = best
    return single, Tip([c.flag for c in combo], fc, _effect(base, fc, combo), [c.note for c in combo if c.note])


# ---------------------------------------------------------------------------------------------
# Shared entry points: the inputs from a meta-built model, and the engine's preflight.
# ---------------------------------------------------------------------------------------------


def forecast_inputs(config, model, free_before: int | None) -> ForecastInputs:
    """Inputs from the meta-device model of a resolved config, as Engine.__init__ builds it."""
    weights = resident_weights(model, config)
    per_expert, max_slots = 0, None
    if _is_offload(config):
        from freetoken.engine.engine import shared_offload_method
        from freetoken.moe.expert_banks import bank_bytes_per_expert

        method = shared_offload_method(model)
        per_expert = bank_bytes_per_expert(config.model_config, method) or 0
        max_slots = method.slot_limit() if method is not None else None
        mc = config.model_config
        if per_expert:
            weights.host["experts"] = per_expert * mc.num_moe_layers * mc.num_experts
        else:
            weights.notes.append("expert bank layout unknown for this format: the GPU slot cache is not priced")
    return ForecastInputs(
        config=config, weights=weights, free_before=free_before, per_expert_bytes=per_expert,
        max_slots=max_slots, activation_per_token=activation_bytes_per_token(model, config),
    )


class PreflightRefused(RuntimeError):
    """The forecast proves the configuration cannot start; raised before any weight is read."""


def run_preflight(config, model, free_before: int) -> Forecast | None:
    """Engine.__init__'s pre-load check: refuse a configuration whose startup solve cannot succeed,
    warn on a tight one. Any failure of the forecast itself only logs: it must never be stricter
    than the engine."""
    try:
        inp = forecast_inputs(config, model, free_before)
        fc = inp.forecast()
        tips, combo = suggest_tips(inp, fc) if fc.verdict != "fits" else ([], None)
    except Exception as exc:  # noqa: BLE001 -- the forecast is advisory
        logger.warning_rank0(f"Memory preflight skipped: {type(exc).__name__}: {exc}")
        return None
    if fc.verdict == "fits":
        logger.info_rank0(
            f"Memory preflight: fits -- {fc.kv_tokens} KV tokens, max context {fc.max_context}, "
            f"{_requests(fc.max_running_req)} (forecast; see `ft info`)"
        )
        return fc
    text = format_forecast(inp, fc, tips, combo)
    if fc.verdict == "does not fit":
        logger.critical_rank0(text)
        raise PreflightRefused(
            f"this configuration cannot fit the GPU ({'; '.join(fc.reasons)})"
            + (f"; try {' '.join(combo.flags)}" if combo else "")
            + ". Pass --skip-preflight to load anyway."
        )
    logger.warning_rank0(text)
    return fc


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------


def _gib(n: int | None) -> str:
    if n is None:
        return "?"
    return f"{n / GiB:.2f} GiB"


def _requests(n: int) -> str:
    return f"{n} concurrent request{'' if n == 1 else 's'}"


def _row(label: str, value: str, note: str = "", indent: int = 0) -> str:
    return f"  {label:<{44 + indent}} {value:>12}" + (f"   {note}" if note else "")


def format_forecast(inp: ForecastInputs, fc: Forecast, tips: list[Tip], combo: Tip | None,
                    checkpoint: dict[str, int] | None = None) -> str:
    config = inp.config
    w = inp.weights
    lines = []
    head = f"{'Weights on the GPU':<46}"
    lines.append(head + (f"{'checkpoint':>12} {'GPU':>12}" if checkpoint is not None else f"{'GPU':>12}"))
    cats = [c for c in CATEGORY_LABELS if w.gpu.get(c) or w.host.get(c) or (checkpoint or {}).get(c)]
    for cat in cats:
        g, h = w.gpu.get(cat, 0), w.host.get(cat, 0)
        note = ""
        if h:
            note = f"+ {_gib(h)} in host RAM"
        elif checkpoint is not None and checkpoint.get(cat) and not g:
            note = "not loaded"
            if cat == "vision" and not config.active_encoders:
                note = "skipped (--text-model-only / --mm-disable)"
        elif checkpoint is not None and checkpoint.get(cat) and g and abs(g - checkpoint[cat]) > 0.05 * checkpoint[cat]:
            note = "smaller than stored (tied, skipped or re-packed)" if g < checkpoint[cat] else "upcast at load"
        ck = f"{_gib(checkpoint.get(cat, 0)):>12} " if checkpoint is not None else ""
        lines.append(f"  {CATEGORY_LABELS[cat]:<44}{ck}{_gib(g):>12}" + (f"   {note}" if note else ""))
    pad = 13 if checkpoint is not None else 0
    lines.append(_row("load overhead (estimate)", _gib(fc.load_overhead), "kernel modules, library handles", pad))
    lines.append(_row("total", _gib(fc.weights_bytes + fc.load_overhead), "", pad))
    lines.append("")
    if fc.free_before is None:
        lines.append("GPU free memory unknown: pass --gpu-memory-gib or --gpu-free-gib for a verdict.")
        lines.append(_row("KV cache per token", f"{fc.kv_bytes_per_token} B"))
        if fc.state_slots:
            lines.append(_row(f"GDN state ({fc.state_slots} slots)", _gib(fc.state_bytes)))
        return "\n".join(lines)
    lines.append(f"Budget: memory_ratio {fc.memory_ratio:g} x {_gib(fc.free_before)} free = {_gib(fc.budget)}")
    lines.append(_row("weights + load overhead", _gib(fc.weights_bytes + fc.load_overhead)))
    if fc.moe_slots or fc.per_expert_bytes:
        lines.append(_row(f"expert slot cache ({fc.moe_slots} x {fc.per_expert_bytes / MiB:.1f} MiB)",
                          _gib(fc.moe_slot_bytes)))
    if fc.state_slots:
        lines.append(_row(f"GDN state pool ({fc.state_slots} slots x {fc.state_bytes_per_slot / MiB:.1f} MiB)",
                          _gib(fc.state_bytes)))
    kv_label = f"KV cache ({fc.kv_bytes_per_token} B/token x {fc.kv_tokens} tokens)"
    lines.append(_row(kv_label, _gib(fc.kv_bytes)))
    if fc.verdict != "does not fit":
        lines.append("Outside the budget")
        lines.append(_row(f"attention workspace ({config.attention_backend})", _gib(fc.attention_workspace)))
        lines.append(_row("page table", _gib(fc.page_table)))
        lines.append(_row("free after init (engine log line)", _gib(fc.free_after_init)))
        bs = ",".join(map(str, fc.graph_batch_sizes)) or "off"
        lines.append(_row(f"CUDA graphs (bs {bs}, estimate)", _gib(fc.cuda_graphs)))
        lines.append(_row(f"prefill activations ({fc.prefill_chunk} tokens, estimate)", _gib(fc.prefill_activations)))
        lines.append(_row("free at prefill peak", _gib(fc.free_at_peak)))
    lines.append("")
    verdict = fc.verdict.upper()
    if fc.verdict != "does not fit":
        verdict += (f": max context {fc.max_context} tokens, {_requests(fc.max_running_req)}"
                    f" ({fc.kv_tokens // max(1, fc.max_context)} at full context)")
    lines.append(f"Verdict: {verdict}")
    lines.extend(f"  - {r}" for r in fc.reasons)
    if tips or combo:
        lines.append("")
        lines.append("Tips")
        for tip in tips:
            note = f" [{'; '.join(tip.notes)}]" if tip.notes else ""
            lines.append(f"  {' '.join(tip.flags):<32} {tip.effect}{note}")
        if combo:
            fcc = combo.forecast
            lines.append(f"  Suggested: {' '.join(combo.flags)} -> {fcc.verdict}, {fcc.kv_tokens} KV tokens, "
                         f"max context {fcc.max_context}, {_requests(fcc.max_running_req)}")
    return "\n".join(lines)
