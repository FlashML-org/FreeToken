from __future__ import annotations

import gc
import math
import glob
import os
import time
from datetime import timedelta
from typing import Any, Dict, Iterable, NamedTuple, Tuple

import torch
from freetoken.attention import AttnType, attention_backend_info, create_attention_backend
from freetoken.core import Batch, Context, Req, set_global_ctx
from freetoken.distributed import (
    destroy_distributed,
    enable_pynccl_distributed,
    get_tp_info,
    set_tp_info,
)
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.layers import set_rope_device
from freetoken.utils.numa import numa_nodes
from freetoken.utils.numa import placement as numa_placement
from freetoken.utils.numa import resolve_placement
from freetoken.models import create_model, load_weight
from freetoken.moe import create_moe_backend, is_offload_moe_backend
from freetoken.moe.expert_banks import load_expert_banks
from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache
from freetoken.utils import align_ceil, init_logger, is_sm90_family, is_sm100_family, mem_GB, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory
from .sample import BatchSamplingArgs, Sampler
from freetoken.kvcache import create_kv_pool, resolve_pool_class
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.kvcache.cache_status import _supports_swa_ratio
from freetoken.kvcache.linear_state_pool import (
    _linear_pool_min_slots, _linear_pool_num_slots, state_pool_bytes,
)

logger = init_logger(__name__)


def _require_offload_cache_size(cache_size: int, num_experts: int) -> None:
    """The offload MoE cache needs at least one slot per expert per layer. A too-small size
    (e.g. a bare offload run with moe_cache_size unset and auto disabled) must fail loudly."""
    if cache_size < num_experts:
        raise ValueError(
            f"moe_cache_size={cache_size} is too small: need at least num_experts={num_experts} "
            f"slots. Pass --moe-cache-size/--moe-cache-rate, or use --moe-cache-auto "
            f"(the default for offload/hybrid backends when no cache-sizing flag is given; "
            f"--moe-backend cpu always sizes its own fixed two-layer buffer and ignores "
            f"cache-sizing flags)."
        )


def _flashinfer_available() -> bool:
    from freetoken.kernel.backend import is_flashinfer_installed

    return is_flashinfer_installed()


def _sgl_flash_attn_available() -> bool:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception as exc:
        detail = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
        logger.warning_rank0(
            "sgl_kernel.flash_attn is unavailable; auto attention backend falls back to fi "
            f"({type(exc).__name__}: {detail})"
        )
        return False
    return True


def _startup_kv_budget(memory_ratio: float, init_free_memory: int, new_free_memory: int) -> int:
    """Bytes available to the KV pool at startup: ratio-scaled pre-load free memory minus
    what the resident model consumed. Kept as a pure function so the composition with the
    pool families' ``solve_num_pages`` stays CPU-testable."""
    return int(memory_ratio * init_free_memory) - (init_free_memory - new_free_memory)


def _page_table_width(max_seq_len: int, page_size: int) -> int:
    """Column count for the page table. ``_write_page_table`` writes WHOLE trailing pages, so the
    highest column touched is ``align_ceil(max_seq_len, page_size) - 1`` -- which the 32-alignment
    alone does not cover once page_size > 32 (an unaligned --max-seq-len-override on DSV4's P=128
    or trtllm's forced 64 would index past the row)."""
    return align_ceil(align_ceil(max_seq_len, page_size), 32)


def _required_attn_types(model_config) -> frozenset[AttnType]:
    """Backend-driving attention types of this model, from the group-spec walk
    (single source shared with the pool factory and the KV cost model). getattr
    fallbacks: duck-typed test configs may not implement the spec walk; for those,
    dsv4_args marks DSV4 (the real config declares a DSV4 attention group)."""
    specs_fn = getattr(model_config, "kv_cache_group_specs", None)
    if specs_fn is None:
        if getattr(model_config, "dsv4_args", None) is not None:
            return frozenset({AttnType.DSV4})
        return frozenset({AttnType.FULL})
    types = frozenset(
        spec.attn_type for spec in specs_fn() if spec.attn_type.backend_driven
    )
    return types or frozenset({AttnType.FULL})


def _backend_parts_serve(name: str, required: frozenset[AttnType]) -> bool:
    return all(
        required <= attention_backend_info(part).supported_types
        for part in name.split(",")
    )


def _backend_requirements_met(name: str) -> bool:
    # flashinfer first across ALL parts: the sgl probe logs a "falls back to fi" warning,
    # which would mislead when the candidate is about to fail on flashinfer anyway.
    infos = [attention_backend_info(part) for part in name.split(",")]
    if any(i.requires_flashinfer for i in infos) and not _flashinfer_available():
        return False
    if any(i.requires_sgl_kernel for i in infos) and not _sgl_flash_attn_available():
        return False
    if any(i.requires_sm100 for i in infos) and not is_sm100_family():
        return False
    return True


def _resolve_auto_attention_backend(
    required: frozenset[AttnType], hybrid_linear: bool
) -> str:
    """First candidate (in per-type priority order) whose arch condition holds,
    whose packages are installed, and whose every comma part serves ALL required
    types. Reproduces the historical hardware tree for FULL-only models:
    sm_100 -> trtllm, sm_90+sgl_kernel -> "fa,fi", flashinfer -> fi, else triton."""
    candidates: list[tuple[str, bool]] = []
    if AttnType.DSV4 in required:
        candidates.append(("dsv4_sparse", True))
    if required & {AttnType.MLA, AttnType.DSA}:
        candidates.append(("dsa", True))
    if AttnType.BSA in required:
        candidates.append(("m3_sparse", True))
    if AttnType.SWA in required:
        candidates.append(("triton", True))
    if AttnType.FULL in required:
        candidates += [
            ("trtllm", is_sm100_family()),
            ("fa,fi", is_sm90_family()),
            ("fi", True),
            ("triton", True),
        ]
    for name, arch_ok in candidates:
        if not arch_ok:
            continue
        if not _backend_parts_serve(name, required):
            continue
        if hybrid_linear and not all(
            attention_backend_info(p).hybrid_linear_ok for p in name.split(",")
        ):
            continue
        if not _backend_requirements_met(name):
            continue
        return name
    raise RuntimeError(
        "No attention backend can serve attention types "
        f"{sorted(t.value for t in required)} on this machine."
    )


def _validate_attention_backend_choice(config, override, required: frozenset[AttnType]) -> None:
    """Config-time type x backend capability check for the resolved (or explicit)
    backend string: every comma part must serve every required type and have its
    packages/arch available. Replaces the per-model gates; in particular this is
    where a DSV4 or MLA checkpoint rejects a generic backend before weights load,
    and where a generic model rejects dsa/dsv4_sparse."""
    from freetoken.attention import validate_attn_backend

    # Name membership first (ArgumentTypeError listing the supported names): the CLI already
    # ran this, but the programmatic EngineConfig path reaches here unvalidated and would
    # otherwise die on a bare KeyError from the info lookup below.
    validate_attn_backend(config.attention_backend, allow_auto=False)

    model_config = config.model_config
    backend_parts = [p.strip() for p in config.attention_backend.split(",")]
    for part in backend_parts:
        info = attention_backend_info(part)
        missing = required - info.supported_types
        if missing:
            valid = [
                name
                for name in ("fa", "fi", "trtllm", "triton", "dsa", "dsv4_sparse", "m3_sparse")
                if required <= attention_backend_info(name).supported_types
            ]
            missing_names = "/".join(sorted(t.value for t in missing))
            raise ValueError(
                f"{getattr(model_config, 'model_type', 'model')} uses {missing_names} "
                f"attention, which backend {part!r} does not support; valid backends: "
                f"{', '.join(valid)} (or auto), got {config.attention_backend!r}."
            )
        if getattr(model_config, "has_linear_attention", False) and not info.hybrid_linear_ok:
            raise ValueError(
                f"backend {part!r} does not support hybrid-linear (GDN/mamba) models, "
                f"got {config.attention_backend!r}."
            )
        if AttnType.SWA in required and not info.consumes_attn_spec:
            # SWA models drive window/sinks/sm_scale through the per-call AttentionSpec;
            # a backend that drops it would attend with the wrong window silently.
            raise ValueError(
                f"backend {part!r} does not consume the per-call AttentionSpec that "
                f"SWA models require, got {config.attention_backend!r}."
            )

    # An explicitly-selected backend may require a package that isn't installed. Auto
    # never resolves to one of these when its package is missing, so this only fires for
    # explicit --attention-backend choices.
    for part in backend_parts:
        info = attention_backend_info(part)
        if info.requires_flashinfer and not _flashinfer_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires flashinfer, which is "
                "not installed. Install it with `pip install 'freetoken[fi]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sgl_kernel and not _sgl_flash_attn_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires sgl_kernel, which is "
                "not installed. Install it with `pip install 'freetoken[sgl]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sm100 and not is_sm100_family():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires a compute capability "
                "10.x GPU: flashinfer's trtllm-gen kernels ship sm_100a/103a cubins only. "
                "Use --attention-backend fi (or triton) instead."
            )

    if required & {AttnType.MLA, AttnType.DSA} and config.page_size != 1:
        # The MLA backend's row addressing (latent scatter, DSA index keys, sparse
        # top-k page indices) assumes page_size == 1 throughout; reject explicitly
        # like the SWA models do rather than corrupting addressing silently.
        raise ValueError(
            f"latent-KV MLA models require --page-size 1, got {config.page_size}."
        )

    for part in backend_parts:
        info = attention_backend_info(part)
        if info.page_sizes is not None and config.page_size not in info.page_sizes:
            override("page_size", info.page_sizes[-1])
            logger.warning_rank0(
                f"Page size is overridden to {info.page_sizes[-1]} for the {part} backend"
            )


def _make_dummy_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    for key, param in model_state.items():
        if e8m0 is not None and param.dtype == e8m0:
            # e8m0 is a bare exponent code (value = 2^(code-127)), so 127 is scale 1.0.
            # It reports is_floating_point, but there is no normal_ kernel for it -- and a
            # RANDOM exponent would scale a block by up to 2^127 anyway, so 1.0 is both the
            # only fillable value and the only sane one. Without this a --dummy-weight run
            # of any block-scaled model dies with "normal_kernel_cuda not implemented".
            t = torch.empty(param.shape, dtype=param.dtype, device=device)
            t.view(torch.uint8).fill_(127)
            state_dict[key] = t
        elif param.dtype in fp8_dtypes:
            # torch.randn is not implemented for fp8; fill via a uint8 view with small
            # codes (avoid NaN/inf fp8 encodings). Lets dummy-weight startup work for
            # block-fp8 models (the dense fp8 linears are fp8 regardless of moe_backend).
            t = torch.empty(param.shape, dtype=param.dtype, device=device)
            t.view(torch.uint8).random_(0, 16)
            state_dict[key] = t
        elif param.dtype.is_floating_point or param.dtype.is_complex:
            state_dict[key] = torch.randn(param.shape, dtype=param.dtype, device=device)
        elif param.dtype == torch.uint8 and key.endswith("weight_scale_inv"):
            # MXFP8 e8m0 exponent codes: 127 encodes scale 1.0; zeros would collapse
            # every scale to 2^-127 and zero the model. Scoped BY NAME: other uint8
            # buffers are packed payloads whose bytes mean something else entirely
            # (GGUF qweight blocks embed fp16 scales -- 0x7F7F is fp16 NaN), so they
            # keep the benign all-zeros fill below.
            state_dict[key] = torch.full(param.shape, 127, dtype=param.dtype, device=device)
        else:
            state_dict[key] = torch.zeros(param.shape, dtype=param.dtype, device=device)
    return state_dict


def _materialize_loaded_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for key, weight in weights:
        expected = model_state.get(key)
        if expected is None:
            state_dict[key] = weight.to(device=device)
        else:
            state_dict[key] = weight.to(device=device, dtype=expected.dtype)
    return state_dict


# FREETOKEN_SPEC_DEBUG=1 dumps the first few blocks' token flow.
_SPEC_DEBUG = os.environ.get("FREETOKEN_SPEC_DEBUG", "0") == "1"
# Diagnostic only: synchronize and report the first speculative cycles by stage.
# Disabled by default because the per-cycle synchronization deliberately removes overlap.
_SPEC_TIMING = os.environ.get("FREETOKEN_SPEC_TIMING", "0") == "1"

class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


def _cpu_moe_max_tokens(config: EngineConfig) -> int:
    """Largest row count one CPU-expert task can receive.

    Ordinary decode has one row per request.  DSpark target verification has the
    anchor plus ``gamma`` proposal rows per request, while the draft itself has only
    ``gamma``; therefore ``1 + gamma`` is the required upper bound.
    """
    rows_per_req = 1
    dsv4 = getattr(config.model_config, "dsv4_args", None)
    if getattr(dsv4, "dspark_enabled", False):
        rows_per_req = 1 + max(
            1, int(getattr(dsv4, "dspark_block_size", 1) or 1)
        )
    max_reqs = max(config.max_running_req, config.cuda_graph_max_bs or 0, 1)
    return max_reqs * rows_per_req


def _bind_rank_to_numa_node(tp_info) -> None:
    """Pin this rank to one NUMA node BEFORE it allocates anything.

    The host expert banks are anonymous mmaps, so their pages land on whichever node
    first TOUCHES them -- the loader's threads. Unbound, those threads scatter across
    every socket, so a rank's banks end up interleaved and the CPU MoE pool, pinned to
    one socket's cores, reads much of its experts across the interconnect. That is the
    critical path: the hybrid backend computes most of each decode step's expert misses
    there.

    Binding first makes first-touch place a rank's banks on the node that will read
    them, which also puts every socket's memory controller to work instead of one.
    No-op on a single-node host.
    """
    # Resolve BEFORE the bind: binding narrows the affinity mask to one node, after
    # which the topology looks single-node and the answer would be lost.
    nodes = numa_nodes()
    placement = resolve_placement(tp_info.rank, tp_info.size)
    if placement is None:
        # Say WHY nothing happened, so "no NUMA lines" is never ambiguous between
        # "single socket" and "the binding silently did not run".
        logger.info_rank0(
            f"NUMA: not placing ranks ({len(nodes)} usable node(s), "
            f"tp_size={tp_info.size}) -- nothing to spread"
        )
        return
    node, cpus, siblings, index = placement
    try:
        os.sched_setaffinity(0, cpus)
    except (AttributeError, OSError) as exc:  # not Linux, or a restricted cpuset
        logger.warning(
            f"NUMA rank{tp_info.rank}: could not pin to node {node} ({exc}); "
            "expert banks may land on a remote node and decode will be slower"
        )
        return
    # EVERY rank reports, not just rank 0. A placement bug shows up as ranks disagreeing
    # with each other -- and that is invisible if only one of them speaks. This is
    # exactly how a real bug hid here: the bind said "2 ranks on this node" while the
    # thread split said 4, and only rank 0 was logging.
    logger.info(
        f"NUMA rank{tp_info.rank}/{tp_info.size}: node {node} of {len(nodes)} | "
        f"cpus {cpus[0]}..{cpus[-1]} ({len(cpus)}) | "
        f"{siblings} rank(s) here, I am #{index} | banks first-touch local"
    )


def _share_cpu_threads_across_ranks(tp_size: int) -> None:
    """Split the machine's CPU threads across the TP ranks, before any tensor work.

    torch sizes its intra-op pool from the WHOLE machine, and every rank does the same,
    so N ranks each start a machine-sized pool and oversubscribe every core N times.
    Measured on a 40-core / 80-thread host at TP=4: 117 threads per rank, 468 in total,
    ~6x the hardware, all of it contending through the weight load -- which is thread-
    bound (host-side copies into the expert banks), not disk-bound. The NVMe sat at
    187 MB/s while the CPU ran at 92% user.

    The engine already narrows threads once the pinned CPU MoE pool exists; this covers
    everything BEFORE that, the weight load included. Honour an explicit OMP_NUM_THREADS.
    """
    if tp_size <= 1 or os.environ.get("OMP_NUM_THREADS"):
        return
    # Count what this rank may actually run on, not what the machine has. After the NUMA
    # bind above, the affinity mask is one node -- dividing the whole machine by tp_size
    # would under-count every rank by the number of nodes.
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = os.cpu_count() or tp_size
    placed = numa_placement()
    siblings = placed[2] if placed is not None else tp_size
    per_rank = max(1, allowed // siblings)
    if per_rank < torch.get_num_threads():
        torch.set_num_threads(per_rank)
        from freetoken.distributed import get_tp_info

        logger.info(
            f"NUMA rank{get_tp_info().rank}/{tp_size}: torch intra-op threads "
            f"{torch.get_num_threads()} | {allowed} cpus visible | "
            f"{siblings} rank(s) share them"
        )


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _ensure_expandable_segments()  # before the first CUDA allocation below
        # Bind BEFORE any allocation: the expert banks land on the node that
        # first touches them, and that must be the node whose cores will read them.
        _bind_rank_to_numa_node(config.tp_info)
        _share_cpu_threads_across_ranks(config.tp_info.size)

        from freetoken.gpu_select import bind_assigned_gpu

        self.device = bind_assigned_gpu(config.tp_info.rank)
        _adjust_config(config)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.config = config  # retained for runtime cache rebuild (rebuild_runtime_cache)
        # KV pool family fixed at construction from the model config: its classmethods own the
        # page-token geometry and cost arithmetic the engine needs BEFORE the pool exists
        # (num_pages sizing, --moe-cache-auto); the instance owns rebuild/validation after.
        self._pool_cls = resolve_pool_class(config.model_config)
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        free_min, free_max = self._sync_get_memory()
        init_free_memory = free_max  # startup KV sizing keeps cross-rank MAX (unchanged)
        self._baseline_free = free_min  # rebuild baseline: cross-rank MIN, deterministic across ranks
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        post_weights_free = self._sync_get_memory()[0]
        self._weights_bytes = self._baseline_free - post_weights_free
        # What the parameters declare vs what the load actually cost. Every byte of the gap
        # is resident-but-unaccounted (staging buffers, per-layer scratch) and comes straight
        # out of the MoE-cache + KV budget, so surface it instead of leaving it to arithmetic
        # on the cache-sizing line.
        declared = sum(t.numel() * t.element_size() for t in self.model.state_dict().values())
        logger.info_rank0(
            f"Weights: {mem_GB(declared)} declared by parameters, "
            f"{mem_GB(self._weights_bytes)} measured on device "
            f"(torch allocator holds {mem_GB(torch.cuda.memory_allocated(self.device))} live, "
            f"{mem_GB(torch.cuda.memory_reserved(self.device))} reserved; "
            f"peak {mem_GB(torch.cuda.max_memory_reserved(self.device))}). "
            "A gap between 'reserved' and 'measured' is held OUTSIDE the allocator."
        )
        # Pool-budget baseline for the desktop cache sliders: free VRAM after the weights are
        # resident but before ANY runtime cache pool (MoE expert cache below, KV pages, GDN
        # state) is allocated. This is the stable "if all free VRAM went to one pool" budget —
        # unlike a query-time mem_get_info it doesn't drift with allocator caching, CUDA
        # graphs, or other processes. Cross-rank MIN, deterministic across ranks.
        self._post_weights_free = post_weights_free
        self.moe_offload_cache = None
        self.cpu_moe_executor = None
        # Speculative accounting: acceptance rate is the one number that says whether
        # speculation is paying for itself, and it cannot be inferred from tokens/s.
        self._spec_accepted = 0
        self._spec_debug_left = 6
        self._spec_drafted = 0
        self._spec_timing_left = 12
        if is_offload_moe_backend(config.moe_backend):
            self._init_offload_moe_cache(config)
        if hasattr(self.model, "prepare_for_runtime"):
            self.model.prepare_for_runtime()

        # ======================= KV cache initialization ========================
        new_free = self._sync_get_memory()[1]
        # The engine measures the budget and settles the sibling GDN state pool's bytes
        # off it; the KV pool family owns every geometry-specific formula behind the rest.
        available_memory = _startup_kv_budget(config.memory_ratio, init_free_memory, new_free)
        available_memory -= state_pool_bytes(config)
        self.num_pages = self._pool_cls.solve_num_pages(config, available_memory)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kv_pool(
            config, self.num_pages, device=self.device, dtype=self.dtype
        )

        # ======================= Linear (GatedDeltaNet) state initialization ========================
        linear_group = config.model_config.linear_attention_group()
        if linear_group is not None:
            from freetoken.kvcache.linear_state_pool import LinearStatePool

            self.linear_state_pool = LinearStatePool(
                group=linear_group,
                num_slots=_linear_pool_num_slots(config),
                dtype=self.dtype,
                device=self.device,
                tp_size=config.tp_info.size,
            )
            self.ctx.linear_state_pool = self.linear_state_pool
        else:
            self.linear_state_pool = None

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        # Pools routed by the shared table but deriving reads through their own mappings (DSV4)
        # re-point here (and again on any table realloc). The graph-input snapshot that reads
        # through them belongs to the attention backend, built later in init_capture_graph.
        self.kv_cache.attach_page_table(self.page_table)

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # padded/dummy rows index the GDN padding slot (0) so gather/scatter hits scratch.
        if self.linear_state_pool is not None:
            self.dummy_req.linear_slot_idx = self.linear_state_pool.padding_slot
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
        )
        self._init_dspark_adaptive_verification()
        if config.attention_backend.split(",")[0] == "triton":
            # Prefill runs on the first comma part; warm its autotune cache.
            self._warmup_prefill()

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        model_state = self.model.state_dict()
        if config.use_dummy_weight:
            return _make_dummy_weight_state_dict(model_state, device=self.device)
        # _materialize casts each loaded tensor to its model-param dtype (model_state), so
        # models declaring per-tensor dtypes (e.g. DSV4's mixed fp8/fp32/bf16) are preserved;
        # offload models exclude experts (served from the offload cache, not dense weights).
        return _materialize_loaded_weight_state_dict(
            model_state,
            load_weight(
                config.model_path,
                self.device,
                include_moe_experts=not is_offload_moe_backend(config.moe_backend),
            ),
            device=self.device,
        )

    def _resolve_auto_moe_cache_size(self, config: EngineConfig, banks) -> tuple[int, int, bool]:
        """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

        Pure glue over the Phase-1 budget policy; isolated here so it is unit-testable
        without a GPU. Reused by the Phase-2 runtime rebuild.
        """
        from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto

        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)  # sibling GDN state pool, engine-summed
        num_experts = config.model_config.num_experts
        total_experts = config.model_config.num_moe_layers * num_experts
        return resolve_moe_cache_auto(
            baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes,
            memory_ratio=config.memory_ratio,
            cache_per_page=cache_per_page,
            fixed_cache_size=fixed_cache_size,
            per_expert_bytes=expert_bytes_per_slot(banks.sources),
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=config.moe_prefill_overlap,
            kv_reserve_tokens=max(config.kv_reserve_tokens, min_reserve),
            page_size=page_tokens,
            quant_format=banks.quant_format,
        )

    def _init_offload_moe_cache(self, config: EngineConfig) -> OffloadMoeCache:
        # A model may fully own cache construction via make_offload_moe_cache.
        # Otherwise load_expert_banks gives the model module a setup hook first, then
        # falls back to per-quant providers, and the engine wires the banks into cache.
        cache_factory = getattr(self.model, "make_offload_moe_cache", None)
        if cache_factory is not None and config.moe_cache_auto:
            raise ValueError(
                "--moe-cache-auto is not supported for models with a custom "
                "make_offload_moe_cache; pass --moe-cache-size explicitly."
            )
        # decode_target picks the bank layout + the per-decode mechanism:
        #   "hybrid" -> GPU-cache + CPU-overflow co-compute, every layer (--moe-backend hybrid);
        #   "cpu"    -> CPU executor for the cpu_layer_ids set (all layers under --moe-backend
        #               cpu, the --moe-cpu-layers subset under offload);
        #   "gpu"    -> plain GPU offload.
        # cpu/hybrid both read experts on the CPU, so banks load in the native (CPU-readable)
        # layout; the GPU slot-cache GEMM reads those same native rows. decode_target also
        # gates the CPU executor build below.
        n_moe = config.model_config.num_moe_layers
        cpu_layer_ids = _resolve_cpu_layers(config, n_moe)
        if (
            not cpu_layer_ids
            and config.moe_cpu_layers is None
            and config.moe_backend in ("offload", "hybrid")
            and _pin_budget_bytes() is not None
        ):
            cpu_layer_ids = _auto_cpu_layers(config, n_moe)
        # The drafter's MoE layers are the tail of the sequence (see the model's
        # _iter_offload_moe_layers). Keep them on the GPU whatever the CPU plan says: the
        # draft is a hard serial dependency of every block -- the verify cannot start
        # until it finishes -- so a draft layer on the CPU pool adds the engine's slowest
        # path to each block, and it is 3 layers against the target's 43.
        # Read the drafter's layer count from dsv4_args, NOT from extra_moe_layers.
        # extra_moe_layers is patched onto model_config later than this (parse_config
        # runs before the flag exists), so it is still 0 here even though
        # num_moe_layers already counts the drafter's layers -- which made this whole
        # path silently inert the first time it was written.
        dsv4 = getattr(config.model_config, "dsv4_args", None)
        n_draft = 0
        if getattr(dsv4, "dspark_enabled", False):
            n_draft = int(getattr(dsv4, "n_draft_layers", 0) or 0)
        if not n_draft:
            n_draft = int(getattr(config.model_config, "extra_moe_layers", 0) or 0)
        draft_layer_ids = frozenset(range(n_moe - n_draft, n_moe)) if n_draft else frozenset()
        if draft_layer_ids:
            # Always report it. The exclusion below is a no-op under --moe-backend
            # hybrid (nothing is assigned to the CPU up front), so its absence from the
            # log says nothing about whether the drafter's layers were identified at
            # all -- and if extra_moe_layers is still 0 here, this whole path is inert
            # and the drafter shares the capped fetch with the target.
            logger.info_rank0(
                f"dSpark: draft MoE layers {min(draft_layer_ids)}..{max(draft_layer_ids)} "
                f"of {n_moe} fetch uncapped on the GPU (serial with every block)"
            )
            if cpu_layer_ids & draft_layer_ids:
                cpu_layer_ids = cpu_layer_ids - draft_layer_ids
        elif getattr(dsv4, "dspark_enabled", False):
            logger.warning_rank0(
                "dSpark is on but no draft MoE layers were identified "
                f"(extra_moe_layers={n_draft}): the drafter's layers will share the "
                "target's capped fetch and can be computed on the CPU pool, in series "
                "with every block"
            )
        if config.moe_backend == "hybrid":
            decode_target = "hybrid"
        elif cpu_layer_ids:
            decode_target = "cpu"
        else:
            decode_target = "gpu"
        # split residency: where pinning is quota-capped (_pin_budget_bytes), pin only the GPU layers' banks and mlock the CPU layers'
        # uncapped hosts keep every bank pinned (CPU decode reads them the same; overlap prefill stays on)
        # not applied to plain --moe-backend cpu; all-locked under a cap = --moe-backend offload --moe-cpu-layers 1.0
        split_residency = (
            bool(cpu_layer_ids)
            and config.moe_backend in ("offload", "hybrid")
            and _pin_budget_bytes() is not None
        )
        if config.moe_backend == "cpu" and not split_residency:
            # cpu mode pins every bank for the prefill double buffer; over the pin cap that dies in cudaHostRegister, so lock everything instead
            from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

            budget = _pin_budget_bytes()
            bank_bytes = None
            if budget is not None:
                bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config)
            if bank_bytes and bank_bytes > budget:
                split_residency = True
                logger.info_rank0(
                    f"--moe-backend cpu: banks {bank_bytes / 2**30:.2f} GiB exceed the "
                    f"pin budget; OS-locking all layers instead of pinning"
                )
        if split_residency and config.moe_prefill_overlap:
            # locked (unregistered) layers cannot feed the async pinned H2D double buffer; their prefill is a synchronous pageable copy via materialize
            logger.info_rank0(
                "--moe-cpu-layers split residency: disabling MoE prefill overlap "
                "(locked layers prefill via synchronous pageable copies)"
            )
            object.__setattr__(config, "moe_prefill_overlap", False)
        if cache_factory is None:
            # Fast path: an FTW checkpoint loads its repacked banks directly.
            # Slow path: load_expert_banks auto-picks parallel vs serial baseline by
            # expert-tensor granularity. Both pin-after-fill.
            # --expert-load: serial/parallel force the read; auto (None) lets load_expert_banks
            # pick (parallel for scattered experts, with a low-RAM fallback to serial).
            expert_parallel = {"serial": False, "parallel": True}.get(config.expert_load, None)
            requested_residency = None
            if split_residency:
                from freetoken.moe.host_banks import HostResidency

                requested_residency = [
                    HostResidency.LOCKED.value if i in cpu_layer_ids
                    else HostResidency.PINNED.value
                    for i in range(config.model_config.num_moe_layers)
                ]
            banks = load_expert_banks(
                config.model_path,
                config.model_config,
                device=self.device,
                dtype=self.dtype,
                dummy=config.use_dummy_weight,
                parallel=expert_parallel,
                prefetch=config.expert_prefetch,
                decode_target=("cpu" if decode_target in ("cpu", "hybrid") else "gpu"),
                layer_residency=requested_residency,
            )
            if config.moe_cache_auto:
                size, pages, overlap = self._resolve_auto_moe_cache_size(config, banks)
                object.__setattr__(config, "moe_cache_size", size)
                object.__setattr__(config, "moe_prefill_overlap", overlap)
                if config.num_page_override is None:
                    # Honor the plan's KV half too: MoE slots and KV pages were solved
                    # against ONE budget (ratio x baseline - weights), so both must come
                    # from it. Re-solving pages later from a fresh free-memory reading
                    # double-counts everything allocated since the weights measurement
                    # (this expert cache, the CPU-executor GPU buffers, allocator
                    # slack) and goes negative whenever the expert fill is exact --
                    # a greedy fill leaves no headroom for the measurement delta.
                    object.__setattr__(config, "num_page_override", pages)
                logger.info_rank0(
                    f"--moe-cache-auto resolved moe_cache_size={size} "
                    f"num_pages={pages} (prefill_overlap={overlap})"
                )
            _require_offload_cache_size(config.moe_cache_size, config.model_config.num_experts)
            cache = OffloadMoeCache(
                # Models with leading dense layers (GLM-4) only have experts on the MoE
                # layers; num_moe_layers == num_layers when first_k_dense_replace == 0.
                num_layers=config.model_config.num_moe_layers,
                num_experts=config.model_config.num_experts,
                cache_size=config.moe_cache_size,
                device=self.device,
                cache_policy=config.moe_cache_policy,
                prefill_overlap=config.moe_prefill_overlap,
                prefill_hit_d2d=config.moe_prefill_hit_d2d,
                quant_format=banks.quant_format,
                decode_target=decode_target,
                hybrid_max_fetch=config.moe_hybrid_max_fetch,
            )
            # before set_bank_sources: the residency validation and the copy plan's skip of non-pinned layers key on the CPU-layer set
            cache.cpu_layer_ids = cpu_layer_ids
            cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency)
            cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        else:
            cache = cache_factory(config, self.device)
            cache.decode_target = decode_target
            cache.hybrid_max_fetch = config.moe_hybrid_max_fetch
            cache.cpu_layer_ids = cpu_layer_ids
        if decode_target == "hybrid":
            self._resolve_hybrid_fetch(config, cache)
        cache.cpu_layer_ids = cpu_layer_ids
        # Must be set before CUDA graph capture so the (device-side) accumulation ops are
        # captured and re-run on every decode replay.
        cache.collect_stats = config.moe_collect_stats
        # attach_offload_moe_cache walks for OffloadMoELayers, or defers to a model's
        # _iter_offload_moe_layers() hook when its MoE blocks are bespoke nn.Modules (DSV4).
        layers = attach_offload_moe_cache(self.model, cache)
        assert len(layers) == config.model_config.num_moe_layers
        if cache.decode_target in ("cpu", "hybrid"):
            self._init_cpu_moe_executor(config, cache, layers)
        self.ctx.moe_offload_cache = cache
        self.moe_offload_cache = cache
        return cache

    def _resolve_hybrid_fetch(self, config: EngineConfig, cache) -> None:
        """Resolve --moe-hybrid-max-fetch -1 (auto) into a bandwidth-matched fetch fraction.

        Perfect fetch/compute overlap wants fetched : cpu-computed misses = pcie_bw :
        (cpu_bw - pcie_bw), i.e. fetching a pcie_bw / cpu_bw fraction of each decode
        step's misses -- both sides then finish together instead of one idling. The
        achieved bandwidths come from the cached `ft bench bw` profile (the same one the
        auto backend pick reads); without a usable profile the old fixed cap of 1 applies.
        """
        if config.moe_hybrid_max_fetch >= 0:
            return  # explicit fixed cap
        from freetoken.moe.bench_profile import load_hybrid_fetch_fraction

        gpu_name, gpu_uuid = _profile_gpu(self.device.index)
        fraction = load_hybrid_fetch_fraction(
            cache.quant_format, gpu_name=gpu_name, gpu_uuid=gpu_uuid
        )
        if fraction is None:
            cache.hybrid_max_fetch = 1
            logger.warning_rank0(
                "--moe-hybrid-max-fetch auto: no usable `ft bench bw` profile for "
                f"{cache.quant_format!r} experts; using a fixed fetch cap of 1"
            )
            return
        cache.hybrid_max_fetch = cache.num_experts  # inert: the fraction is the cap
        cache.hybrid_fetch_fraction = fraction
        logger.info_rank0(
            f"--moe-hybrid-max-fetch auto: fetching {fraction:.1%} of each decode step's "
            "expert misses over PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU"
        )

    def _init_cpu_moe_executor(self, config: EngineConfig, cache, layers) -> None:
        """Build the persistent CPU MoE executor (decode-time expert compute).

        Must run before CUDA graph capture: the worker pool has to be live for the
        eager warmup forward, and the pinned IO buffers / host-func task pointers
        must be stable for the captured nodes. Buffers/tasks themselves are
        allocated lazily on the first (eager) forward at each batch size.
        """
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        sample = layers[0]
        required = ("top_k", "activation", "apply_router_weight_on_input")
        if not all(hasattr(sample, attr) for attr in required):
            raise NotImplementedError(
                "CPU MoE backend is not yet supported for this model architecture "
                f"(MoE layer {type(sample).__name__} is missing {required})."
            )
        # Decode batches never exceed max_running_req, but CUDA-graph padding can
        # round a batch up to the largest captured size; cover both.
        #
        # A dSpark target verify carries 1 + block_size ROWS per request: the anchor
        # plus every drafted token. max_tokens sizes the C++ pool's
        # per-task scratch, so a pool built for one row per request would be handed
        # more rows than it owns and overrun it.
        max_tokens = _cpu_moe_max_tokens(config)
        # gpt-oss mxfp4 carries clamped-swiglu scalars; other formats use the defaults.
        executor = CpuMoeExecutor(
            cache,
            top_k=sample.top_k,
            activation=sample.activation,
            apply_router_weight_on_input=sample.apply_router_weight_on_input,
            num_threads=config.moe_cpu_threads,
            max_tokens=max_tokens,
            device=self.device,
            swiglu_alpha=getattr(sample, "hidden_act_alpha", 1.702),
            swiglu_limit=getattr(sample, "swiglu_limit", None),
        )
        cache.set_cpu_executor(executor)
        self.cpu_moe_executor = executor

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def _target_moe_and_expert_bytes(self, moe_cache_size: int | None) -> tuple[int, int]:
        from freetoken.engine.cache_budget import expert_bytes_per_slot

        target_moe = (
            moe_cache_size
            if moe_cache_size is not None
            else (self.moe_offload_cache.cache_size if self.moe_offload_cache else 0)
        )
        per_expert_bytes = (
            expert_bytes_per_slot(self.moe_offload_cache.bank_sources)
            if self.moe_offload_cache is not None else 0
        )
        return target_moe, per_expert_bytes

    def _resize_kv_pool(self, config, num_pages: int, num_swa_pages: int | None) -> None:
        # IN-PLACE, identity-preserving: the CacheManager's swa_pool reference, ctx.kv_cache and
        # the model's per-access pool property all keep pointing at THIS pool, which frees its old
        # buffers before allocating the new ones. mark_for_rebind re-binds per-bind scratch on the
        # next forward (graph re-capture); the prefix tree + page bookkeeping reset is the
        # scheduler's generic cache_manager.rebuild.
        if self.kv_cache.needs_rebind_on_rebuild:
            self.model.mark_for_rebind()
        self.kv_cache.rebuild_from_config(config, num_pages, num_swa_pages=num_swa_pages)
        self.num_pages = num_pages

    def _refresh_seq_state(self, config) -> None:
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        if aligned_max_seq_len != self.page_table.shape[1]:
            # max_seq_len changed (e.g. KV grew past the startup token budget); the page table
            # columns must track it or new requests would index out of bounds. The scheduler
            # re-points its managers to engine.page_table on a num_pages rebuild.
            self.ctx.page_table = self.page_table = torch.zeros(
                (config.max_running_req + 1, aligned_max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        self.kv_cache.attach_page_table(self.page_table)

    @torch.inference_mode()
    def rebuild_runtime_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only in-place resize of the MoE slot cache, KV page pool, GDN (mamba) state pool,
        and/or the window pool (num_swa_pages: an absolute pinned window), followed by CUDA-graph
        re-capture. Does NOT reload weights or host expert banks. The caller (scheduler) must
        guarantee no in-flight prefill/decode.
        """
        config = self.config
        if (moe_cache_size is None and num_pages is None and num_mamba_slots is None
                and num_swa_pages is None):
            return

        # 0a. Geometry prevalidation BEFORE any destructive free. An invalid target (moe
        #     slots on a model with no offload cache, moe below num_experts / above the
        #     marlin cap, non-positive pages, or too few GDN slots to run) must reject
        #     recoverably with the old cache intact -- NOT after teardown, which would
        #     leave the server unable to serve. These checks are model-agnostic.
        if moe_cache_size is not None:
            if self.moe_offload_cache is None:
                raise CacheRebuildRejected(
                    "moe_cache_size requested but this model has no MoE offload cache"
                )
            try:
                self.moe_offload_cache.validate_rebuild(moe_cache_size)
            except ValueError as e:
                raise CacheRebuildRejected(str(e)) from e
        if num_pages is not None and num_pages <= 0:
            raise CacheRebuildRejected(f"num_pages must be positive, got {num_pages}")
        if num_mamba_slots is not None:
            if self.linear_state_pool is None:
                raise CacheRebuildRejected(
                    "num_mamba_slots requested but this model has no GDN state pool"
                )
            # num_mamba_slots is the USABLE slot count (what the user sets and the status bar
            # shows); the pool also reserves a padding sink (slot 0), so the physical pool is
            # num_mamba_slots + 1. _linear_pool_min_slots is the physical floor -> usable - 1.
            min_usable = _linear_pool_min_slots(config) - 1
            if num_mamba_slots < min_usable:
                raise CacheRebuildRejected(
                    f"num_mamba_slots {num_mamba_slots} is below the minimum {min_usable} "
                    f"(non-evictable working set for max_running_req={config.max_running_req}) "
                    f"needed to run; admission would deadlock"
                )
        if num_swa_pages is not None:
            # An absolute window pin for the radix-SWA window pool (Gemma) or the DSV4 window tier;
            # meaningless for dense/MHA models and the naive SWA path (concurrency x window).
            if not _supports_swa_ratio(config):
                raise CacheRebuildRejected(
                    "num_swa_pages requested but this model has no window pool "
                    "(needs DSV4 or a sliding-window model with --cache-type radix)"
                )
            if num_swa_pages <= 0:
                raise CacheRebuildRejected(
                    f"num_swa_pages must be positive, got {num_swa_pages}"
                )

        # 0b. Pool-family budget fit-check BEFORE any destructive free: an unfit geometry
        #     must reject (recoverable) so the old caches stay intact and serving continues,
        #     rather than freeing and then OOMing into permanent failure. The engine supplies
        #     the memory account; the pool answers whether its target geometry fits.
        target_moe, per_expert_bytes = self._target_moe_and_expert_bytes(moe_cache_size)
        # Price the sibling GDN state pool at ITS target (physical slots = usable + padding
        # sink) and hand the bytes in -- the KV pool only budgets its own tiers.
        target_mamba = (
            num_mamba_slots + 1
            if num_mamba_slots is not None
            else (self.linear_state_pool.num_slots if self.linear_state_pool is not None else None)
        )
        self.kv_cache.validate_rebuild(
            config, num_pages=num_pages,
            num_swa_pages=num_swa_pages, target_moe=target_moe,
            per_expert_bytes=per_expert_bytes, baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes, current_num_pages=self.num_pages,
            extra_fixed_bytes=(
                state_pool_bytes(config, target_mamba) if target_mamba is not None else 0
            ),
            extra_note=(
                f", mamba={target_mamba - 1} slots" if target_mamba is not None else ""
            ),
        )

        torch.cuda.synchronize(self.device)
        # Preserve the CUDA-graph batch-size set resolved at startup. The auto heuristic keys
        # off free memory, which is far smaller now that the caches are resident (post-cache
        # free << startup pre-load free), so re-deriving it here would silently drop large
        # batch sizes after the first rebuild. Reusing the already-resolved list keeps the
        # captured coverage identical (the fit-check above guarantees the graph headroom fits).
        prior_graph_bs = self.graph_runner.graph_bs_list
        # Point of no return for the scheduler's rollback logic: from here the live graphs and
        # pools start being freed. A failure BEFORE this flag flips leaves the engine serving
        # untouched (no rollback needed); after it, only a rebuild restores service.
        self.rebuild_teardown_started = True
        # 1. Tear down CUDA graphs + backend capture scratch (free-before-alloc).
        self.attn_backend.reset_capture()
        self.graph_runner.destroy_cuda_graphs()
        # 2. Resize caches in place (each frees its old GPU tensors before allocating).
        # Pin the new window first (validated above) so any KV-pool rebuild below sizes the window
        # to it (_dsv4_pool_sizes / _swa_paged_num_tokens read config.swa_num_pages_override).
        # frozen EngineConfig — mutate in place like the moe_cache_size path; `config.x = y` raises
        # FrozenInstanceError, which here aborts the rebuild after the CUDA graphs are gone (→ 503).
        if num_swa_pages is not None:
            object.__setattr__(config, "swa_num_pages_override", num_swa_pages)
        if moe_cache_size is not None:
            assert self.moe_offload_cache is not None, "no MoE offload cache to resize"
            self.moe_offload_cache.rebuild(moe_cache_size)
        if num_pages is not None:
            # sets self.num_pages (rebuilds KV + window)
            self._resize_kv_pool(config, num_pages, num_swa_pages)
        elif num_swa_pages is not None:
            # Window-only change: no page-count change, but re-derive the window pool at the new
            # pin against the CURRENT page count. This re-allocs the same-size full pool and
            # the resized window, both inside the pool's own rebuild_from_config.
            self._resize_kv_pool(config, self.num_pages, num_swa_pages)
        if num_mamba_slots is not None:
            # Reallocate the GDN state pool (frees old tensors first). Must sit between graph
            # teardown and re-capture so the recaptured graphs bind the new state tensors.
            # +1 for the reserved padding sink: num_mamba_slots is the usable count.
            self.linear_state_pool.rebuild(num_mamba_slots + 1)
        # 3. Refresh max_seq_len (+ generic page table) for the new token budget.
        self._refresh_seq_state(config)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        # 4. Re-capture CUDA graphs against the new tensors (reset_capture above re-armed
        #    the backend; _sync_get_memory empties the cache so freed memory is reclaimed).
        gc.collect()
        free_min = self._sync_get_memory()[0]
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=prior_graph_bs,  # reuse the startup-resolved set (see above)
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=free_min,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
        )
        self._init_dspark_adaptive_verification()

    def _init_dspark_adaptive_verification(self) -> None:
        """Install the paper's measured-cost scheduler when every prefix is captured."""
        self._adaptive_verification = None
        curve = list(getattr(self.graph_runner, "spec_verify_cost_curve", ()) or ())
        block_size = int(getattr(self.graph_runner, "spec_block_size", 0) or 0)
        if not curve or block_size < 1:
            return
        if self.config.max_running_req != 1:
            logger.warning_rank0(
                "DSpark adaptive verification needs the paper's marker-tensor varlen "
                "layout for request batches >1; using fixed gamma for this run"
            )
            return

        # Every TP rank must select the same graph shape.  vLLM broadcasts its
        # profiled cost curves from rank 0; do exactly that rather than letting
        # small per-GPU timing noise choose different collective graphs.
        tp = get_tp_info()
        if tp.size > 1:
            payload = [curve if tp.is_primary() else None]
            torch.distributed.broadcast_object_list(
                payload, src=0, group=self.tp_cpu_group
            )
            curve = payload[0]
        from freetoken.models.deepseek_v4.dspark import DSparkAdaptiveVerification

        self._adaptive_verification = DSparkAdaptiveVerification(
            block_size, curve, self.device
        )

    def adapt_speculative_batch(self, batch: Batch) -> None:
        """Compact one prepared DSpark block to the paper-selected prefix width.

        Allocation deliberately remains at gamma until acceptance, so abandoned
        page/window slots are still returned by ``release_speculative_tail``.  Only
        the target's input views and metadata shrink before graph replay.
        """
        manager = self._adaptive_verification
        if manager is None:
            return
        if len(batch.reqs) != 1 or batch.padded_size != 1:
            raise RuntimeError(
                "adaptive DSpark compaction currently requires one unpadded request"
            )
        confidence = batch.draft_confidence
        if confidence is None:
            raise RuntimeError("adaptive DSpark received no confidence probabilities")

        max_width = int(batch.spec_block)
        if max_width != manager.block_size:
            raise RuntimeError(
                f"DSpark prepared width {max_width}, expected checkpoint gamma "
                f"{manager.block_size}"
            )
        req = batch.reqs[0]
        width = manager.record_and_choose(confidence, req.uid)
        if width == max_width:
            return
        if not 0 <= width < max_width:
            raise RuntimeError(f"adaptive DSpark selected invalid width {width}")

        base = req.input_ids.numel() - max_width
        span = width + 1
        req.input_ids = req._ids_buf[: base + width]
        batch.input_ids = batch.input_ids[:span]
        batch.positions = batch.positions[:span]
        if batch.out_loc is not None:
            batch.out_loc = batch.out_loc[:span]
        if batch.draft_tokens is None or batch.draft_probs is None:
            raise RuntimeError("adaptive DSpark cannot compact a missing draft")
        batch.draft_tokens = batch.draft_tokens[:width]
        batch.draft_probs = batch.draft_probs[:width]
        batch.draft_confidence = confidence[:width]
        segments = getattr(batch.attn_metadata, "segments", None)
        if segments is None or len(segments) != 1:
            raise RuntimeError("adaptive DSpark needs one target metadata segment")
        _off, _old_n, table_idx, start_pos = segments[0]
        batch.attn_metadata.segments = [(0, span, table_idx, start_pos)]
        batch.spec_block = width

    def _record_adaptive_draft_cost(self, batch: Batch) -> None:
        """Publish rank-0's real drafter time to the shared five-sample profile."""
        manager = self._adaptive_verification
        if manager is None or not manager.needs_draft_profile:
            return
        start = getattr(batch, "_spec_draft_start", None)
        end = getattr(batch, "_spec_draft_end", None)
        if start is None or end is None:
            raise RuntimeError("adaptive DSpark draft timing events are missing")
        tp = get_tp_info()
        cost = start.elapsed_time(end) if tp.is_primary() else 0.0
        if tp.size > 1:
            shared = torch.tensor([cost], dtype=torch.float64)
            torch.distributed.broadcast(shared, src=0, group=self.tp_cpu_group)
            cost = float(shared.item())
        manager.record_draft_cost(cost)

    def _finish_speculative(
        self, batch: Batch, logits: torch.Tensor, args: BatchSamplingArgs, target_features
    ) -> ForwardOutput:
        """Keep the prefix of each block the target agrees with, and collapse the rest.

        The verify pass scored every drafted position, so ``logits[j]`` is the target's
        own prediction for the token after position j. Acceptance is a PREFIX: scan in
        order, stop at the first disagreement, and take the target's token there. That
        is what makes speculation invisible in the output -- the emitted sequence is
        exactly what the target alone would have produced.

        Each request keeps a different amount, so the state is fixed up per request:
        ``input_ids`` truncates to the accepted prefix, the bonus token is appended, and
        ``cached_len`` / ``device_len`` move to match the KV the verify actually wrote.
        """
        from freetoken.models.deepseek_v4.dspark import (
            accepted_prefix,
            rejection_accept_device,
            sampling_probs,
        )

        k = batch.spec_block
        # Acceptance reads one logits row per position of every block. Getting fewer
        # means the verify scored only each request's LAST token -- the default for a
        # prefill -- and the failure would otherwise surface as an opaque IndexError
        # inside the sampler, several frames from the cause.
        expected = (1 + k) * len(batch.reqs)
        if logits.shape[0] != expected:
            raise RuntimeError(
                f"speculative verify produced {logits.shape[0]} logits rows, expected "
                f"{expected} ({1 + k} per request x {len(batch.reqs)}). The forward must "
                "pass logit_indices covering every drafted position."
            )
        any_sampled = any(not req.sampling_params.is_greedy for req in batch.reqs)
        any_greedy = any(req.sampling_params.is_greedy for req in batch.reqs)
        # Not self.sampler: a speculative batch has 1+k rows per request while the
        # sampling args are sized per request. Greedy verification needs target argmax
        # ids. Probabilistic rejection consumes p directly; reducing and copying argmax
        # before its p/q test inserted an otherwise unused synchronization every cycle.
        target_cpu = (
            logits.argmax(dim=-1).to(torch.int32).to("cpu", non_blocking=False)
            if any_greedy
            else None
        )
        if any_sampled and batch.draft_probs is None:
            raise RuntimeError("sampled DSpark verification is missing draft probabilities")
        confidence = batch.draft_confidence
        proposed_gpu = batch.draft_tokens
        if proposed_gpu is None or proposed_gpu.numel() != k * len(batch.reqs):
            raise RuntimeError(
                "DSpark verify is missing the gamma proposals produced by its sequential stage"
            )
        # Greedy acceptance consumes proposal ids immediately. Sampled acceptance keeps
        # them on device until rejection has queued directly behind target verification.
        proposed_cpu = (
            proposed_gpu.to("cpu", non_blocking=False).to(torch.int32)
            if any_greedy
            else None
        )
        draft_cost_recorded = False
        if any_greedy:
            # The blocking id copies prove the draft events have completed.
            self._record_adaptive_draft_cost(batch)
            draft_cost_recorded = True

        emitted: list[torch.Tensor] = []
        release_tail = getattr(batch, "release_tail", None)
        selected_rows: list[int] = []
        accepted_counts: list[int] = []
        off = 0
        for i, req in enumerate(batch.reqs):
            sp = req.sampling_params
            greedy = sp.is_greedy
            span = 1 + k                       # this request's rows in the flat batch
            # Where the block begins IN THE BUFFER. Not device_len - k: device_len runs
            # one ahead of the buffer during decode (see the write-back above), so that
            # form silently yielded k-1 proposals -- a short slice, not an error.
            start = req.input_ids.numel() - k
            # Fixed-width verification is the paper/vLLM fallback when the profiled
            # hardware-aware scheduler is disabled.  A static per-token threshold is
            # deliberately not used: DSpark section 3.2 schedules a GLOBAL token budget
            # from cumulative survival probabilities and the measured draft/verify cost
            # curves.  Until that manager is ported, verify the full gamma without
            # claiming that the confidence head is itself a threshold rule.
            width = k
            # A block emits n_acc accepted tokens PLUS the bonus token, so it can carry
            # up to width+1 past `start`. Nothing upstream clamps that to the request's
            # output budget: a block that starts with 2 tokens left would write 6, run
            # off the end of _ids_buf, and leave device_len past max_device_len -- where
            # remain_len goes negative, can_decode never turns False, and the request
            # decodes forever instead of finishing on "length".
            budget = req.max_device_len - start - 1
            width = max(0, min(width, budget))
            if greedy:
                assert proposed_cpu is not None and target_cpu is not None
                proposed = proposed_cpu[i * k:(i + 1) * k]
                n_acc, bonus = accepted_prefix(
                    proposed[:width], target_cpu[off:off + width + 1]
                )
            else:
                # Sampled: accept with probability min(1, p(x)/q(x)) and resample from
                # the residual on rejection, so the emitted token stays exactly
                # p-distributed. An argmax comparison here would bias towards the mode.
                p_req = sampling_probs(
                    logits[off:off + width + 1],
                    sp.temperature,
                    sp.top_p,
                    sp.top_k,
                )
                assert batch.draft_probs is not None
                n_acc, bonus = rejection_accept_device(
                    proposed_gpu[i * k:i * k + width],
                    batch.draft_probs[i * k:i * k + width],
                    p_req,
                )
                if not draft_cost_recorded:
                    # The rejection sampler's one small D2H result proves the draft
                    # timing events have completed; do not add another synchronization.
                    self._record_adaptive_draft_cost(batch)
                    draft_cost_recorded = True
                proposed = proposed_gpu[i * k:(i + 1) * k].to(
                    "cpu", non_blocking=False
                ).to(torch.int32)
            # Snapshot before req._ids_buf is rewritten below. Without it, debug logs
            # display post-mutation tokens and invent proposal/target agreements.
            proposed_snapshot = proposed.clone() if _SPEC_DEBUG else None
            if proposed.numel() != k:
                raise RuntimeError(
                    f"block has {proposed.numel()} proposals, expected {k}; the "
                    "request's buffer and the batch's block width disagree"
                )
            keep = start + n_acc
            req.input_ids = req._ids_buf[:start]
            if n_acc:
                req.append_host(proposed[:n_acc].to(req.input_ids.dtype))
            req.append_host(torch.tensor([bonus], dtype=req.input_ids.dtype))
            # Hand back the pages and SWA slots of the positions this block did not
            # keep, BEFORE device_len drops past them. allocate_paged sized itself from
            # the full block width, and nothing else walks a range above the request's
            # current length -- so what is not released here is leaked until the next
            # idle integrity check fails, far from the cause.
            if release_tail is not None:
                release_tail(req, keep)
            req.cached_len, req.device_len = keep, keep + 1
            emitted.append(
                torch.cat([proposed[:n_acc], torch.tensor([bonus], dtype=torch.int32)])
            )
            if _SPEC_DEBUG and self._spec_debug_left > 0:
                self._spec_debug_left -= 1
                # One block's whole token flow, in the order acceptance sees it. Reading
                # this beats reasoning about the indices: the greedy output showed every
                # word duplicated ("are are", "first first"), which is an emission fault,
                # and only the actual tokens say where.
                logger.info_rank0(
                    "spec block: greedy=%s temp=%s topp=%s conf=%s "
                    "committed=%s proposed=%s target=%s width=%d n_acc=%d bonus=%s "
                    "emitted=%s",
                    greedy, sp.temperature, sp.top_p,
                    None if confidence is None else
                    [round(float(c), 3) for c in confidence[i * k:(i + 1) * k]],
                    int(req._ids_buf[start - 1]) if start > 0 else None,
                    proposed_snapshot[:width].tolist(),
                    None if target_cpu is None else
                    target_cpu[off:off + width + 1].tolist(),
                    width, n_acc, bonus,
                    emitted[-1].tolist() if emitted else None,
                )
            self._spec_accepted += n_acc
            self._spec_drafted += width
            selected_rows.append(off + n_acc)
            accepted_counts.append(n_acc)
            off += span

        # Select the target's saved compressor state after anchor + accepted prefix.
        # Later rejected rows may share its 128-token page and overwrite the live ring.
        self._restore_speculative_carry(batch, selected_rows)
        committed_features = self._trim_dspark_target_features(
            target_features, batch, accepted_counts
        )
        self._commit_dspark_target_features(committed_features)

        # The reply path reads one token per request; hand it the LAST emitted token and
        # let the scheduler read the rest off req.input_ids, which already holds them.
        last = torch.tensor([int(e[-1]) for e in emitted], dtype=torch.int32)
        batch.spec_emitted = emitted
        gpu = last.to(self.device, non_blocking=True)
        done = torch.cuda.Event()
        done.record(self.stream)
        return ForwardOutput(gpu, last, done)

    def _restore_speculative_carry(self, batch: Batch, selected_rows: list[int]) -> None:
        """Commit the per-token partial state selected by DSpark acceptance."""
        journal = batch.spec_carry_states
        if not journal:
            raise RuntimeError("a DSpark target verify produced no compressor partial states")
        device_rows = torch.tensor(selected_rows, dtype=torch.long, device=self.device)
        md = batch.attn_metadata
        table_rows = torch.tensor(
            [md.segments[i][2] for i in range(len(selected_rows))],
            dtype=torch.long,
            device=self.device,
        )
        positions = batch.positions.index_select(0, device_rows).long()
        slots = self.attn_backend.window_slots_at(table_rows, positions)
        for (layer_id, tier, ring_size), pieces in journal.items():
            if len(pieces) != batch.input_ids.numel():
                raise RuntimeError(
                    f"DSpark partial-state journal for layer {layer_id}/{tier} has "
                    f"{len(pieces)} rows, expected {batch.input_ids.numel()}"
                )
            states = torch.cat(pieces, dim=0).index_select(0, device_rows)
            self.attn_backend.write_carry_blocks(
                layer_id, tier, slots, ring_size, states
            )

    def _trim_dspark_target_features(self, features, batch: Batch, accepted: list[int]):
        """Keep target features through each request's accepted predecessor row."""
        if features is None:
            raise RuntimeError("a DSpark target verify returned no target features")
        from freetoken.models.deepseek_v4.model import DSparkTargetFeatures

        segments = batch.attn_metadata.segments
        if segments is None or len(segments) != len(accepted):
            raise RuntimeError(
                "DSpark feature trimming needs one target segment per accepted count"
            )
        indices = []
        for (off, _n, _ti, _start), n_acc in zip(segments, accepted, strict=True):
            indices.append(
                torch.arange(off, off + n_acc + 1, dtype=torch.long, device=self.device)
            )
        idx = torch.cat(indices)
        if features.hidden.shape[0] != batch.input_ids.numel():
            raise RuntimeError(
                f"DSpark target returned {features.hidden.shape[0]} feature rows for "
                f"{batch.input_ids.numel()} verify inputs"
            )
        return DSparkTargetFeatures(
            features.hidden.index_select(0, idx),
            features.positions.index_select(0, idx),
            features.table_rows.index_select(0, idx),
        )

    def _real_dspark_target_features(self, features, batch: Batch):
        """Drop CUDA-graph padding rows before context KV is committed."""
        if features is None or batch.is_prefill or features.hidden.shape[0] == batch.size:
            return features
        from freetoken.models.deepseek_v4.model import DSparkTargetFeatures

        if features.hidden.shape[0] < batch.size:
            raise RuntimeError(
                f"target produced {features.hidden.shape[0]} feature rows for "
                f"{batch.size} real decode requests"
            )
        idx = torch.arange(batch.size, dtype=torch.long, device=self.device)
        return DSparkTargetFeatures(
            features.hidden.index_select(0, idx),
            features.positions.index_select(0, idx),
            features.table_rows.index_select(0, idx),
        )

    def _commit_dspark_target_features(self, features) -> None:
        """Precompute draft context KV as soon as target rows become committed.

        vLLM performs this precompute immediately before its next proposal.  FreeToken
        cannot safely retain one model-global bundle across unrelated prefill/decode
        batches (or across several captured batch sizes), so it performs the same write
        once the target rows are known to be valid.  The next proposal observes the
        identical draft-layer KV without stale cross-request ownership.
        """
        if features is None:
            return
        catch_up = getattr(self.model, "catch_up_draft_context", None)
        if catch_up is not None:
            catch_up(features)

    def draft_into_batch(self, batch: Batch) -> torch.Tensor | None:
        """Fill a speculative batch's placeholder positions with the drafter's proposal.

        Called on a batch the scheduler has already prepared and extended: its segments
        cover ``1 + k`` positions per request, ``allocate_paged`` has given every layer
        -- target and draft alike -- slots at those positions, and the token ids past the
        first are the checkpoint's noise placeholder.

        The draft runs first and writes the DRAFT layers' KV; the verify forward that
        follows runs over the same positions and writes the TARGET layers' KV. The two
        never collide, because the draft layers' ids continue past the target's.

        Returns the confidence per position, or None when the model cannot draft yet.
        """
        drafter = getattr(self.model, "draft", None)
        if drafter is None:
            return None
        measure_draft = (
            self._adaptive_verification is not None
            and self._adaptive_verification.needs_draft_profile
        ) or (_SPEC_TIMING and self._spec_timing_left > 0)
        if measure_draft:
            batch._spec_wall_start = time.perf_counter()
            batch._spec_draft_start = torch.cuda.Event(enable_timing=True)
            batch._spec_draft_end = torch.cuda.Event(enable_timing=True)
            batch._spec_draft_start.record(self.stream)
        with self.ctx.forward_batch(batch):
            out = drafter(
                [req.sampling_params for req in batch.reqs],
            )
        if measure_draft:
            batch._spec_draft_end.record(self.stream)
        if out is None:
            raise RuntimeError("DSpark was scheduled without a loaded drafter")
        proposed, q, confidence = out
        k = batch.spec_block
        span = 1 + k
        if proposed.numel() != k * len(batch.reqs):
            raise RuntimeError(
                f"DSpark proposed {proposed.numel()} tokens, expected "
                f"{k} x {len(batch.reqs)}"
            )
        for i in range(len(batch.reqs)):
            batch.input_ids[i * span + 1:(i + 1) * span].copy_(
                proposed[i * k:(i + 1) * k].to(batch.input_ids.dtype)
            )
        batch.draft_tokens = proposed
        batch.draft_probs = q
        batch.spec_carry_states = {}
        return confidence

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        target_features = None
        timing = bool(
            batch.speculative and _SPEC_TIMING and self._spec_timing_left > 0
        )
        if timing:
            target_start = torch.cuda.Event(enable_timing=True)
            target_end = torch.cuda.Event(enable_timing=True)
            target_start.record(self.stream)
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_spec_cuda_graph(batch):
                logits = self.graph_runner.replay_spec(batch)
                target_features = self.graph_runner.dspark_spec_target_features(batch)
            elif self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
                target_features = self.graph_runner.dspark_target_features(batch)
            else:
                logits = self.model.forward()
                get_features = getattr(self.model, "dspark_target_features", None)
                target_features = get_features() if get_features is not None else None
        if timing:
            target_end.record(self.stream)
        if self.cpu_moe_executor is not None:
            # One pinned read: surfaces a fired flag-handshake watchdog (dead coordinator
            # -> stale expert outputs) as a loud error instead of silent corruption.
            self.cpu_moe_executor.raise_if_unhealthy()

        if batch.speculative:
            finish_wall_start = time.perf_counter()
            output = self._finish_speculative(batch, logits, args, target_features)
            if timing:
                cycle_end = torch.cuda.Event(enable_timing=True)
                cycle_end.record(self.stream)
                cycle_end.synchronize()
                draft_start = batch._spec_draft_start
                draft_end = batch._spec_draft_end
                logger.info_rank0(
                    "spec timing: draft_cuda=%.2fms target_cuda=%.2fms "
                    "post_cuda=%.2fms finish_wall=%.2fms cycle_wall=%.2fms",
                    draft_start.elapsed_time(draft_end),
                    target_start.elapsed_time(target_end),
                    target_end.elapsed_time(cycle_end),
                    (time.perf_counter() - finish_wall_start) * 1000,
                    (time.perf_counter() - batch._spec_wall_start) * 1000,
                )
                self._spec_timing_left -= 1
            return output

        self._commit_dspark_target_features(
            self._real_dspark_target_features(target_features, batch)
        )

        for req in batch.reqs:
            req.complete_one()

        batch_logits = logits[: batch.size]
        next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    @torch.inference_mode()
    def _warmup_prefill(self) -> None:
        """Compile the Triton prefill path before the first real request.

        Decode CUDA graph capture warms the decode path, but the first prefill
        can still pay Triton/cublas setup costs. Use the dummy request row and
        restore it afterwards so padded decode graph replay keeps using the
        dedicated dummy KV slot.
        """
        if self.max_seq_len < 2:
            return

        warmup_lens = [min(80, self.max_seq_len)]
        if self.max_seq_len >= 128:
            warmup_lens.append(128)
        warmup_lens = sorted({length for length in warmup_lens if length >= 2})
        if not warmup_lens:
            return

        dummy_row = self.page_table[self.dummy_req.table_idx]
        dummy_slot = int(dummy_row[0].item())
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        try:
            for length in warmup_lens:
                dummy_row[:length] = torch.arange(
                    length, dtype=torch.int32, device=self.device
                )
                warm_req = Req(
                    input_ids=torch.zeros(length, dtype=torch.int32, device="cpu"),
                    table_idx=self.dummy_req.table_idx,
                    cached_len=0,
                    output_len=1,
                    uid=-1,
                    sampling_params=None,  # type: ignore[arg-type]
                    cache_handle=None,  # type: ignore[arg-type]
                )
                batch = Batch(reqs=[warm_req], phase="prefill")
                batch.padded_reqs = batch.reqs
                batch.input_ids = torch.zeros(length, dtype=torch.int32, device=self.device)
                batch.positions = torch.arange(length, dtype=torch.int32, device=self.device)
                batch.out_loc = dummy_row[:length]
                self.attn_backend.prepare_metadata(batch)
                with self.ctx.forward_batch(batch):
                    self.model.forward()
        finally:
            dummy_row.fill_(dummy_slot)
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.reset()
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        logger.info_rank0(
            f"Prefill warmup complete for lengths {warmup_lens} "
            f"in {started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _profile_gpu(index: "int | None" = None) -> Tuple[str | None, str | None]:
    """(name, uuid) of visible device ``index`` (default: the current, i.e. bound, device); (None, None) without CUDA."""
    if not torch.cuda.is_available():
        return None, None
    ident = gpu_identity(torch.cuda.current_device() if index is None else index)
    return ident["name"], ident["uuid"]


def _ensure_expandable_segments() -> None:
    """Default the CUDA allocator to expandable segments.

    The motivating case is the offload prefill, which repeatedly dequantizes
    variable-sized NVFP4 expert blocks to BF16 (a different size per layer as the
    active-expert count varies). Under that alloc/free churn the default caching
    allocator fragments badly -- reserved memory can balloon far past the actual peak
    allocation (observed ~78GiB reserved for a <30GiB working set).
    ``expandable_segments`` lets freed regions of any size be reused, keeping
    reserved ~= allocated, so it is applied to every run, not just offload ones.

    Env vars are parsed once at import and ignored if set afterwards, so we apply the
    setting via the runtime API instead. Must run before the first CUDA allocation (the
    caller guarantees CUDA is not yet initialized). Any user-provided allocator config
    is respected and left untouched.
    """
    if os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception as exc:  # pragma: no cover - depends on torch build
        logger.info_rank0(f"Could not enable expandable_segments ({exc}); continuing")
        return
    logger.info_rank0("Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)")


def _resolve_cache_type(has_linear_attention: bool, requested: str) -> str:
    # Hybrid GDN models default to the HybridRadixCache (snapshots GDN state at chunk
    # boundaries -> cross-request prefix reuse). An explicit ``--cache-type naive`` opts out
    # to the old no-reuse path (debugging / parity baseline / lower GDN-state memory).
    if has_linear_attention:
        return "naive" if requested == "naive" else "hybrid_radix"
    return requested


def _adjust_dsv4_config(config: EngineConfig, override) -> None:
    """DSV4 engine-config reconciliation at config-resolution time (before the pool exists).
    Syncs the resolved runtime config into the opaque ``dsv4_args`` payload, sets
    page_size to the window page P, forces single-chunk prefill, and clamps cuda_graph_bs/max_bs to
    the DSV4 decode batch size.
    """
    if config.tp_info.size > 1:
        from freetoken.checkpoint.ftw import is_ftw_checkpoint

        if is_ftw_checkpoint(config.model_path):
            raise ValueError(
                "DeepSeek-V4 tensor parallelism currently requires the original "
                "safetensors checkpoint. FTW stores the TP=1 dense weights and expert "
                "banks, so it cannot be replayed into rank-local DSV4 shapes; use the "
                "raw checkpoint or run with --tp-size 1."
            )
    model_config = config.model_config
    model_config.dsv4_args.max_seq_len = config.max_seq_len
    model_config.dsv4_args.max_batch_size = config.max_running_req + 1  # +1 dummy
    # dSpark is opt-in: the drafter's routed experts enlarge both the host expert banks
    # and the GPU slot cache, so a run that will not speculate must not pay for them.
    if getattr(config, "speculative_dspark", False):
        if not model_config.dsv4_args.has_dspark:
            raise ValueError(
                "--speculative-dspark: this checkpoint ships no dSpark drafter "
                "(needs mtp.* weights with dspark_block_size > 1 in inference/config.json)"
            )
        from freetoken.models.deepseek_v4.args import set_dspark_enabled

        model_config.dsv4_args.dspark_enabled = True
        set_dspark_enabled(True)  # so the weight reader builds the same model
        # parse_config already ran, before the flag existed, so its extra_moe_layers is
        # still 0. The expert banks would then build n_layers + n_draft entries while
        # the offload cache was sized for n_layers, and the two assert against each
        # other AFTER the full expert load -- five minutes to learn it.
        # frozen dataclass -- same in-place idiom the offload-cache sizing uses below.
        object.__setattr__(
            model_config, "extra_moe_layers", model_config.dsv4_args.n_draft_layers
        )
        logger.info_rank0(
            f"dSpark drafter enabled: {model_config.dsv4_args.n_draft_layers} draft layers, "
            f"block size {model_config.dsv4_args.dspark_block_size}, "
            f"target layers {model_config.dsv4_args.dspark_target_layer_ids}"
        )
    # config.swa_full_tokens_ratio is the DSV4 window/full ratio directly (default sizing);
    # a runtime rebuild pins an absolute window via swa_num_pages_override instead.
    # DSV4's KV page IS the P-token window page (window == radix reuse granularity == lcm of
    # the compress ratios), so max_num_tokens = num_pages * page_size holds like every model.
    P = model_config.dsv4_args.window_size
    override("page_size", P)
    logger.info_rank0(f"DSV4 KV pages are {P}-token window pages; page_size set to {P}")
    # The generic CacheManager materializes DSV4 'radix' as the shared SWARadixCache (is_swa);
    # 'naive' stays naive with the pool's swa currency riding swa_paged.
    if getattr(config, "cache_type", "radix") != "naive":
        override("cache_type", "swa_radix")
    # 'radix' (SWARadixCache on the full-loc currency, carry-aware re-prefill) is the default and is
    # honored, as is an explicit 'naive'. Don't let max_extend_tokens force a second chunk within
    # one prompt (the pool's prefill_chunk_budget still chunks prompts larger than the window
    # pool); prefill batches ragged (bs>=1), each segment resuming from its own cached_len.
    if getattr(config, "max_extend_tokens", 0) < config.max_seq_len:
        override("max_extend_tokens", config.max_seq_len)

    # DSV4 decode batches at most max_running_req rows; its full-loc snapshot is sized to that,
    # so a graph bs above it would exceed the backend's captured snapshot rows. Clamp any
    # oversized explicit list / max_bs here (before GraphRunner ever sees it).
    mr = config.max_running_req
    if config.cuda_graph_max_bs is not None and config.cuda_graph_max_bs > mr:
        logger.warning_rank0(
            f"cuda_graph_max_bs {config.cuda_graph_max_bs} exceeds DSV4 max_running_req {mr}; "
            "clamping to max_running_req (larger decode batches never occur)."
        )
        override("cuda_graph_max_bs", mr)
    if config.cuda_graph_bs is not None:
        kept = [bs for bs in config.cuda_graph_bs if bs <= mr]
        if kept != list(config.cuda_graph_bs):
            dropped = [bs for bs in config.cuda_graph_bs if bs > mr]
            logger.warning_rank0(
                f"dropping cuda_graph_bs entries {dropped} above DSV4 max_running_req {mr} "
                "(larger decode batches never occur)."
            )
            override("cuda_graph_bs", kept)


def _parse_cpu_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
    """Parse ``--moe-cpu-layers``: an explicit MoE-layer id list (``"3,7,11"``), a count
    (``"8"`` -> 8 layers evenly strided across depth), or a fraction (``"0.5"``). Ids are
    indices into the MoE layers, ``[0, num_moe_layers)``."""
    s = spec.strip()
    if not s:
        return frozenset()
    if "," in s:
        ids = {int(x) for x in s.split(",") if x.strip()}
        for i in ids:
            if not 0 <= i < num_moe_layers:
                raise ValueError(
                    f"--moe-cpu-layers id {i} out of range [0, {num_moe_layers})"
                )
        return frozenset(ids)
    if "." in s:
        frac = float(s)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"--moe-cpu-layers fraction {frac} must be in [0, 1]")
        k = round(frac * num_moe_layers)
    else:
        k = int(s)
        if not 0 <= k <= num_moe_layers:
            raise ValueError(f"--moe-cpu-layers count {k} must be in [0, {num_moe_layers}]")
    # k layers spread evenly across depth (frozenset dedups any rounding collisions;
    # k == 0 yields an empty range, hence an empty set).
    return frozenset(round(i * num_moe_layers / k) for i in range(k))


def _resolve_cpu_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """MoE layer ids whose decode runs on the CPU executor.

    ``--moe-backend cpu`` -> every layer. ``--moe-backend offload`` + ``--moe-cpu-layers``
    -> the parsed subset (the rest stay on the GPU offload/PCIe path). Otherwise none.
    """
    if config.moe_backend == "cpu":
        return frozenset(range(num_moe_layers))
    spec = config.moe_cpu_layers
    if not spec or not is_offload_moe_backend(config.moe_backend):
        return frozenset()
    return _parse_cpu_layers_spec(spec, num_moe_layers)


# expert activations the CPU MoE executor supports (csrc ActKind)
_CPU_MOE_ACTS = (
    "silu", "swish", "gelu", "gelu_tanh", "gelu_pytorch_tanh", "swigluoai",
)


def _cpu_moe_executor_viable(model_config) -> bool:
    """Whether an automatic CPU-decode decision may target the CPU MoE executor.

    A default boot must degrade to GPU offload instead of crashing in CpuMoeExecutor after the whole load; explicit cpu/hybrid/--moe-cpu-layers picks still fail loudly."""
    from freetoken.moe.cpu_executor import _WFMT_IDS, compiled_extension_supports

    try:
        from freetoken.kernel import _cpu_moe  # noqa: F401
    except ImportError:
        return False
    act = getattr(model_config, "hidden_act", "silu")
    moe_wfmt = getattr(model_config, "moe_weight_format", None)
    if act not in _CPU_MOE_ACTS and moe_wfmt != "mxfp4":
        return False
    if moe_wfmt != "mxfp4" and not compiled_extension_supports(act):
        return False
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
    return fmt == "mxfp4" or fmt in _WFMT_IDS


def _pin_budget_bytes() -> int | None:
    """Bytes this process can safely cudaHostRegister, or None when the platform does not cap pinning (plain Linux).

    WSL's WDDM-backed CUDA caps pinning near half of RAM, shared across processes -- budget 40%. FREETOKEN_PIN_BUDGET_GB overrides anywhere."""
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        return int(float(env) * 2**30)
    if not hasattr(os, "uname") or "microsoft" not in os.uname().release.lower():  # WSL kernel tag
        return None
    return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.4)


def _auto_cpu_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """Pick CPU (locked) MoE layers automatically when the banks exceed the pin budget.

    Locks just enough head+tail layers: per-layer decode miss rates are U-shaped, so the ends are the cheapest to move off the slot cache."""
    from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

    bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config)
    if not bank_bytes:
        return frozenset()
    budget = _pin_budget_bytes()
    if budget is None or bank_bytes <= budget:
        return frozenset()
    if not _cpu_moe_executor_viable(config.model_config):
        logger.info_rank0(
            f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB exceed the "
            f"pin budget {budget / 2**30:.2f} GiB, but the CPU MoE executor cannot "
            f"serve this model; keeping every layer pinned on the GPU offload path"
        )
        return frozenset()
    n = min(num_moe_layers, math.ceil(num_moe_layers * (1 - budget / bank_bytes)))
    head = (n + 1) // 2
    ids = frozenset(range(head)) | frozenset(range(num_moe_layers - (n - head), num_moe_layers))
    logger.info_rank0(
        f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB > pin budget "
        f"{budget / 2**30:.2f} GiB; locking {n} head+tail MoE layers for CPU decode "
        f"({sorted(ids)})"
    )
    return ids


# MoE-only knobs and the value each resolves to on a dense model. moe_backend is handled
# separately (its dense value is 'fused', but 'auto' resolves there without a warning).
_DENSE_MOE_SETTINGS = {
    "moe_cache_size": 0,
    "moe_cache_rate": None,
    "moe_cache_auto": False,
    "moe_cpu_layers": None,
    "moe_cpu_threads": 0,
    "moe_hybrid_max_fetch": -1,
    "moe_prefill_overlap": True,
    "moe_prefill_hit_d2d": False,
    "expert_load": "auto",
    "expert_prefetch": 2,
}


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    model_config = config.model_config
    single_stream_only = getattr(model_config, "single_stream_only", False)
    is_dsv4 = getattr(model_config, "dsv4_args", None) is not None
    has_swa_attention = getattr(model_config, "has_swa_attention", False)
    has_linear_attention = getattr(model_config, "has_linear_attention", False)
    is_moe = getattr(model_config, "is_moe", False)
    expert_quant = getattr(model_config, "expert_quant", "none")

    if not is_moe:
        # A dense model has no routed experts: the MoE knobs are inert, and the offload family
        # is worse than inert -- engine init would build an expert cache for a model that has
        # none and abort startup (weights already resident) on an unrelated expert-source
        # error. Drop them at this one choke point, which the CLI and the programmatic
        # LLM(...) path both pass through. 'auto'/'fused' is the silent dense resolution;
        # anything else was asked for explicitly, so report what is being ignored.
        dropped = [
            f"{name}={getattr(config, name)!r}"
            for name, dense_value in _DENSE_MOE_SETTINGS.items()
            if getattr(config, name, dense_value) != dense_value
        ]
        if config.moe_backend not in ("auto", "fused"):
            dropped.insert(0, f"moe_backend={config.moe_backend!r}")
        override("moe_backend", "fused")
        for name, dense_value in _DENSE_MOE_SETTINGS.items():
            override(name, dense_value)
        if dropped:
            logger.warning_rank0(
                f"{getattr(model_config, 'model_type', 'model')} is a dense model (no routed "
                f"experts); ignoring MoE settings: {', '.join(dropped)}"
            )

    if single_stream_only:
        # The model runs one sequence at a time: it collapses the batch to one row and the
        # decode CUDA graph is captured at bs=1. Force the runtime knobs so the KV pool, page
        # table and graph capture all stay bs=1.
        if config.max_running_req != 1:
            override("max_running_req", 1)
        if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs >= 1:
            override("cuda_graph_bs", [1])
            override("cuda_graph_max_bs", 1)

    if config.cuda_graph_max_bs is None:
        override("cuda_graph_max_bs", config.max_running_req)

    if is_dsv4:
        _adjust_dsv4_config(config, override)

    if has_swa_attention:
        # Both SWA cache paths use the global-paged swa pool (page_size==1 only for now).
        if config.page_size != 1:
            raise ValueError(
                f"SWA models currently support only page_size=1, got {config.page_size}."
            )
        # naive keeps cache_type='naive' (NaivePrefixCache, no reuse) on the paged pool (==
        # sglang SWAChunkCache); radix materializes as swa_radix (SWARadixCache, cross-request
        # reuse == sglang SWARadixCache). Both allocate from the same swa pool + free out-of-window.
        if getattr(config, "cache_type", "radix") != "naive":
            if not 0.0 < config.swa_full_tokens_ratio <= 1.0:
                raise ValueError(
                    f"swa_full_tokens_ratio must be in (0, 1], got {config.swa_full_tokens_ratio}"
                )
            override("cache_type", "swa_radix")

    if has_linear_attention:
        override(
            "cache_type",
            _resolve_cache_type(True, getattr(config, "cache_type", "radix")),
        )

    # Type x backend capability matrix: resolve auto from the per-type priority
    # lists, then validate whatever is now selected (explicit or auto) -- every
    # comma part must serve every required type, with packages/arch available.
    required_attn_types = _required_attn_types(model_config)
    _dtype = getattr(config, "dtype", None)  # duck-typed test configs omit it
    if AttnType.BSA in required_attn_types and _dtype is not None and _dtype.itemsize != 2:
        # Reject at config time: the BSA pool's own assert only fires after the
        # model is resident (and not at all under `python -O`).
        raise ValueError(
            f"--dtype {config.dtype}: block-sparse attention serves 16-bit "
            "compute only (the index slab budgets 2 bytes/token); use bfloat16 "
            "or float16."
        )
    if _dtype == torch.float16 and "mxfp8" in (
        getattr(model_config, "attn_quant", "none"),
        getattr(model_config, "dense_quant", "none"),
    ):
        # The MXFP8 GEMV folds the pow2-descaled fp8 weight into the activation
        # dtype; fp16's narrow exponent can overflow/flush what bf16 represents
        # exactly, and the combination was never numerically validated.
        raise ValueError(
            "--dtype float16 with MXFP8 resident weights is unsupported (the "
            "W8A16 fold is only validated exact in bfloat16); use bfloat16."
        )
    if config.attention_backend == "auto":
        override(
            "attention_backend",
            _resolve_auto_attention_backend(required_attn_types, has_linear_attention),
        )
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")
    _validate_attention_backend_choice(config, override, required_attn_types)

    if config.moe_cache_rate is not None:
        total_experts = config.model_config.num_moe_layers * config.model_config.num_experts
        override("moe_cache_size", math.ceil(total_experts * config.moe_cache_rate))

    # The CPU MoE executor supports the silu/gelu family plus the clamped
    # swigluoai (csrc ActKind; "gpt_oss_swiglu" rides inside the mxfp4 kernel and
    # swigluoai the generic GEMV epilogue). A model with any other expert
    # activation cannot decode on the CPU: reject an explicit cpu/hybrid pick at
    # config time, and keep auto from upgrading offload -> hybrid off the profile.
    # hidden_act (the dense activation) stands proxy for the expert activation --
    # true for every in-tree model. mxfp4 experts pass regardless: their act runs
    # inside the mxfp4 kernel, not the generic epilogue.
    _cpu_moe_act_ok = getattr(model_config, "hidden_act", "silu") in _CPU_MOE_ACTS or (
        getattr(model_config, "moe_weight_format", None) == "mxfp4"
    )
    if (
        is_moe
        and not _cpu_moe_act_ok
        and (config.moe_backend in ("cpu", "hybrid") or config.moe_cpu_layers)
    ):
        asked = (
            f"--moe-cpu-layers={config.moe_cpu_layers!r}"
            if config.moe_backend not in ("cpu", "hybrid")
            else f"--moe-backend {config.moe_backend!r}"
        )
        raise ValueError(
            f"{asked}: the CPU MoE executor does not support this model's expert "
            f"activation {getattr(model_config, 'hidden_act', None)!r}; drop the flag "
            "and let every layer decode on the GPU offload path instead."
        )

    if is_moe and config.moe_backend == "auto":
        # A MoE model always defaults to the offload family: experts stream from pinned host
        # banks into an auto-sized GPU slot cache, which is the only default that serves a model
        # bigger than the GPU. The resident 'fused' path (bf16 / block-fp8 experts, the two
        # formats MoELayer can allocate) is still reachable, but only when asked for explicitly
        # -- auto never picks it, because nothing here knows whether the experts would fit in
        # HBM and a wrong guess is a weight-load OOM rather than a slower-but-working run.
        default_backend = "offload"
        # Hardware-adaptive config: a cached `ft bench bw` profile can upgrade
        # the offload default to hybrid when this machine's CPU MoE bandwidth clears its PCIe
        # gather bandwidth by the bench threshold (default 2x). hybrid is VRAM-equivalent to
        # offload -- same auto-sized GPU slot cache (_resolve_auto_moe_cache_size), plus a
        # host-RAM CPU executor -- so this never raises the OOM risk; with no profile (or one
        # from different hardware) it stays offload. offload remains the always-safe fallback.
        # Key the lookup on the real expert format: mxfp4/q4_0 live in moe_weight_format when
        # expert_quant is "none", and "none" with no weight format means plain bf16 experts.
        moe_wfmt = getattr(model_config, "moe_weight_format", None)
        bench_fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
        from freetoken.moe.bench_profile import load_backend_recommendation

        gpu_name, gpu_uuid = _profile_gpu()
        if load_backend_recommendation(bench_fmt, gpu_name=gpu_name, gpu_uuid=gpu_uuid) == "hybrid":
            from freetoken.moe.cpu_executor import compiled_extension_supports

            _act = getattr(model_config, "hidden_act", "silu")
            if not _cpu_moe_act_ok:
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the CPU MoE executor does not "
                    f"support this model's expert activation "
                    f"{getattr(model_config, 'hidden_act', None)!r}; staying on offload"
                )
            elif moe_wfmt != "mxfp4" and not compiled_extension_supports(_act):
                # Stale prebuilt _cpu_moe.so: an explicit cpu/hybrid pick still
                # hard-fails in the executor, but a default must not turn into a
                # post-load crash -- degrade to offload.
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the compiled _cpu_moe "
                    f"extension predates activation {_act!r} (rebuild with "
                    f"`python setup.py build_ext --inplace`); staying on offload"
                )
            else:
                default_backend = "hybrid"
                logger.info_rank0(
                    f"benchbw profile recommends hybrid for {bench_fmt!r} experts on this GPU"
                )
        override("moe_backend", default_backend)
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

        if (
            is_offload_moe_backend(config.moe_backend)
            and config.moe_cache_size <= 0
            and config.moe_cache_rate is None
            and not getattr(config, "moe_cache_auto", False)
        ):
            # args.py's "no sizing flag -> default --moe-cache-auto" only fires when the
            # backend is already offload-family at *parse* time. A bare `ft serve <FTW MoE
            # checkpoint>` (no --moe-backend, no cache flags) still has moe_backend=="auto" at
            # parse time -- the auto -> offload/cpu/hybrid resolution above is the first point
            # the concrete backend is known, so mirror the same default here: no sizing flag
            # was given, so let the scheduler resolve the cache size from free VRAM instead of
            # failing the _require_offload_cache_size guard with size=0.
            override("moe_cache_auto", True)
            logger.info_rank0(
                "No MoE cache sizing flag given; defaulting to --moe-cache-auto for "
                f"auto-selected backend {config.moe_backend!r}"
            )

    if is_moe and config.moe_backend == "fused":
        # An explicit 'fused' keeps the experts resident, so there is no slot cache to size. The
        # sizing flags no longer redirect the backend, so ignore them here and say so -- the
        # geometry the user asked for is what runs. Report the flag actually passed: --moe-cache-
        # rate was already folded into moe_cache_size above, and the three are mutually exclusive.
        if config.moe_cache_rate:
            inert = f"--moe-cache-rate={config.moe_cache_rate}"
        elif config.moe_cache_size:
            inert = f"--moe-cache-size={config.moe_cache_size}"
        elif getattr(config, "moe_cache_auto", False):
            inert = "--moe-cache-auto"
        else:
            inert = None
        if inert:
            logger.warning_rank0(
                f"MoE backend 'fused' keeps its experts resident; ignoring {inert} "
                "(use --moe-backend offload to serve experts from a slot cache)"
            )
            override("moe_cache_size", 0)
            override("moe_cache_rate", None)
            override("moe_cache_auto", False)

    if is_moe and config.moe_backend == "cpu":
        # CPU-compute decode keeps experts in host RAM and computes them on the CPU;
        # the GPU only holds the two-layer prefill double buffer. So the slot cache is
        # fixed at exactly two expert layers (prefill overlap requires >= 2*num_experts)
        # and --moe-cache-size / --moe-cache-auto / --moe-cache-rate do not apply.
        num_experts = config.model_config.num_experts
        if getattr(config, "moe_cache_auto", False):
            override("moe_cache_auto", False)
        override("moe_cache_size", 2 * num_experts)
        override("moe_prefill_overlap", True)
        logger.info_rank0(
            f"MoE backend 'cpu': decode computes experts on CPU; GPU keeps a "
            f"two-layer prefill buffer (moe_cache_size={2 * num_experts})"
        )

    if (
        is_moe
        and expert_quant not in ("none", "fp8_block")
        and not is_offload_moe_backend(config.moe_backend)
    ):
        raise ValueError(
            f"{expert_quant} experts require --moe-backend offload or cpu, "
            f"got {config.moe_backend!r}"
        )

    if is_moe and config.moe_cpu_layers and config.moe_backend not in ("offload", "hybrid"):
        # the layer split needs the offload host banks + slot cache; 'cpu' already runs every layer on CPU, 'fused' keeps experts resident on the GPU (no host banks)
        raise ValueError(
            "--moe-cpu-layers requires --moe-backend offload or hybrid (got "
            f"{config.moe_backend!r}); use --moe-backend cpu to run all layers on CPU"
        )

    if is_moe:
        object.__setattr__(model_config, "moe_backend", config.moe_backend)
    object.__setattr__(model_config, "nvfp4_backend", config.nvfp4_backend)

    # Must stay LAST: page_size is only final here (_adjust_dsv4_config sets P=128, the
    # TRTLLM block sets 64). Also covers the programmatic LLM(...) path that bypasses parse_args.
    if config.num_token_override is not None:
        if config.num_page_override is not None:
            raise ValueError("--num-tokens and --num-pages are mutually exclusive")
        if config.num_token_override % config.page_size != 0:
            raise ValueError(
                f"--num-tokens {config.num_token_override} is not a multiple of the resolved "
                f"page size {config.page_size}; nearest valid values: "
                f"{config.num_token_override // config.page_size * config.page_size} or "
                f"{(config.num_token_override // config.page_size + 1) * config.page_size}"
            )
        override("num_page_override", config.num_token_override // config.page_size)

    # The rope cos/sin table is baked to rotary_config.max_position, and neither rope kernel
    # bounds-checks the position it gathers with -- a longer ceiling reads past the table.
    # DSV4 is exempt: it sizes its own table from the resolved max_seq_len (_adjust_dsv4_config).
    rotary = getattr(model_config, "rotary_config", None)
    seq_override = getattr(config, "max_seq_len_override", None)
    if seq_override is not None and rotary is not None and not is_dsv4:
        if seq_override > rotary.max_position:
            raise ValueError(
                f"--max-seq-len-override {seq_override} exceeds the model's "
                f"rope table ({rotary.max_position} positions). Serving past it would read "
                "out of bounds; extend the checkpoint's rope_scaling / "
                "max_position_embeddings in config.json instead."
            )

    # The startup ServerArgs dump is the *requested* config, printed in the frontend process
    # before any of the resolution above ran -- so "moe_backend='auto'" is all it can say. This
    # is the one line that reports what actually runs, for every path (explicit backends never
    # hit an "Auto-selected ..." log at all).
    resolved = [
        f"attention_backend={config.attention_backend!r}",
        f"cache_type={getattr(config, 'cache_type', 'radix')!r}",
        f"page_size={config.page_size}",
    ]
    if is_moe:
        resolved.insert(0, f"moe_backend={config.moe_backend!r}")
    logger.info_rank0(f"Resolved config: {', '.join(resolved)}")
