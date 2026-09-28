from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat

if TYPE_CHECKING:
    from freetoken.moe.hot_pin import HotExpertRepinManager
    from freetoken.moe.hotness import ExpertHotness

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024

from freetoken.utils import init_logger

logger = init_logger(__name__)

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

# lives in kernel/aot_models.py: the AOT row table shares it and must stay importable in the torch-only kernel-cache build env, which cannot import freetoken.moe
from freetoken.kernel.aot_models import fp8_block_scale_pad


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992

# 未活跃容量槽的 usage 哨兵：钉住区按每层固定容量 K_cap 预留后，"缩 K"会让部分
# 容量槽失去映射（id_of_slot == -1）。flashlib 合并查询路径没有 pin_base 排除，
# 受害槽 = 全 cache argmin(usage)，陈旧低 usage 的空容量槽会被选中并把冷专家装进
# 钉住区（后续扩 K 会覆盖活槽）。哨兵取 2^62：远大于任何 step 计数、远小于
# flashlib 的 "usage == step 不可驱逐" 打包值 INT64_MAX，argmin 永不选它
# （LRU 地板保证恒有 usage < step 的真 LRU 槽存在）。hybrid 内核按
# off_c >= pin_base 整段排除容量区，不依赖该值；对它无害。
_PIN_USAGE_SENTINEL = 1 << 62


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = 0.0
    # bank layout from the expert kernel (a BankSpec per role); when given it replaces the _BANK_SCHEMAS lookup and the slot cap comes from max_slots
    layout: dict | None = None
    max_slots: int | None = None
    # Optional expert-routing hotness collector (--hot-stats-out); attached by the
    # engine before CUDA graph capture. The layers' forward calls its record() between
    # fused_topk and the slot-id rewrite. None = collection disabled (zero overhead).
    hotness: ExpertHotness | None = None
    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        if self.layout is None:
            assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-strategy cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        # 显存钉住状态（--hot-expert-list，经 engine 调 init_hot_pins 装配；None = 不钉住）。
        # 布局为"固定容量区 + 活跃计数动态"（运行中可用 --tune-file 的 pin_k 调每层
        # 活跃钉住数 K）：pin_base = cache_size - L * K_cap 静态不变，层 l 的容量槽 =
        # [pin_base + l*K_cap, pin_base + (l+1)*K_cap)，活跃钉住占用各层前 count_l 个槽。
        # pin_ids [L, K_cap] int32（GPU）：前 pin_counts[l] 个为活跃钉住专家 id，尾部
        # 填充值复首钉 id（合并查询全容量行拷贝时充无害命中，见 _ensure_experts_pinned）；
        # pin_counts[l] 每层活跃钉住数（0 = 该层不钉，如 CPU 解码层）；pin_floors[l]
        # 动态下界 = 加载期初始钉住数（宿主冷压缩 bank 行数 E - floors[l] 决定缩 K
        # 时换出行的供给上限）；pin_capacity 即 K_cap（None = 未钉住）。
        # cold_row [L, E] int32（GPU，pinned -> -1，冷专家 -> 冷行号，bank 冷压缩行映射）；
        # _pin_free_rows[l] 是缩 K 空出的 bank 行号池（install 换入时装的行来自
        # cold_row，unpin 换出时从池里取行回填宿主）。
        # pin_base（host int 镜像）= cache_size - L*K_cap，eager 路径/日志/CPU 镜像用；
        # pin_base_dev（int32 设备标量，cache 持有永不重分配）是 hybrid 内核的按指针
        # 读取边界（捕获冻结的是指针而非值）。先于 validate_rebuild 初始化（后者
        # 读取 pin_capacity 做 rebuild 地板校验）。
        self.pin_ids: torch.Tensor | None = None
        self.pin_counts: list[int] | None = None
        self.cold_row: torch.Tensor | None = None
        self.pin_base: int | None = None
        self.pin_base_dev: torch.Tensor | None = None
        self.pin_capacity: int | None = None
        self.pin_floors: list[int] | None = None
        self._pin_free_rows: list[set[int]] | None = None
        self.pin_slots: torch.Tensor | None = None  # [L, K_cap] 各层容量槽位（静态不变）
        # 合并查询缓冲：{(batch*num_topk) -> [n + K_max] int32}，ensure_experts 的
        # flashlib 路径把路由与钉住 id 拼进同一个 query。按需惰性分配（eager warmup
        # 时物化，形状随后固定，CUDA graph 捕获安全）。
        self._pin_query_buffers: dict[int, torch.Tensor] = {}
        self.validate_rebuild(self.cache_size)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # hybrid only: [2] int32 运行时可变 fetch 参数，[0]=max_fetch、[1]=frac_q16
        #（Q16 定点的 fetch_fraction）。hybrid ensure 内核按指针读取（捕获进 CUDA
        # graph 的是指针而非值），set_fetch_params 只改值 -> 值更新对已捕获 decode
        # 图立即生效（与动态重钉同款"值更新图安全"模式）。cache 持有、永不重分配
        # （与 pin_slots 同约束），rebuild 不动它（调参值跨 rebuild 存续）。非 hybrid
        # 模式为 None。
        self.fetch_params: torch.Tensor | None = None
        if self.decode_target == "hybrid":
            self.fetch_params = torch.zeros(2, dtype=torch.int32, device=self.device)
            self.set_fetch_params(self.hybrid_max_fetch, self.hybrid_fetch_fraction)
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        if self.layout is not None:
            self.bank_schema = tuple(role for role, spec in self.layout.items() if not spec.resident)
        else:
            self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst/feat are layer-invariant.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # Per-bank [2, num_experts, ...] double-buffer views over the slot cache's
        # first 2 * num_experts slots (set up when prefill_overlap is enabled).
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        # prefill 三源组装（钉住 + overlap）状态：双缓冲首行指针与组合填充描述符
        # （_init_prefill_overlap_buffers 建，CUDA only）、逐层 gather 索引与宿主
        # 冷行镜像（_build_pin_gather_buffers 建，init_hot_pins/rebuild 时刷新）。
        # 未钉住时全部保持空/None，prefill 行为与不钉住逐字节一致。
        self._prefill_buffer_ptrs: list[torch.Tensor] = []
        self._compose_cache_ptrs: torch.Tensor | None = None
        self._compose_host_ptrs: list[torch.Tensor] | None = None
        self._compose_feat_bytes: torch.Tensor | None = None
        self._overlap_small_bank_ids: list[int] = []
        self._pin_cold_dst: list[torch.Tensor] = []
        self._pin_cold_src: list[torch.Tensor] = []
        self._pin_cold_num: list[torch.Tensor] = []
        self._pin_gather_dst: list[torch.Tensor] = []
        self._pin_gather_src: list[torch.Tensor] = []
        self._pin_gather_num: list[torch.Tensor] = []
        self._cold_row_np = None
        # 动态重钉管理器（HotExpertRepinManager）：engine 在钉住且
        # --hot-expert-repin-interval-s > 0 时装配；scheduler 的 idle 安全点经
        # getattr 触发。None = 动态重钉关闭。
        self.repin_manager: HotExpertRepinManager | None = None
        # 重钉行级交换的宿主暂存（每 bank 一个 [rows, *row] pinned 张量，
        # _repin_scratch 惰性分配、跨调用复用；None = 尚未重钉过）。
        self._repin_scratch_bufs: list[torch.Tensor] | None = None

    def init_hot_pins(
        self,
        pin_ids: torch.Tensor,
        pin_counts: list[int],
        cold_row: torch.Tensor,
        pin_capacity: int | None = None,
    ) -> None:
        """装配显存钉住：校验几何、占用 slot 顶部容量区并预填钉住映射。

        Business Logic（为什么需要这个函数）:
            钉住专家必须从加载期就"已驻留"——slot 顶部区预填映射后，每次 decode 的
            合并查询都会命中钉住槽（usage 被刷新为当前 step，flashlib 的不可驱逐语义
            自我续期），钉住权重由此单副本常驻显存，host bank 只装冷专家。容量参数
            （K_cap）为运行中调 K（--tune-file 的 pin_k）预留扩容槽位：不传时容量 =
            初始钉住数，行为与固定 K 完全一致。

        Code Logic（这个函数做什么）:
            校验 pin_ids [L, K_max] / pin_counts / cold_row [L, E] 的一致性（层内
            不重复、K ≤ E、cold_row 与钉住集互逆）与容量显式值 >= 初始钉住数；随后
            解算 K_cap = max(初始钉住数, pin_capacity)，检查 LRU 区
            cache_size - L*K_cap ≥ max(2E, 512)（防 LRU 退化；prefill overlap 双缓冲
            与钉住的兼容由三源组装保证，不再互斥）。计算各层容量槽位（顶部区按层
            定宽切分，静态不变），预填 slot_for_id / id_of_slot（usage 保持 0；未活跃
            容量槽盖 usage 哨兵防 flashlib 选中），按首钉 dup 约定垫填充行，预建三源
            组装的固定 shape gather 索引与宿主冷行镜像，并打印钉住字节 / host 节省 /
            LRU 槽数。须在 set_bank_sources 之后调用（记账需要 bank 行宽）。
        """
        L, E = self.num_layers, self.num_experts
        k_init = max(pin_counts, default=0)
        if k_init > E:
            raise ValueError(f"每层钉住数 {k_init} 超过专家数 {E}")
        if pin_capacity is not None and pin_capacity < k_init:
            raise ValueError(
                f"钉住容量 pin_capacity={pin_capacity} 小于初始每层钉住数 {k_init}"
                "（容量区必须装得下初始钉住集）"
            )
        k_cap = max(k_init, pin_capacity or 0)
        if k_cap > E:
            raise ValueError(f"钉住容量 {k_cap} 超过专家数 {E}")
        assert pin_ids.shape == (L, k_init), (pin_ids.shape, pin_counts)
        assert len(pin_counts) == L
        assert all(0 <= c <= E for c in pin_counts), pin_counts
        assert cold_row.shape == (L, E) and cold_row.dtype == torch.int32
        for layer_id, count in enumerate(pin_counts):
            experts = pin_ids[layer_id, :count].tolist()
            assert len(set(experts)) == len(experts) and all(0 <= e < E for e in experts), experts
            pinned = cold_row[layer_id] < 0
            assert pinned.sum().item() == count, (layer_id, count, pinned.sum().item())
            assert all(int(cold_row[layer_id, e].item()) == -1 for e in experts)
        capacity_slots = L * k_cap
        lru_slots = self.cache_size - capacity_slots
        if lru_slots < max(2 * E, 512):
            raise ValueError(
                f"钉住容量区 L×K_cap={capacity_slots} 槽（K_cap={k_cap}）后 LRU 区只剩 "
                f"{lru_slots} < max(2*E, 512) = {max(2 * E, 512)}（防 LRU 退化）："
                f"moe_cache_size={self.cache_size} 需 ≥ L×K_cap + max(2E, 512)，"
                f"请调小每层钉住数/--hot-expert-slots 或提高 --moe-cache-size"
            )
        # pin_ids 按容量宽度存储（[L, K_cap]）：合并查询每次拷贝整条容量行，运行中
        # 扩 K 的新钉住 id 要有位可写；初始列先搬入，再按首钉 dup 约定垫填充行
        pin_ids_dev = torch.full((L, k_cap), -1, dtype=torch.int32, device=self.device)
        pin_ids_dev[:, :k_init] = pin_ids.to(device=self.device, dtype=torch.int32)
        self.pin_ids = pin_ids_dev
        self.pin_counts = list(pin_counts)
        self.cold_row = cold_row.to(device=self.device, dtype=torch.int32)
        # 运行中调 K 的静态几何与动态界：容量（层槽位定宽）、每层动态下界（= 初始
        # 钉住数：宿主冷压缩 bank 只有 E - floors[l] 行，缩 K 换出的行必须从
        # "曾钉住专家让出的行"池里取，池深 = K_active - floors[l]）与空闲行池。
        self.pin_capacity = k_cap
        self.pin_floors = list(pin_counts)
        self._pin_free_rows = [set() for _ in range(L)]
        for layer_id in range(L):
            self._pad_pin_row(layer_id)
        self._init_pin_geometry()
        self._pin_query_buffers = {}
        self._fill_pin_maps()
        self._build_pin_gather_buffers()
        pinned_bytes = self.pinned_bytes()
        logger.info_rank0(
            f"hot expert pinning: {sum(pin_counts)} experts pinned at slots "
            f"[{self.pin_base}, {self.cache_size}) (active K per layer = "
            f"{self.pin_counts}, capacity K_cap = {k_cap}); "
            f"pinned {pinned_bytes / 2**20:.1f} MiB in VRAM, host banks save the same, "
            f"LRU region = {lru_slots} slots"
        )

    def _pad_pin_row(self, layer_id: int) -> None:
        """把 pin_ids[layer_id] 的未活跃尾部垫成合并查询的无害值（首钉 dup / 全 -1）。

        Business Logic（为什么需要这个函数）:
            合并查询路径（_ensure_experts_pinned）每次调用把整条容量行拷进查询尾部，
            未活跃槽位的值也会被当成专家 id 查询：任取值可能命中 miss（误触发换入）
            甚至越界。约定"尾部 = 首钉 dup"使填充查询重复命中首钉的钉住槽（usage
            刷新，无 fetch、无越界）；层内无活跃钉住时整行 -1（该层的查询走
            comb[:1] 路由 dup 分支，不读本行）。

        Code Logic（这个函数做什么）:
            count > 0 时 pin_ids[l, count:] = pin_ids[l, 0]；count == 0 时整行填 -1。
            值改写（shape 不变），init/install/unpin 后调用。
        """
        assert self.pin_ids is not None and self.pin_counts is not None
        count = self.pin_counts[layer_id]
        row = self.pin_ids[layer_id]
        if count:
            row[count:] = row[0]
        else:
            row.fill_(-1)

    def _init_pin_geometry(self) -> None:
        """
        Business Logic（为什么需要这个函数）:
            钉住容量区槽位是 (cache_size, K_cap) 的纯函数：每层定宽 K_cap、层 l 占
            [pin_base + l*K_cap, pin_base + (l+1)*K_cap)，运行中调 K 只改活跃前缀
            长度、不改槽位布局，因此几何只在 init 与 rebuild（cache_size 变化）时
            重解算；收敛成一个无副作用除外的几何函数供两处复用。

        Code Logic（这个函数做什么）:
            写 pin_base = cache_size - L*pin_capacity（host int 镜像）与
            pin_slots [L, K_cap] int32（GPU，全容量槽位 arange，无填充值）；并把
            pin_base 同步进 pin_base_dev 设备标量（值改写、指针稳定——hybrid 内核
            按指针读取，rebuild 后已捕获 decode 图立即按新边界排除）。
        """
        L = self.num_layers
        k_cap = self.pin_capacity
        assert k_cap is not None, "init_hot_pins must set pin_capacity first"
        pin_base = self.cache_size - L * k_cap
        base = torch.arange(pin_base, pin_base + L * k_cap, dtype=torch.int32, device=self.device)
        slots = base.view(L, k_cap)
        self.pin_base = pin_base
        self.pin_slots = slots
        if self.pin_base_dev is None:
            self.pin_base_dev = torch.tensor(pin_base, dtype=torch.int32, device=self.device)
        else:
            self.pin_base_dev.fill_(pin_base)

    def _fill_pin_maps(self) -> None:
        """
        Business Logic（为什么需要这个函数）:
            钉住映射是"钉住专家永远命中"的前提：reset/rebuild 清空全部映射后必须
            立即重钉，否则钉住专家会被当成 miss 走一次冷行换入（host 已无其行），
            产生永久错误。同时未活跃容量槽必须在每次映射重建时盖 usage 哨兵：
            flashlib 合并查询路径按全 cache argmin(usage) 选受害槽，空容量槽的陈旧
            低 usage 会让冷专家被装进钉住区、被后续扩 K 覆盖。

        Code Logic（这个函数做什么）:
            逐层 scatter：slot_for_id[l, e] = 钉住槽位、id_of_slot[钉住槽位] =
            l*E + e；未活跃容量槽（j ∈ [count, K_cap)）id_of_slot 保持 -1 且 usage
            盖 _PIN_USAGE_SENTINEL。活跃槽 usage 保持 0（首次合并查询即命中并刷新为
            当前 step）。仅在 init/rebuild/reset 时调用（host 侧循环 + 设备 scatter，
            不在热路径）。
        """
        assert self.pin_ids is not None and self.pin_slots is not None
        L, E = self.num_layers, self.num_experts
        k_cap = self.pin_capacity
        for layer_id, count in enumerate(self.pin_counts):
            if count:
                flat_ids = layer_id * E + self.pin_ids[layer_id, :count].long()
                slots = self.pin_slots[layer_id, :count]
                self.slot_for_id.view(-1)[flat_ids] = slots
                self.id_of_slot[slots.long()] = flat_ids.to(torch.int32)
            if count < k_cap:
                idle_slots = self.pin_slots[layer_id, count:]
                self.id_of_slot[idle_slots.long()] = -1
                self.usage[idle_slots.long()] = _PIN_USAGE_SENTINEL

    def _build_pin_gather_buffers(self) -> None:
        """预建三源组装（钉住 + prefill overlap）的固定 shape gather 索引与宿主冷行镜像。

        Business Logic（为什么需要这个函数）:
            双缓冲的组合填充（冷行自 bank、钉住行自顶部槽）与 miss run-list 的冷行
            remap 需要逐层冷专家 id 与钉住 id/槽位索引；热路径上现算会引入分配与
            host 往返，加载期按固定 shape 预建是零热路径分配（CUDA graph 友好）的
            前提。

        Code Logic（这个函数做什么）:
            逐层调 _refresh_pin_gather_layer 取 cold_row >= 0 的冷专家 id（int32，
            冷行序）作组合填充的 dst 行、恒等 arange 作 src 行；取 pin_ids/pin_slots
            前缀作钉住行 gather 的 dst/src；行数为 [1] int64 device 标量。另把
            cold_row 复制成宿主 numpy 镜像（miss remap 的查表源，顶部槽恒命中故
            pinned -> -1 不会进 miss）。rebuild 重解算钉住几何（pin_slots 变化）后
            必须重跑；动态重钉改写 pin_ids/cold_row 值（shape 不变）后按层刷新。
        """
        assert self.pin_ids is not None and self.pin_slots is not None and self.cold_row is not None
        # 预定长度后逐层刷新（_refresh_pin_gather_layer 按索引赋值，与重钉路径共用）
        self._pin_cold_dst = [None] * self.num_layers
        self._pin_cold_src = [None] * self.num_layers
        self._pin_cold_num = [None] * self.num_layers
        self._pin_gather_dst = [None] * self.num_layers
        self._pin_gather_src = [None] * self.num_layers
        self._pin_gather_num = [None] * self.num_layers
        for layer_id in range(self.num_layers):
            self._refresh_pin_gather_layer(layer_id)
        self._cold_row_np = self.cold_row.detach().cpu().numpy()

    def _refresh_pin_gather_layer(self, layer_id: int) -> None:
        """
        Business Logic（为什么需要这个函数）:
            三源组装的固定 shape gather 索引描述"当前钉住集/冷集"的布局，动态重钉
            交换行身份后（shape 全部不变）必须按层重建，且与加载期全量构建共享
            同一份逻辑，否则两处演化会漂移。

        Code Logic（这个函数做什么）:
            重建 layer_id 的六项：_pin_cold_dst = cold_row >= 0 的冷专家 id（升序，
            即冷行序）、_pin_cold_src = 恒等 arange、两者行数 [1] 标量；
            _pin_gather_dst/src = pin_ids/pin_slots 前缀 clone、行数标量。全量重建
            （_build_pin_gather_buffers）与单次重钉交换（swap_pinned_experts）共用。
        """
        assert self.pin_ids is not None and self.pin_slots is not None and self.cold_row is not None
        count = self.pin_counts[layer_id]
        cold_ids = torch.nonzero(self.cold_row[layer_id] >= 0).flatten().to(torch.int32)
        c = int(cold_ids.numel())
        self._pin_cold_dst[layer_id] = cold_ids
        self._pin_cold_src[layer_id] = torch.arange(c, dtype=torch.int32, device=self.device)
        self._pin_cold_num[layer_id] = torch.tensor([c], dtype=torch.int64, device=self.device)
        self._pin_gather_dst[layer_id] = self.pin_ids[layer_id, :count].clone()
        self._pin_gather_src[layer_id] = self.pin_slots[layer_id, :count].clone()
        self._pin_gather_num[layer_id] = torch.tensor([count], dtype=torch.int64, device=self.device)

    def pin_query_buffer(self, num_route_ids: int) -> torch.Tensor:
        """
        Business Logic（为什么需要这个函数）:
            flashlib 路径的钉住靠"钉住 id 随每次查询附带"实现（合并 query），逐次
            分配拼接缓冲会在 decode 热路径上制造分配器压力；固定 per-形状缓冲同时
            是 CUDA graph 捕获的前提。

        Code Logic（这个函数做什么）:
            返回按 num_route_ids 惰性预分配的 [num_route_ids + K_cap] int32 device
            缓冲（尾部按容量预留：运行中扩 K 后新钉住 id 也有位可写，首次分配发生
            在 eager warmup，之后形状固定，replay 只重复读写）。
        """
        key = int(num_route_ids)
        buf = self._pin_query_buffers.get(key)
        if buf is None:
            k_cap = (
                self.pin_capacity
                if self.pin_capacity is not None
                else max(self.pin_counts, default=0)
            )
            buf = torch.empty(key + k_cap, dtype=torch.int32, device=self.device)
            self._pin_query_buffers[key] = buf
        return buf

    def pinned_bytes(self) -> int:
        """钉住权重占用的显存字节（也是 host bank 相对全量布局节省的字节）。

        逐 bank 按各层钉住数 × 单行字节累加；未钉住时为 0。"""
        if self.pin_counts is None:
            return 0
        total = 0
        for per_layer, _cache in self.banks:
            row_bytes = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            total += sum(c * row_bytes for c in self.pin_counts)
        return total

    def load_pinned_rows(self, stage: dict[str, torch.Tensor]) -> None:
        """把钉住权重的 GPU 暂存拷进 slot cache 顶部区（钉住权重入显存的最后一步）。

        Business Logic（为什么需要这个函数）:
            钉住专家的权重在 bank 加载期经 per-layer host 暂存 + GPU arena 流转过来
            （host 峰值 ≤ 单层钉住字节），但加载期 slot cache 尚未分配，只能在
            set_bank_sources 之后由本方法一次性 D2D 就位。

        Code Logic（这个函数做什么）:
            逐层把 stage[role] 的前 pin_counts[l] 行按 pin_slots[l]（行序 == pin list
            序 == 槽位分配序）写进对应 bank cache 的顶部槽；arena 行数是各层钉住数的
            最大值，超出该层钉住数的残余行不会被映射、不被读取。init_hot_pins 之后、
            首次 forward 之前调用一次。
        """
        assert self.pin_counts is not None and self.pin_slots is not None, (
            "load_pinned_rows requires init_hot_pins first"
        )
        assert self.banks, "set_bank_sources must register the banks first"
        for layer_id, count in enumerate(self.pin_counts):
            if not count:
                continue
            slots = self.pin_slots[layer_id, :count].long()
            for role, tensor in stage.items():
                cache = self.bank_caches[role]
                cache[slots] = tensor[:count].to(device=cache.device, dtype=cache.dtype)

    def pinned_id_lists(self) -> list[list[int]]:
        """每层当前钉住专家 id 的 host 快照（pin list 序 == 槽位序；未钉住为全空）。

        动态重钉决策与 ``--hot-stats-out`` 落盘 pinned 字段的数据源：只在 idle 安全点
        或退出落盘时调用，一次 pin_ids 的 D2H 可接受。
        """
        if self.pin_ids is None or self.pin_counts is None:
            return [[] for _ in range(self.num_layers)]
        host = self.pin_ids.detach().cpu()
        return [host[layer_id, :count].tolist() for layer_id, count in enumerate(self.pin_counts)]

    def _repin_scratch(self, rows: int) -> list[torch.Tensor]:
        """行级交换的宿主暂存（每 bank 一个 [rows, *row] 张量），惰性分配、跨调用复用。

        Business Logic（为什么需要这个函数）:
            重钉把被替换钉住行 D2H 到宿主再写回 bank，必须先落在可写的 host 缓冲；
            重钉低频但每周期 ≤ max_swaps × L 对，按需分配一次并复用即可，避免每次
            交换都向分配器要 pinned 内存。

        Code Logic（这个函数做什么）:
            首次调用（或现有缓冲行数不足）时按各 bank 行形状分配 rows 行（CUDA 设备
            用 pin_memory 加速 D2H）；返回每 bank 的暂存张量列表。
        """
        bufs = self._repin_scratch_bufs
        if bufs is None or bufs[0].shape[0] < rows:
            bufs = [
                torch.empty(
                    (rows, *cache.shape[1:]),
                    dtype=cache.dtype,
                    pin_memory=(self.device.type == "cuda"),
                )
                for _per_layer, cache in self.banks
            ]
            self._repin_scratch_bufs = bufs
        return bufs

    def swap_pinned_experts(self, layer_id: int, swaps: list[tuple[int, int]]) -> None:
        """动态重钉的行级交换：候选冷专家 c 上位到被替换钉住专家 h 的顶部槽。

        Business Logic（为什么需要这个函数）:
            域漂移后冷专家可能比旧钉住专家更热，把两者的权威驻留对调（c 的字节上
            显存、h 的字节回宿主 bank），热集就能跟随负载演进；交换必须同时搬字节
            与改映射，且对 decode/prefill 全部读者表现为"原子"——这要求它在 idle
            安全点（所有流同步、无在途 GEMM/CPU GEMV/未完成 prefill chunk）执行，
            本方法内不再处理并发。

        Code Logic（这个函数做什么）:
            对每对 (c, h)：host 解析 c 的冷行 r_c 与 h 的钉住槽 s_h（校验 c ∈ 冷集、
            s_h ∈ 顶部区、swaps 无重复专家）后，a) s_h 行 D2H 进复用的宿主暂存；
            c) bank[r_c] H2D 进 s_h（c 上位——必须先于写回，写回会覆盖 c 的原始字
            节）；b) 暂存 host-to-host 写回 bank[r_c]（h 回填宿主，bank 的唯一运行期
            写者，见 host_banks 的并发约定）；d) 原子化改写映射值：pin_ids[l, j_h] =
            c、cold_row[l, c] = -1、cold_row[l, h] = r_c、slot_for_id[l, c] = s_h、
            slot_for_id[l, h] = -1、id_of_slot[s_h] = l*E+c（usage 刷成当前 step，
            衔接 flashlib 不可驱逐语义的自我续期）；e) 刷新该层三源组装的 gather
            索引与 cold_row 宿主镜像。全部为既有张量的值改写（shape 不变 ⇒ CUDA
            graph 兼容），flashlib 合并查询/hybrid 范围保护天然读新值。
        """
        # 缓冲是 launch.py 的 torch.inference_mode() 下构造的 inference tensor：本原语
        # 的宿主侧就地写（copy_/index_copy_/索引赋值）必须在模式内执行（idle 安全点
        # 在调度线程、模式外），否则抛 Inplace-update 运行时错误。
        with torch.inference_mode():
            self._swap_pinned_experts_impl(layer_id, swaps)

    def _swap_pinned_experts_impl(self, layer_id: int, swaps: list[tuple[int, int]]) -> None:
        """
        Business Logic（为什么需要这个函数）:
            swap_pinned_experts 的实现体（公开方法已进入 inference_mode 栅栏）。

        Code Logic（这个函数做什么）:
            见 swap_pinned_experts 的 docstring。
        """
        assert self.pin_ids is not None and self.pin_counts is not None, "init_hot_pins first"
        assert self.pin_slots is not None and self.cold_row is not None, "init_hot_pins first"
        assert self.banks and self._cold_row_np is not None, "swap needs pinning + banks"
        assert 0 <= layer_id < self.num_layers, layer_id
        if not swaps:
            return
        count = self.pin_counts[layer_id]
        assert count > 0, f"layer {layer_id} has no pinned experts"
        E = self.num_experts
        # host 侧解析当前映射（idle 安全点，同步读取）
        pin_host = self.pin_ids[layer_id, :count].cpu().tolist()
        slot_host = self.slot_for_id[layer_id].cpu().tolist()
        cold_host = self.cold_row[layer_id].cpu().tolist()
        pin_pos = {e: j for j, e in enumerate(pin_host)}
        cs: list[int] = []
        hs: list[int] = []
        js: list[int] = []
        s_list: list[int] = []
        r_list: list[int] = []
        stale_slots: list[int] = []
        seen_c: set[int] = set()
        seen_h: set[int] = set()
        for cand, replaced in swaps:
            r = cold_host[cand]
            if r < 0:
                raise ValueError(f"layer {layer_id}: 候选专家 {cand} 不在冷集（cold_row={r}），拒绝交换")
            s = slot_host[replaced]
            if s < self.pin_base:
                raise ValueError(
                    f"layer {layer_id}: 被替换专家 {replaced} 的槽 {s} 不在钉住区 "
                    f"[{self.pin_base}, {self.cache_size})，拒绝交换"
                )
            pair = (cand, replaced)
            if cand in seen_c or replaced in seen_h:
                raise ValueError(f"layer {layer_id}: 交换对 {pair} 与前面的交换重复使用同一专家")
            seen_c.add(cand)
            seen_h.add(replaced)
            cs.append(cand)
            hs.append(replaced)
            js.append(pin_pos[replaced])
            s_list.append(s)
            r_list.append(r)
            # 候选可能恰有 LRU 残留副本槽（冷 ⟺ 未钉住，不 ⟺ 不在显存）：换入后其
            # 权威槽变为 s，旧 LRU 槽的 id 映射必须解除，否则后续驱逐按旧 id 反清
            # slot_for_id 会杀掉新钉映射。旧槽本身保留为匿名 LRU 内容（usage 陈旧，
            # 自然回到正常驱逐循环）。
            prev = slot_host[cand]
            if prev >= 0:
                stale_slots.append(prev)
        n = len(cs)
        c_t = torch.tensor(cs, dtype=torch.long, device=self.device)
        h_t = torch.tensor(hs, dtype=torch.long, device=self.device)
        j_t = torch.tensor(js, dtype=torch.long, device=self.device)
        s_t = torch.tensor(s_list, dtype=torch.long, device=self.device)
        import os as _os

        if not _os.getenv("FREETOKEN_REPIN_SKIP_BYTES"):
            scratch = self._repin_scratch(n)
            for i, (per_layer, cache) in enumerate(self.banks):
                # a. 被替换钉住行 -> 宿主暂存（同步 D2H；idle 点无并发读者）。
                # 逐行 slice copy_：fp8 等 bank dtype 的 index 系内核（index_copy_ /
                # 花式索引 gather）在 CPU 与 CUDA 上均未实现，copy_ 则是纯字节搬运，
                # 对任意 dtype 都可用——真实 NVFP4 演示验证过这一点。
                for k in range(n):
                    scratch[i][k].copy_(cache[s_list[k]])
            for i, (per_layer, cache) in enumerate(self.banks):
                # c. 候选冷行 -> 顶部槽（H2D）：c 的权重上显存，c 从此由钉住槽权威驻留。
                # 必须先于 b：b 会把 c 的原始字节覆盖掉。
                for k in range(n):
                    cache[s_list[k]].copy_(per_layer[layer_id][r_list[k]])
            for i, (per_layer, cache) in enumerate(self.banks):
                # b. 暂存 -> 候选旧冷行（host-to-host）：h 的权重回填 bank，h 从此由 bank 权威驻留
                for k in range(n):
                    per_layer[layer_id][r_list[k]] = scratch[i][k]
        # d. 映射原子化更新（值改写，shape 不变 -> CUDA graph 兼容；int32/int64 映射
        # 张量的 index 内核不受 bank dtype 限制）
        flat_c = torch.tensor([layer_id * E + c for c in cs], dtype=torch.int32, device=self.device)
        self.pin_ids[layer_id].index_copy_(0, j_t, c_t.to(torch.int32))
        self.cold_row[layer_id, c_t] = -1
        self.cold_row[layer_id, h_t] = torch.tensor(r_list, dtype=torch.int32, device=self.device)
        self.slot_for_id[layer_id, c_t] = s_t.to(torch.int32)
        self.slot_for_id[layer_id, h_t] = -1
        self.id_of_slot[s_t] = flat_c
        if stale_slots:
            # 候选旧 LRU 副本槽去映射（见上文 stale_slots 收集处的说明）
            self.id_of_slot[torch.tensor(stale_slots, dtype=torch.long, device=self.device)] = -1
        # usage 刷成当前 step：flashlib 路径的钉住槽靠"usage == step 不可驱逐"自我
        # 续期，交换后到该层下一次合并查询之间有其他层 ensure 的窗口，旧 usage 的
        # 钉住槽理论上可能成为全局 argmin 受害者；刷成当前 step 后它是最年轻的
        # stale 槽（平局时 argmin 取最小槽位号，顶部区槽号最大，永不胜出）。
        self.usage[s_t] = int(self.step.item())
        self._cold_row_np[layer_id, cs] = -1
        self._cold_row_np[layer_id, hs] = r_list
        # e. 该层三源组装的 gather 索引随新钉住集/冷集重建
        self._refresh_pin_gather_layer(layer_id)

    def install_pinned_experts(self, layer_id: int, experts: list[int]) -> list[int]:
        """运行中扩 K：把冷专家逐个装入该层的空闲容量槽（动态钉住的原语之一）。

        Business Logic（为什么需要这个函数）:
            负载漂移后每层钉住数 K 需要"扩"：窗口热度选出的新热专家要拿到权威显存
            槽位——容量区按 K_cap 静态预留，层内第一个未活跃槽就是它的位置；扩容后
            对 decode/prefill 全部读者表现为"该专家从此恒命中"，与加载期钉住语义
            完全一致。须在 idle 安全点（所有流同步、无在途 GEMM/CPU GEMV/未完成
            prefill chunk）调用，本方法内不处理并发。

        Code Logic（这个函数做什么）:
            对每个冷专家 e（校验 e ∈ 冷集、未越容量）：a) 取该层第 count 个容量槽
            s = pin_slots[l, count]；若 e 恰有 LRU 残留副本槽则解除其 id 映射（防
            后续驱逐按旧 id 反清新钉映射，同 swap 的 stale_slots 处理）；b) bank 冷
            行 H2D 进 s（e 的字节上显存）；c) 原子化改写映射值：pin_ids[l, count] =
            e、cold_row[l, e] = -1、slot_for_id[l, e] = s、id_of_slot[s] = l*E+e、
            usage[s] = 当前 step（衔接 flashlib 不可驱逐语义的自我续期），冷行让给
            空闲行池；d) pin_counts[l] += 1，按首钉 dup 约定重垫填充行，刷新该层三源
            组装 gather 索引与 cold_row 宿主镜像。全部为既有张量的值改写（shape 不变
            ⇒ CUDA graph 兼容），返回实际装入的专家列表。
        """
        # inference tensor 栅栏：同 swap_pinned_experts 的说明。
        with torch.inference_mode():
            return self._install_pinned_experts_impl(layer_id, experts)

    def _install_pinned_experts_impl(self, layer_id: int, experts: list[int]) -> list[int]:
        """
        Business Logic（为什么需要这个函数）:
            install_pinned_experts 的实现体（公开方法已进入 inference_mode 栅栏）。

        Code Logic（这个函数做什么）:
            见 install_pinned_experts 的 docstring。
        """
        self._require_dynamic_pin_state(layer_id)
        free = self._pin_free_rows
        assert free is not None and self.pin_slots is not None and self.pin_ids is not None
        assert self.cold_row is not None and self.banks and self._cold_row_np is not None
        E = self.num_experts
        # host 侧解析当前映射（idle 安全点，同步读取）
        cold_host = self.cold_row[layer_id].cpu().tolist()
        slot_host = self.slot_for_id[layer_id].cpu().tolist()
        step = int(self.step.item())
        installed: list[int] = []
        s_list: list[int] = []
        r_list: list[int] = []
        stale_slots: list[int] = []
        for e in experts:
            if not 0 <= e < E:
                raise ValueError(f"layer {layer_id}: 专家 id {e} 越界（E={E}）")
            if self.pin_counts[layer_id] + len(installed) >= self.pin_capacity:
                raise ValueError(
                    f"layer {layer_id}: 容量槽已满（K_cap={self.pin_capacity}），"
                    f"无法再装入专家 {e}"
                )
            r = cold_host[e]
            if r < 0:
                raise ValueError(
                    f"layer {layer_id}: 专家 {e} 不在冷集（cold_row={r}），无法钉住"
                )
            s = int(self.pin_slots[layer_id, self.pin_counts[layer_id] + len(installed)].item())
            prev = slot_host[e]
            if prev >= 0:
                stale_slots.append(prev)
            installed.append(e)
            s_list.append(s)
            r_list.append(r)
        if not installed:
            return installed
        n = len(installed)
        s_t = torch.tensor(s_list, dtype=torch.long, device=self.device)
        e_t = torch.tensor(installed, dtype=torch.long, device=self.device)
        r_t = torch.tensor(r_list, dtype=torch.long, device=self.device)
        flat_e = torch.tensor(
            [layer_id * E + e for e in installed], dtype=torch.int32, device=self.device
        )
        if not os.getenv("FREETOKEN_REPIN_SKIP_BYTES"):
            for i, (per_layer, cache) in enumerate(self.banks):
                # 冷行 -> 容量槽（H2D）：逐行 slice copy_，任意 bank dtype 可用
                #（理由同 swap_pinned_experts 的注释）
                for k in range(n):
                    cache[s_list[k]].copy_(per_layer[layer_id][r_list[k]])
        # 映射原子化更新（值改写，shape 不变 -> CUDA graph 兼容）
        j0 = self.pin_counts[layer_id]
        j_t = torch.arange(j0, j0 + n, dtype=torch.long, device=self.device)
        self.pin_ids[layer_id].index_copy_(0, j_t, e_t.to(torch.int32))
        self.cold_row[layer_id, e_t] = -1
        self.slot_for_id[layer_id, e_t] = s_t.to(torch.int32)
        self.id_of_slot[s_t] = flat_e
        if stale_slots:
            self.id_of_slot[torch.tensor(stale_slots, dtype=torch.long, device=self.device)] = -1
        # usage 刷成当前 step（理由同 swap_pinned_experts）；冷行让给空闲行池
        self.usage[s_t] = step
        for r in r_list:
            free[layer_id].add(r)
        self.pin_counts[layer_id] += n
        self._cold_row_np[layer_id, installed] = -1
        self._pad_pin_row(layer_id)
        self._refresh_pin_gather_layer(layer_id)
        return installed

    def unpin_tail_experts(self, layer_id: int, new_count: int) -> list[int]:
        """运行中缩 K：把该层尾部活跃钉住专家换出到宿主空闲行（动态钉住的原语之一）。

        Business Logic（为什么需要这个函数）:
            负载漂移后每层钉住数 K 需要"缩"：尾部位（pin list 序末尾）的钉住专家让
            出权威地位、字节回宿主 bank；容量槽留空（仍在排除区，不参与 LRU——固定
            容量布局的已知取舍）。换出的宿主行必须来自空闲行池：bank 行数 = 加载期
            E - floors[l] 固定，缩 K 释放的行恰好供扩 K 复用，池深 = K_active -
            floors[l] 保证缩到 floors[l] 之前永不行穷。

        Code Logic（这个函数做什么）:
            校验 floors[l] <= new_count < pin_counts[l] 后，对尾部每个位置 j（从末
            尾向前）：a) 取空闲行 r = min(池)（确定性）；b) 钉住槽字节 D2H 写回
            bank[r]；c) 原子化改写映射值：cold_row[l, e] = r、slot_for_id[l, e] =
            -1、id_of_slot[s] = -1、usage[s] 盖 _PIN_USAGE_SENTINEL（空容量槽对
            flashlib argmin 隐形）；d) pin_counts[l] 递减。最后按首钉 dup 约定重垫
            填充行并刷新该层三源组装 gather 索引与 cold_row 宿主镜像。返回换出的
            专家列表（pin list 序）。
        """
        # inference tensor 栅栏：同 swap_pinned_experts 的说明。
        with torch.inference_mode():
            return self._unpin_tail_experts_impl(layer_id, new_count)

    def _unpin_tail_experts_impl(self, layer_id: int, new_count: int) -> list[int]:
        """
        Business Logic（为什么需要这个函数）:
            unpin_tail_experts 的实现体（公开方法已进入 inference_mode 栅栏）。

        Code Logic（这个函数做什么）:
            见 unpin_tail_experts 的 docstring。
        """
        self._require_dynamic_pin_state(layer_id)
        free = self._pin_free_rows
        assert free is not None and self.pin_slots is not None and self.pin_ids is not None
        assert self.cold_row is not None and self.banks and self._cold_row_np is not None
        floor = self.pin_floors[layer_id]
        old_count = self.pin_counts[layer_id]
        if not old_count > new_count >= floor:
            raise ValueError(
                f"layer {layer_id}: 目标钉住数 {new_count} 必须在 "
                f"[floors={floor}, 当前={old_count}) 区间内才能缩"
            )
        unpinned: list[int] = []
        s_list: list[int] = []
        r_list: list[int] = []
        for j in range(old_count - 1, new_count - 1, -1):
            if not free[layer_id]:
                raise RuntimeError(
                    f"layer {layer_id}: 空闲行池为空（K={old_count} 已到加载期下界 "
                    f"{floor}），无法再换出"
                )
            e = int(self.pin_ids[layer_id, j].item())
            s = int(self.pin_slots[layer_id, j].item())
            r = min(free[layer_id])
            free[layer_id].discard(r)
            unpinned.append(e)
            s_list.append(s)
            r_list.append(r)
        if not unpinned:
            return unpinned
        n = len(unpinned)
        s_t = torch.tensor(s_list, dtype=torch.long, device=self.device)
        e_t = torch.tensor(unpinned, dtype=torch.long, device=self.device)
        r_t = torch.tensor(r_list, dtype=torch.long, device=self.device)
        if not os.getenv("FREETOKEN_REPIN_SKIP_BYTES"):
            for i, (per_layer, cache) in enumerate(self.banks):
                # 钉住槽 -> 空闲冷行（D2H）：逐行 slice copy_，任意 bank dtype 可用
                for k in range(n):
                    per_layer[layer_id][r_list[k]].copy_(cache[s_list[k]])
        # 映射原子化更新（值改写，shape 不变 -> CUDA graph 兼容）
        self.cold_row[layer_id, e_t] = r_t.to(torch.int32)
        self.slot_for_id[layer_id, e_t] = -1
        self.id_of_slot[s_t] = -1
        # 空容量槽盖哨兵：flashlib argmin 永不选中（hybrid 由 pin_base 排除，不受影响）
        self.usage[s_t] = _PIN_USAGE_SENTINEL
        self.pin_counts[layer_id] = new_count
        self._cold_row_np[layer_id, unpinned] = r_list
        self._pad_pin_row(layer_id)
        self._refresh_pin_gather_layer(layer_id)
        return unpinned

    def _require_dynamic_pin_state(self, layer_id: int) -> None:
        """动态扩缩原语的公共前置校验（钉住已装配、bank 就绪、层号合法）。"""
        assert self.pin_ids is not None and self.pin_counts is not None, "init_hot_pins first"
        assert self.pin_capacity is not None and self.pin_floors is not None, "init_hot_pins first"
        assert self.pin_slots is not None and self.cold_row is not None, "init_hot_pins first"
        assert self.banks and self._cold_row_np is not None, "dynamic pin ops need pinning + banks"
        assert 0 <= layer_id < self.num_layers, layer_id

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
        per_layer_rows: list[int] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.layers.quantization.moe.nvfp4`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``per_layer_rows``（显存钉住的冷压缩 bank）：每层期望的 bank 行数（钉住层为
        ``num_experts - K_l``，其余层为 ``num_experts``）；缺省 None = 全部
        ``num_experts``（既有行为）。

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to the CPU executor (``cpu_layer_ids``, set BEFORE this call), the copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.
        """
        from freetoken.moe.legacy_format import canonical_role
        from freetoken.moe.host_banks import HostResidency

        # loaders and FTW files may still name the banks the old way (gate_up_packed, ...)
        by_role = {canonical_role(name): per_layer for name, per_layer in sources.items()}
        if set(by_role) != {canonical_role(n) for n in self.bank_schema}:
            raise AssertionError(
                f"banks {sorted(sources)} do not match the {self.quant_format!r} schema {self.bank_schema}"
            )
        sources = {name: by_role[canonical_role(name)] for name in self.bank_schema}
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        rows = per_layer_rows or [self.num_experts] * self.num_layers
        assert len(rows) == self.num_layers and all(
            isinstance(r, int) and 1 <= r <= self.num_experts for r in rows
        ), rows
        unpinned = frozenset(
            i for i, r in enumerate(residency) if r != HostResidency.PINNED.value
        )
        if unpinned:
            if not unpinned <= self.cpu_layer_ids:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - self.cpu_layer_ids)} are not in "
                    f"cpu_layer_ids: a layer without a device address can only decode on "
                    f"the CPU executor (set cache.cpu_layer_ids before set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            if self.layout is not None:
                spec = self.layout[name]
                if tuple(head.shape[1:]) != tuple(spec.shape) or head.dtype != spec.dtype:
                    raise ValueError(
                        f"bank {name!r} rows are {tuple(head.shape[1:])} {head.dtype} but the expert kernel's layout "
                        f"wants {tuple(spec.shape)} {spec.dtype}; the banks were packed for another kernel"
                    )
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                # 冷压缩 bank 行数 = per_layer_rows[l]（钉住层 E-K_l）；行形状逐层一致
                assert source.size(0) == rows[layer_id], (name, layer_id, source.shape, rows[layer_id])
                assert source.shape[1:] == head.shape[1:] and source.dtype == head.dtype, (
                    name, layer_id, source.shape, head.dtype,
                )
            self.bank_sources[name] = list(per_layer)
            self.bank_caches[name] = torch.zeros(
                (self.cache_size, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _build_copy_plan(self) -> None:
        self._build_fused_copy_plan()
        if self._copy_fused_ok or self.device.type != "cuda" or not self.banks:
            return
        for name in self.bank_schema:
            cache = self.bank_caches[name]
            feat = math.prod(cache.shape[1:]) * cache.element_size()
            if feat % 128:
                raise RuntimeError(
                    f"MoE bank {name!r} rows are {feat} bytes (not a multiple of 128): "
                    f"only the fused multi-bank copy can move them, but it is disabled"
                )

    def _build_fused_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        dst_ptrs, feats = [], []
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for per_layer, cache in self.banks:
            feat = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                return  # leave fused disabled; copy_missing uses the per-bank path
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
            dst_ptrs.append(cache.data_ptr())
            feats.append(feat)
        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        if self.pin_counts is not None:
            # 按容量区校验（K_cap 静态预留，与运行中活跃 K 无关）：缩到下限/扩到上限
            # 都不能侵犯 LRU 地板
            assert self.pin_capacity is not None
            capacity_slots = self.num_layers * self.pin_capacity
            lru_slots = cache_size - capacity_slots
            if lru_slots < max(2 * self.num_experts, 512):
                raise ValueError(
                    f"rebuild 目标 cache_size {cache_size} 扣除钉住容量区 "
                    f"L×K_cap={capacity_slots}（K_cap={self.pin_capacity}）后 LRU 区 "
                    f"{lru_slots} < max(2*E, 512) = {max(2 * self.num_experts, 512)}"
                )
        if self.max_slots is not None and cache_size > self.max_slots:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the expert kernel's slot limit of {self.max_slots}; "
                f"pass --moe-cache-size {self.max_slots} or less, or let the default kernel serve the experts"
            )
        if self.layout is None and self.quant_format == "nvfp4_marlin" and cache_size > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --quant-backend moe.nvfp4=triton"
            )

    def rebuild(self, cache_size: int) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        self.validate_rebuild(cache_size)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self._prefill_buffer_ptrs = []
        self._compose_cache_ptrs = None
        self._compose_host_ptrs = None
        self._compose_feat_bytes = None
        self._overlap_small_bank_ids = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self.cache_size = cache_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            self.bank_caches[name] = torch.zeros(
                (cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        # 钉住映射是 cache_size 的函数：几何重解算 + 立即重钉（reset 后 slot_for_id
        # 全空，不重钉的话钉住专家会被当 miss 换入，而 host 已无其行）
        if self.pin_ids is not None:
            self._pin_query_buffers = {}
            self._init_pin_geometry()
            self._fill_pin_maps()
            # pin_slots 随 cache_size 变了，钉住行 gather 索引同步刷新
            self._build_pin_gather_buffers()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and cache_size < 2 * self.num_experts:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< 2*num_experts {2 * self.num_experts}."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid"), (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'}"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        idx = layer_id * self.num_experts + (
            self.id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the full ``[S]`` slot cache
        (decode), or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # The double buffers borrow the slot cache's first 2 * num_experts slots
        # (one full expert layer per buffer), one view per registered bank.
        self.prefill_bank_buffers = [
            cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
            for _, cache in self.banks
        ]
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # 三源组装（钉住 + overlap）的组合填充描述符：双缓冲首行指针、slot cache 与
        # 宿主 bank 的 base 指针、行字节数、小 bank 集合。独立于 fused copy plan 的
        # 启用状态（FREETOKEN_FUSED_COPY=0 等场景下 copy_missing 退化为 per-bank，
        # 而组合填充仍可用自己的描述符）。unpinned 层与 overlap 互斥（见
        # set_bank_sources），所以宿主指针永远可解析。
        feats = [math.prod(cache.shape[1:]) * cache.element_size() for _, cache in self.banks]
        self._overlap_small_bank_ids = [b for b, f in enumerate(feats) if f < _SMALL_BANK_FEAT_BYTES]
        if self.device.type == "cuda":
            from freetoken.kernel.pinned import device_ptr

            self._prefill_buffer_ptrs = [
                torch.tensor(
                    [buf[b].data_ptr() for buf in self.prefill_bank_buffers],
                    dtype=torch.int64,
                    device=self.device,
                )
                for b in range(2)
            ]
            self._compose_cache_ptrs = torch.tensor(
                [cache.data_ptr() for _, cache in self.banks], dtype=torch.int64, device=self.device
            )
            self._compose_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
            for f in feats:
                # fast_index_copy_multi 内核要求行字节 16 对齐（torch 分配天然满足；
                # 显式断言以防未来引入奇异行宽）
                assert f % 16 == 0, f
            self._compose_host_ptrs = [
                torch.tensor(
                    [device_ptr(per_layer[layer_id]) for per_layer, _ in self.banks],
                    dtype=torch.int64,
                    device=self.device,
                )
                for layer_id in range(self.num_layers)
            ]

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
        # ensure_experts evicts them first.
        self.usage[slot_start:slot_end].zero_()

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            # 快照一致性前提：钉住映射（顶部区 slot_for_id / cold_row）在 chunk 之间
            # 不变——动态重钉只允许发生在 idle 安全点（chunk 边界，无在途 prefill），
            # 因此 begin 时刻的快照对整个 chunk 的 hit/miss 分类与 miss 冷行 remap
            # 都有效。
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")

        assert self.banks and self.prefill_bank_buffers

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            if self.pin_ids is None:
                for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
                    buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)
                return
            # 钉住：bank 行数 E-K != E，整层 copy_ 形状不符——三源组合填充
            # （冷行自宿主冷压缩 bank + 钉住行自 slot 顶部区），position == id 不变。
            self._compose_prefill_banks(layer_id, buffer_id, small_only=False)

        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif self.pin_ids is not None and self.pin_base < 2 * self.num_experts:
            # 钉住区必须整体落在 hit 阈值之上：顶部槽低于 2E 会被分类成 miss，而
            # 钉住专家在宿主 bank 里没有行可拷。LRU 地板（pin_base >= max(2E,512)）
            # 使其实际不可达，这里兜底并给出可读原因。
            reason = (
                f"the pinned top region starts at slot {self.pin_base}, inside the double "
                f"buffers' borrow region [0, {2 * self.num_experts}): pinned rows would "
                f"classify as misses and no host bank row exists for them"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.

        三源组装（钉住）：钉住槽位于顶部区（>= 2E）恒分类为 hit，由 D2D gather 从
        权威槽取行；miss 侧的宿主 run-list 经 cold_row 宿主镜像把专家 id 重映射为
        冷压缩 bank 行号（cold_row 保序压缩，连续专家 id 的 run 仍对应连续冷行，
        coalescing 不变；钉住 -> -1 不可能出现在 miss 里，出现即映射损坏，防御性
        拒绝）。小行宽 bank 不进 batch（子 256KB 条目会让 batch 退化同步拷贝）也
        不进 gather，其整行集合（冷 + 钉住）由组合填充覆盖。未钉住时行为与原先
        逐字节一致。
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= 2 * E
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        pinned = self.pin_ids is not None
        cold_np = self._cold_row_np[layer_id] if pinned else None
        if self._gather_dst_ptrs is not None:
            prefill_hit_compact(self, layer_id, buffer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
        miss = np.nonzero(~hit_mask)[0]
        if pinned and miss.size:
            # 钉住槽恒命中（pin_base >= 2E ⇒ slot >= 2E ⇒ hit）；miss 中出现
            # cold_row == -1 说明钉住映射已损坏，拒绝错拷贝胜过静默给错权重。
            stale = cold_np[miss] < 0
            if stale.any():
                raise RuntimeError(
                    f"layer {layer_id}: pinned expert(s) {miss[stale].tolist()} classified "
                    f"as prefill miss (snapshot slots {snap[miss[stale]].tolist()}); "
                    "the pinned slot map is corrupted"
                )
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            starts = lengths = cold_starts = None
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
                if pinned:
                    # cold_row 表是 int32；指针偏移乘法（rows * feat）会超 int32，
                    # 与 np.nonzero 的 starts 一样统一升到 int64
                    cold_starts = cold_np[starts].astype(np.int64)
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                if feat < _SMALL_BANK_FEAT_BYTES:
                    if pinned:
                        # 小 bank 不进 batch（batch 混入子 256KB 条目会整体退化为
                        # 同步拷贝）：冷行与钉住行都交给组合填充（bank 字节即权威）。
                        continue
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._copy_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    rows = cold_starts if pinned else starts
                    dst.extend(self._copy_dst_ptrs_host[b] + (buffer_id * E + starts) * feat)
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + rows * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            if pinned and self._overlap_small_bank_ids:
                self._compose_prefill_banks(layer_id, buffer_id, small_only=True)
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def _compose_prefill_banks(self, layer_id: int, buffer_id: int, small_only: bool) -> None:
        """三源组合填充双缓冲的一层：冷行自宿主冷压缩 bank、钉住行自 slot 顶部区。

        Business Logic（为什么需要这个函数）:
            钉住后 host bank 只装冷专家（[E-K] 行），buffer [E, *row] 的每一行必须
            由两个来源拼出：冷专家取 bank 冷行（无论运行期命中与否——bank 字节即冷
            专家的权威驻留），钉住专家取顶部权威槽。hit-d2d 不可用时的整层拷贝回退
            （全部 bank）与 hit-d2d 路径的小 bank 填充（小行宽 bank 既不进 batch
            也不进命中 gather）共用本函数；buffer 的 position == 专家 id 契约不变，
            GEMM 侧零感知。

        Code Logic（这个函数做什么）:
            CUDA 上对目标 bank 集合做两次 fast_index_copy_multi_jit：冷行 gather
            （宿主 bank UVA 指针，src 行 = 恒等冷行序，dst 行 = 冷专家 id）、钉住
            行 gather（slot cache 指针，src 行 = 钉住槽，dst 行 = 钉住 id）；索引
            与行数缓冲由 _build_pin_gather_buffers 按固定 shape 预建，热路径零分配，
            描述符不依赖 fused copy plan 的启用状态。small_only 时把描述符按预建的
            小 bank 集合切片。CPU 设备（测试镜像）走等价的 torch index_copy_ 组合。
            FREETOKEN_SKIP_FAST_INDEX_COPY=1 时与其他 fast_index_copy 消费方一样
            整体跳过（消融旋钮的既定语义：输出仅在缓存已有内容时有意义）。
        """
        if self.device.type != "cuda":
            bank_ids = range(len(self.banks))
            if small_only:
                bank_ids = self._overlap_small_bank_ids
            for i in bank_ids:
                per_layer, cache = self.banks[i]
                target = self.prefill_bank_buffers[i][buffer_id]
                # 冷源按活跃冷行数切片：运行中调 K 后 bank 行数（E - 加载期 K）可能
                # 多于活跃冷专家数（空闲行池），多出的行不进组合填充
                cold_dst = self._pin_cold_dst[layer_id]
                target.index_copy_(
                    0, cold_dst.long(), per_layer[layer_id][: cold_dst.numel()]
                )
                if self.pin_counts[layer_id]:
                    rows = cache[self._pin_gather_src[layer_id].long()]
                    target.index_copy_(0, self._pin_gather_dst[layer_id].long(), rows)
            return
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        buf_ptrs = self._prefill_buffer_ptrs[buffer_id]
        host_ptrs = self._compose_host_ptrs[layer_id]
        cache_ptrs, feats = self._compose_cache_ptrs, self._compose_feat_bytes
        if small_only:
            if not self._overlap_small_bank_ids:
                return
            sel = torch.tensor(self._overlap_small_bank_ids, dtype=torch.int64, device=self.device)
            buf_ptrs = buf_ptrs.index_select(0, sel)
            host_ptrs = host_ptrs.index_select(0, sel)
            cache_ptrs = cache_ptrs.index_select(0, sel)
            feats = feats.index_select(0, sel)
        # 冷源：宿主冷压缩 bank，第 r 行 -> buffer 的第 cold_ids[r] 行（position==id）
        fast_index_copy_multi_jit(
            buf_ptrs,
            host_ptrs,
            feats,
            self._pin_cold_dst[layer_id],
            self._pin_cold_src[layer_id],
            self._pin_cold_num[layer_id],
        )
        if self.pin_counts[layer_id]:
            # 钉住源：slot cache 顶部权威槽，第 j 个钉住槽 -> buffer 的第 pin id 行
            fast_index_copy_multi_jit(
                buf_ptrs,
                cache_ptrs,
                feats,
                self._pin_gather_dst[layer_id],
                self._pin_gather_src[layer_id],
                self._pin_gather_num[layer_id],
            )

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        assert self.prefill_overlap
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts(self, layer_id, expert_ids)

    def set_fetch_params(self, max_fetch: int, fetch_fraction: float) -> None:
        """运行中更新 hybrid 拉取参数（cap + 比例），对已捕获 decode 图立即生效。

        Business Logic（为什么需要这个方法）:
            hybrid 每步的 PCIe 拉取 cap/比例原先以内核标量实参传入，CUDA graph 捕获
            时被冻结，运行中改 Python 属性无法影响已捕获的 decode 图；运行中调参
            （--tune-file 轮询 fetch_fraction）因此必须改走"设备张量按指针读取"的
            值更新路径——本方法是唯一的合法写入口（无 profile 旧语义 = 固定 cap 1、
            无比例分流，仍以 set_fetch_params(1, 0.0) 表达）。

        Code Logic（这个函数做什么）:
            宿主侧用与 ensure_experts_hybrid 入口完全一致的公式（fetch_fraction_q16）
            把比例换算成 Q16 定点；同步更新 Python 属性 hybrid_max_fetch /
            hybrid_fetch_fraction（CPU 参考镜像与测试读属性），并把 [max_fetch,
            frac_q16] copy_ 进设备张量 fetch_params（指针稳定，只改值，图安全）。
            非 hybrid 模式 fetch_params 为 None，此时只更新属性（hybrid 内核不会
            被调用）。
        """
        self.hybrid_max_fetch = int(max_fetch)
        self.hybrid_fetch_fraction = float(fetch_fraction)
        if self.fetch_params is not None:
            from freetoken.moe.offload_kernels import fetch_fraction_q16

            # cache 缓冲在 launch.py 的 torch.inference_mode() 下构造（inference
            # tensor）：模式外的宿主侧就地写会抛 Inplace-update 运行时错误
            # （--tune-file 轮询线程不在该模式内），就地写必须显式进入该模式。
            with torch.inference_mode():
                self.fetch_params.copy_(
                    torch.tensor(
                        [self.hybrid_max_fetch, fetch_fraction_q16(fetch_fraction)],
                        dtype=torch.int32,
                        device=self.device,
                    )
                )

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction
        )

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)
        # 显存钉住（仅非 overlap 的 materialize prefill 路径）：把钉住权重从顶部权威
        # 槽 D2D 安装进 [0, E) 的 position==id 暂存位，prefill GEMM 因而保持原始形态
        # （原始 id 直通、views 取前 E 行、n=E）。暂存位在映射上是空槽（materialize
        # kernel 清成 id=-1/usage=0），只服务本层本步的 prefill GEMM，任何后续
        # ensure/materialize 都会按既有语义改写它们。overlap 路径绝不走到这里
        # （layers/moe.py 分支唯一），且 [0, E) 届时是双缓冲借用区，钉住行由
        # _compose_prefill_banks/_prefetch_split 的顶部槽 gather 服务——此守卫
        # 明确两套装配的互斥边界。
        if self.pin_ids is not None and not self.prefill_overlap:
            k = self.pin_counts[layer_id]
            if k:
                # [0, E) 是各层共享的层内 position==id 窗口，dst 即层内专家 id
                src_rows = self.pin_slots[layer_id, :k].long()
                dst_rows = self.pin_ids[layer_id, :k].long()
                for _per_layer, cache in self.banks:
                    cache[dst_rows] = cache[src_rows]

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)
        # reset 清空了全部 slot 映射（含钉住槽）：立即重钉，保持"钉住专家永远命中"不变式。
        if self.pin_ids is not None:
            self._fill_pin_maps()

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and
        the active count. The CPU computes (missing - fetched) experts. Device-side;
        accumulates both the scalar totals and the per-layer breakdown."""
        assert 0 <= layer_id < self.num_layers, f"layer_id {layer_id} out of range [0, {self.num_layers})"
        missing = self.num_missing_full.sum()
        fetched = self.num_indices.sum()
        active = self.active_mask.sum()
        self.stat_missing += missing
        self.stat_fetched += fetched
        self.stat_active += active
        self.stat_calls += 1
        self.stat_missing_layer[layer_id] += missing
        self.stat_fetched_layer[layer_id] += fetched
        self.stat_active_layer[layer_id] += active
        self.stat_steps_layer[layer_id] += 1

    def decode_miss_stats(self) -> dict:
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        return {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.cache_size / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if layer_id in self._unpinned_layers:
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            # the only copy a non-pinned layer ever needs is the non-overlap prefill materialize, which schedules the whole layer into slots [0, num_experts) with position == expert id -- a plain synchronous pageable H2D copy
            # never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)
            for per_layer, cache in self.banks:
                cache[: self.num_experts].copy_(per_layer[layer_id])
            return
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            fast_index_copy_multi_jit(
                self._copy_dst_ptrs,
                self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes,
                self.evict_slots,
                self.src_indices,
                self.num_indices,
            )
            return

        from freetoken.kernel import fast_index_copy_jit

        for per_layer, cache in self.banks:
            fast_index_copy_jit(
                cache,
                self.evict_slots,
                per_layer[layer_id],
                self.src_indices,
                self.num_indices,
            )


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
