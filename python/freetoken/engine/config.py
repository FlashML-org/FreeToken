from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, List, Mapping

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # --hf-overrides: applied to the checkpoint config the model is built from (cached_load_hf_config)
    hf_overrides: Mapping[str, Any] = field(default_factory=dict)
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch 50% of each step's misses over PCIe and compute the
    # rest on the CPU. That split is the measured decode optimum (K=32 hybrid FP8);
    # a non-negative value is an explicit per-step fetch cap instead.
    moe_hybrid_max_fetch: int = -1
    # Expert-routing hotness collection (offload family): when set, every MoE layer
    # accumulates its raw routing ids into a device-side counter and the scheduler
    # drains it to the host every hot_stats_interval_s (+ at exit), writing the
    # per-(layer, expert) histogram JSON consumed by the pin-selection tool.
    hot_stats_out: str | None = None
    # Wall-clock seconds between host-side drains of the hotness counters.
    hot_stats_interval_s: float = 60.0
    # 显存钉住热点专家：--hot-expert-list 指向 pin list JSON（选点工具
    # `python -m freetoken.hotness select -o` 的输出）；或配合 hot_expert_slots 指向
    # --hot-stats-out 格式的热度统计 JSON，由引擎在加载期内部选每层 top-K。设置后每层
    # top-K 专家在加载期单副本常驻显存 slot cache 顶部区（不参与 LRU 驱逐），host bank
    # 只装其余冷专家。两者都未设置时行为与不钉住完全一致。
    hot_expert_list: str | None = None
    # 每层钉住槽位数 K（双重语义）：配合 stats 文件时是加载期选点数（复用
    # hotness/select.py 的 select_pins，容量同值 = 静态钉住）；配合 pin list JSON
    # 时是钉住容量 K_cap（list 定初始 K，容量为运行中调 K 预留扩容空间）。
    # 未设置时 hot_expert_list 必须已是 pin list JSON，容量 = 初始 K。
    hot_expert_slots: int | None = None
    # 加载期实际钉住的前缀长度。设置后 pin list 只取前 active_k 个作为初始钉住集
    # （宿主冷 bank 的下界），其余排名留在目录里供 pin_k 按原序扩容；EMA 换血关闭。
    # None = 整张 pin list 都是初始钉住集（原行为）。
    hot_expert_active_k: int | None = None
    # LRU 区地板（槽数）。1（默认）允许钉住容量把 LRU 收到 1 槽；空着的容量槽
    # 仍算 LRU。0 = 内置地板 max(2×专家数, 512)。
    hot_expert_lru_floor: int = 1
    # 动态重钉（设计 §10）：滑动窗口热度驱动的运行期重钉，钉住模式下按墙钟间隔
    # （秒）把 device 热度计数 D2H 成"当前窗口"，经 EMA（半衰期 = 窗口时长）平滑后
    # 与当前钉住集比较，仅在 idle 安全点做行级交换。60（默认）= 钉住模式下每 60s
    # 封一个热度窗口并允许 EMA 换血；0 = 关闭。未配置钉住表时此值不生效。
    hot_expert_repin_interval_s: float = 60.0
    # 重钉迟滞：候选专家的 EMA 计数 ≥ 被替换钉住专家 × gain 才交换（防抖，>1）。
    hot_expert_repin_gain: float = 1.5
    # 每周期每层最多交换的专家对数。
    hot_expert_repin_max_swaps: int = 8
    # 运行中调参文件（--tune-file）：指向 JSON 文件，引擎守护线程每 2s 检查 mtime，
    # 变化则应用其中的键："fetch_fraction"（hybrid 每步 PCIe 拉取比例，[0,1]；经
    # cache.set_fetch_params 写设备张量，对已捕获 decode 图立即生效，cap 不变）；
    # "pin_k"（整数，每层活跃钉住数 K 的全局目标：经动态重钉管理器记录最新值，
    # 下一 idle 安全点在容量区内扩缩落地）。非法 JSON / mtime 未变静默跳过；越界值
    # 忽略并告警一次。None = 关闭。
    tune_file: str | None = None
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path, self.hf_overrides)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        return replace(model_config, quant=quant)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.table_positions

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
