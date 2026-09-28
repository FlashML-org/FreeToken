"""显存钉住热点专家的加载期装配与运行期动态重钉：pin list/stats 读取、校验、
cold_row 构建、重钉决策、运行中调 K（pin_k）与 idle 安全点迁移管理。

本模块是选点工具（``freetoken.hotness.select``，纯 Python）与运行期
``OffloadMoeCache`` 之间的契约层：

* ``resolve_hot_pin_plan`` 读取 ``--hot-expert-list``（pin list JSON，或配合
  ``--hot-expert-slots`` 的热度统计 JSON）并产出每层钉住专家列表与
  ``cold_row [L, E] int32`` 冷行映射；pin list + ``--hot-expert-slots`` 同传时
  slots 是钉住容量 K_cap（运行中调 K 的扩容上限）；
* CPU 解码层（``--moe-cpu-layers`` 命中的层）的钉住项在此剥离并告警——这些层的
  host bank 保持全量 ``[E]``（CPU executor 按原始专家 id 取行），只有 GPU 层参与
  冷压缩；
* ``plan_repin_swaps``（纯宿主计算）比较滑动窗口 EMA 热度与当前钉住集，产出
  迟滞过滤后的交换对；
* ``HotExpertRepinManager`` 在 idle 安全点（scheduler 的 ``_execute_pending_rebuild``
  同位置）周期性触发决策并经 ``OffloadMoeCache.swap_pinned_experts`` 执行行级迁移
  （设计文档 §10）；``set_target_k``/``apply_target_k`` 承接 --tune-file 的
  ``pin_k``：写入只记最新目标，idle 安全点在容量区内扩/缩每层活跃钉住数。

纯几何/纯 Python（torch 仅用于 cold_row 张量与管理器的同步栅栏），可在无 GPU 环境单测。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.moe.hotness import ExpertHotness
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)

# pin list 文件 schema 与选点工具共用一个版本常量（单一事实来源）
from freetoken.hotness.select import SCHEMA_VERSION  # noqa: E402,F401

__all__ = [
    "SCHEMA_VERSION",
    "HotPinPlan",
    "HotExpertRepinManager",
    "cold_row_from_pins",
    "plan_repin_swaps",
    "resolve_hot_pin_plan",
]


@dataclass(frozen=True)
class HotPinPlan:
    """一份可执行的钉住计划（已剥离 CPU 解码层）。

    ``pins`` 为 ``[num_layers][K]`` 的每层钉住专家 id（层内按 pin list 顺序，即
    热度降序）；未参与钉住的层（CPU 解码层或该层无钉住项）为空列表。``K_max``
    是各层长度的最大值（0 表示没有任何钉住项，调用方应放弃钉住）。
    """

    pins: list[list[int]]
    num_experts: int
    skipped: dict[int, list[int]] = field(default_factory=dict)  # layer -> 被剥离的钉住项

    @property
    def k_max(self) -> int:
        """各层钉住数的最大值（0 = 计划为空）。"""
        return max((len(p) for p in self.pins), default=0)


def _load_pin_list(path: str, num_layers: int, num_experts: int) -> list[list[int]]:
    """
    Business Logic（为什么需要这个函数）:
        pin list JSON 是选点工具与引擎之间的磁盘契约；引擎在分配任何显存/内存之前
        必须确信文件结构与模型几何一致，否则静默钉错专家比直接报错更糟。

    Code Logic（这个函数做什么）:
        读取并校验 pin list JSON：schema_version 必须为 1；num_layers/num_experts
        必须与当前模型一致；per_layer_slots 为正整数且每层 experts 长度与之相等、
        专家 id 在 [0, num_experts) 内且层内不重复。校验失败抛 ValueError；成功
        返回 [num_layers][K] 的专家 id 列表（保持 pin list 顺序）。
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"pin list 文件顶层必须是 JSON object，实际为 {type(payload).__name__}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"pin list schema_version 必须为 {SCHEMA_VERSION}，实际为 {payload.get('schema_version')!r}"
        )
    if "pins" not in payload and "counts" in payload:
        raise ValueError(
            f"{path} 是热度统计文件（含 counts），不是 pin list；要直接吃统计文件请在 "
            "--hot-expert-list 的同时传 --hot-expert-slots K"
        )
    if payload.get("num_layers") != num_layers or payload.get("num_experts") != num_experts:
        raise ValueError(
            f"pin list 几何 ({payload.get('num_layers')} 层 x {payload.get('num_experts')} 专家) "
            f"与模型 ({num_layers} 层 x {num_experts} 专家) 不一致: {path}"
        )
    slots = payload.get("per_layer_slots")
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        raise ValueError(f"pin list per_layer_slots 必须是正整数，实际为 {slots!r}")
    pins = payload.get("pins")
    if not isinstance(pins, list) or len(pins) != num_layers:
        raise ValueError(f"pin list pins 必须是 {num_layers} 个条目的 list: {path}")
    result: list[list[int]] = []
    for entry in pins:
        if not isinstance(entry, dict) or not {"layer", "experts"} <= set(entry):
            raise ValueError(f"pin list 每项须含 layer/experts 字段: {entry!r}")
        layer_id = entry["layer"]
        experts = entry["experts"]
        if not isinstance(layer_id, int) or not 0 <= layer_id < num_layers:
            raise ValueError(f"pin list layer 越界: {layer_id!r}")
        if not isinstance(experts, list) or len(experts) != slots:
            raise ValueError(
                f"pin list layer {layer_id} 的 experts 长度必须为 per_layer_slots={slots}，"
                f"实际为 {experts if not isinstance(experts, list) else len(experts)}"
            )
        if any(isinstance(e, bool) or not isinstance(e, int) or not 0 <= e < num_experts for e in experts):
            raise ValueError(f"pin list layer {layer_id} 含越界/非法专家 id: {experts}")
        if len(set(experts)) != len(experts):
            raise ValueError(f"pin list layer {layer_id} 内专家 id 重复: {experts}")
        result.append(list(experts))
    return result


def cold_row_from_pins(pins: list[list[int]], num_experts: int) -> torch.Tensor:
    """
    Business Logic（为什么需要这个函数）:
        冷压缩 host bank 的行号语义（行 = 按专家 id 升序压缩后的冷专家序号）是
        hybrid kernel remap、materialize 计划压缩与 prefill 组装的共同前提；
        cold_row 表是这一语义的唯一权威编码，构建处必须只有一处。

    Code Logic（这个函数做什么）:
        输入 [num_layers][K] 的钉住专家 id，返回 [num_layers, num_experts] int32
        张量（CPU）：pinned 专家 -> -1；冷专家 -> 其冷行号（在该层所有冷专家中按
        id 升序的序号）。无钉住项的层为恒等映射 arange(num_experts)。
    """
    cold_rows = []
    for layer_pins in pins:
        pinned = set(layer_pins)
        if not pinned:
            cold_rows.append(torch.arange(num_experts, dtype=torch.int32))
            continue
        cold_ids = [e for e in range(num_experts) if e not in pinned]
        row = torch.full((num_experts,), -1, dtype=torch.int32)
        row[torch.tensor(cold_ids, dtype=torch.long)] = torch.arange(len(cold_ids), dtype=torch.int32)
        cold_rows.append(row)
    if not cold_rows:
        return torch.zeros((0, num_experts), dtype=torch.int32)
    return torch.stack(cold_rows)


def _looks_like_pin_list(path: str) -> bool:
    """
    Business Logic（为什么需要这个函数）:
        --hot-expert-list 与 --hot-expert-slots 同传时有两种合法语义：stats 文件
        （slots = 加载期选点数 = 容量）与 pin list 文件（slots = 钉住容量 K_cap，
        为运行中调 K 预留扩容空间）；必须先按文件种类分派，否则 pin list 会被
        误当 stats 解析报错。

    Code Logic（这个函数做什么）:
        读文件顶层 JSON（IO/解析失败按"不是 pin list"处理，交由后续校验报错），
        含 "pins" 键即 pin list。
    """
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and "pins" in doc


def resolve_hot_pin_plan(
    hot_expert_list: str | None,
    hot_expert_slots: int | None,
    num_layers: int,
    num_experts: int,
    cpu_layer_ids: frozenset[int] = frozenset(),
) -> HotPinPlan | None:
    """
    Business Logic（为什么需要这个函数）:
        引擎在加载专家 bank 之前需要一份已校验、已剥离 CPU 解码层的钉住计划；
        pin list / stats 两种入口、几何校验与 CPU 层剥离都收敛在这一处，engine
        只消费结果，避免装配逻辑散落在加载路径里。

    Code Logic（这个函数做什么）:
        两者都未设置返回 None（不钉住）。hot_expert_slots 与 pin list 同传时 slots
        是钉住容量 K_cap（运行中调 K 的扩容上限，须 >= pin list 每层钉住数）；与
        热度统计 JSON 同传时 slots 是加载期选点数（复用 hotness.select.load_stats
        校验 + select_pins 选每层 top-K，容量同值 = 静态）。随后剥离 cpu_layer_ids
        命中层的钉住项（告警一次，列出层与专家数），全部层为空时也返回 None（无可
        钉项）。文件校验失败抛 ValueError。
    """
    if hot_expert_list is None and hot_expert_slots is None:
        return None
    pins: list[list[int]]
    pin_list_capacity: int | None = None
    if (
        hot_expert_list is not None
        and hot_expert_slots is not None
        and _looks_like_pin_list(hot_expert_list)
    ):
        # pin list + slots：slots = 钉住容量（初始 K 取 pin list，扩容空间由容量预留）
        if hot_expert_slots < 1:
            raise ValueError(f"--hot-expert-slots 必须 >= 1，实际为 {hot_expert_slots}")
        pin_list_capacity = hot_expert_slots
        pins = _load_pin_list(hot_expert_list, num_layers, num_experts)
    elif hot_expert_slots is not None:
        if hot_expert_slots < 1:
            raise ValueError(f"--hot-expert-slots 必须 >= 1，实际为 {hot_expert_slots}")
        if hot_expert_list is None:
            raise ValueError("--hot-expert-slots 需要配合 --hot-expert-list 指向热度统计 JSON")
        from freetoken.hotness.select import load_stats, select_pins

        stats = load_stats(hot_expert_list)
        meta = stats["meta"]
        if meta["num_layers"] != num_layers or meta["num_experts"] != num_experts:
            raise ValueError(
                f"热度统计几何 ({meta['num_layers']} 层 x {meta['num_experts']} 专家) 与模型 "
                f"({num_layers} 层 x {num_experts} 专家) 不一致: {hot_expert_list}"
            )
        slots = min(hot_expert_slots, num_experts)
        if slots != hot_expert_slots:
            logger.warning_rank0(
                f"--hot-expert-slots {hot_expert_slots} 超过专家数 {num_experts}，已截断为 {slots}"
            )
        pins = select_pins(stats["counts"], slots)
    else:
        pins = _load_pin_list(hot_expert_list or "", num_layers, num_experts)

    if pin_list_capacity is not None:
        widest = max((len(p) for p in pins), default=0)
        if pin_list_capacity < widest:
            raise ValueError(
                f"--hot-expert-slots {pin_list_capacity} 小于 pin list 每层钉住数 {widest}："
                "容量区必须装得下初始钉住集"
            )

    skipped: dict[int, list[int]] = {}
    if cpu_layer_ids:
        for layer_id in sorted(cpu_layer_ids):
            if pins[layer_id]:
                skipped[layer_id] = pins[layer_id]
                pins[layer_id] = []
        if skipped:
            logger.warning_rank0(
                f"--hot-expert-list: CPU 解码层 {sorted(skipped)} 的 "
                f"{sum(len(v) for v in skipped.values())} 个钉住项被跳过 "
                "(CPU executor 直接读全量 host bank，冷压缩只作用于 GPU 解码层)"
            )
    if all(not p for p in pins):
        logger.warning_rank0("--hot-expert-list: 没有任何可钉住的层，本次启动不钉住")
        return None
    return HotPinPlan(pins=pins, num_experts=num_experts, skipped=skipped)


def plan_repin_swaps(
    ema: np.ndarray,
    pinned: list[list[int]],
    *,
    gain: float,
    max_swaps: int,
) -> dict[int, list[tuple[int, int]]]:
    """
    Business Logic（为什么需要这个函数）:
        动态重钉的核心判断必须与执行解耦：决策是纯宿主计算（可单测、可解释），
        才能把"何时换、换谁"的迟滞规则与"如何搬字节"的迁移细节分开演进，也才能
        在不触碰任何张量的情况下验证防抖语义。

    Code Logic（这个函数做什么）:
        逐层比较 EMA 热度 top-K 与当前钉住集：候选 = EMA 前 K 名中不在钉住集的
        冷专家（按 EMA 降序、平局小 id）；受害 = 钉住集中 EMA 计数最低者（升序、
        平局小 id）。仅当 EMA(候选) >= EMA(受害) × gain 才交换（迟滞防抖）——候选
        与受害各自按序推进，首个不满足即终止（后续候选更冷、受害不会更冷，必然
        全不满足）；EMA 为 0 的候选不参与（零热度不构成"更热"证据）。每层最多
        max_swaps 对。返回 {layer_id: [(候选, 被替换), ...]}，无迁移的层不出现。
    """
    if gain < 1.0:
        raise ValueError(f"gain 必须 >= 1.0（迟滞下限），实际为 {gain}")
    if max_swaps < 1:
        raise ValueError(f"max_swaps 必须 >= 1，实际为 {max_swaps}")
    if ema.ndim != 2 or ema.shape[0] != len(pinned):
        raise ValueError(f"ema 形状 {ema.shape} 与钉住层数 {len(pinned)} 不一致")
    plan: dict[int, list[tuple[int, int]]] = {}
    for layer_id, pins in enumerate(pinned):
        count = len(pins)
        if count == 0:
            continue
        row = ema[layer_id]
        num_experts = row.shape[0]
        # 全序：EMA 降序、平局取小 id（与选点工具 select_pins 同一规则）
        order = sorted(range(num_experts), key=lambda e: (-row[e], e))
        # 候选：EMA 前 K 名中的当前冷专家（不在钉住集 ⟺ 在冷集）
        candidates = [e for e in order[:count] if e not in set(pins)]
        # 受害排序：钉住专家按 EMA 升序、平局小 id（最冷者优先被替换）
        victims = sorted(pins, key=lambda e: (row[e], e))
        swaps: list[tuple[int, int]] = []
        vi = 0
        for cand in candidates:
            if len(swaps) >= max_swaps or vi >= len(victims):
                break
            cand_ema = float(row[cand])
            if cand_ema <= 0.0:
                break  # 候选按 EMA 降序，其后必然同样为零热度
            victim = victims[vi]
            if cand_ema < float(row[victim]) * gain:
                break  # 迟滞未过线：后续候选更冷、受害不变，必然同样不过线
            swaps.append((cand, victim))
            vi += 1
        if swaps:
            plan[layer_id] = swaps
    return plan


class HotExpertRepinManager:
    """动态重钉管理器：周期触发窗口决策并在 idle 安全点执行行级迁移（设计 §10）；
    兼管运行中调 K（``set_target_k``/``apply_target_k``，容量区内扩缩每层活跃钉住数）。

    由 engine 在钉住装配完成时挂到 ``OffloadMoeCache.repin_manager``；scheduler 在
    ``_execute_pending_rebuild`` 同一个 idle 安全点调用 :meth:`maybe_repin`（内部
    先执行未落地的目标 K，再做常规 EMA 换血）。
    """

    def __init__(
        self,
        cache: OffloadMoeCache,
        hotness: ExpertHotness,
        *,
        interval_s: float,
        gain: float,
        max_swaps: int,
    ) -> None:
        """
        Business Logic（为什么需要这个函数）:
            重钉的触发节奏、迟滞与上限是用户可调的三个独立旋钮；把它们与 cache/
            hotness 的绑定收敛到一个管理器对象，scheduler 只需要一次 getattr 触发，
            不感知决策与迁移细节。

        Code Logic（这个函数做什么）:
            校验参数（interval_s > 0、gain >= 1.0、max_swaps >= 1）后保存 cache、
            窗口计数器与三旋钮；初始化上次触发墙钟（首个窗口需 interval_s 填充，
            之前的触发会被 has_window 门挡住）；目标 K 置空（pin_k 未写入）。
        """
        if interval_s <= 0:
            raise ValueError(f"--hot-expert-repin-interval-s 必须 > 0，实际为 {interval_s}")
        if gain < 1.0:
            raise ValueError(f"--hot-expert-repin-gain 必须 >= 1.0（迟滞下限），实际为 {gain}")
        if max_swaps < 1:
            raise ValueError(f"--hot-expert-repin-max-swaps 必须 >= 1，实际为 {max_swaps}")
        self._cache = cache
        self._hotness = hotness
        self._interval_s = float(interval_s)
        self._gain = float(gain)
        self._max_swaps = int(max_swaps)
        self._last = time.monotonic()
        # 运行中调 K 的最新目标（set_target_k 写入，apply_target_k 消费后清空；
        # None = 无待落地目标，apply_target_k 在 idle 点的每次轮询直接快速返回）
        self._target_k: int | None = None
        # 扩容等待原因（无窗口 / 零热度候选）的限流日志标记：同类原因只打一次，
        # 有实际进展后复位，避免每个 idle 安全点刷屏又保留"为什么还没扩"的可观测性
        self._defer_logged: bool = False

    @property
    def target_k(self) -> int | None:
        """当前待落地的目标 K（None = 无；测试与状态查询用）。"""
        return self._target_k

    def set_target_k(self, k: int) -> None:
        """设置全局目标钉住数 K（各层同值，按层 clamp 到 [floors[l], K_cap]）。

        Business Logic（为什么需要这个函数）:
            运行中调 K（--tune-file 的 pin_k）需要与常规换血解耦的入口：写入只记录
            "最新意图"（写多少次只保留最新值），真正的字节迁移推迟到下一个 idle
            安全点；越界目标在这里就地拒绝，poller 侧转为告警，不让坏值进入执行。

        Code Logic（这个函数做什么）:
            校验 0 <= k <= cache.pin_capacity（容量由 --hot-expert-slots 或初始钉住
            数决定；未装配钉住时抛 RuntimeError）后覆写 _target_k。应用是异步的：
            apply_target_k 在 idle 安全点执行。
        """
        cap = self._cache.pin_capacity
        if cap is None:
            raise RuntimeError("cache 未装配钉住（init_hot_pins），无法设置 pin_k 目标")
        if not 0 <= k <= cap:
            raise ValueError(f"pin_k 目标 {k} 超出 [0, {cap}]（容量 = --hot-expert-slots 或初始钉住数）")
        self._target_k = int(k)

    def _growth_candidates(self, layer_id: int, want: int, pinned: set[int], ema: np.ndarray) -> list[int]:
        """
        Business Logic（为什么需要这个函数）:
            扩容选点必须与选点工具/常规换血同一热度规则（EMA 降序、平局小 id），
            且零热度候选不钉（零热度不构成"更热"证据，钉住只浪费容量槽）。

        Code Logic（这个函数做什么）:
            按全序遍历该层 EMA，跳过已在钉住集与零热度专家，取前 want 个尚未钉住
            的专家 id（可能不足 want：窗口热度不足时扩多少算多少，余量留给下轮）。
        """
        row = ema[layer_id]
        order = sorted(range(self._cache.num_experts), key=lambda e: (-row[e], e))
        out: list[int] = []
        for e in order:
            if len(out) >= want:
                break
            if e in pinned or float(row[e]) <= 0.0:
                continue
            out.append(e)
        return out

    def apply_target_k(self) -> bool:
        """
        Business Logic（为什么需要这个函数）:
            pin_k 的应用必须是 idle 安全点的同步操作（改写钉住槽字节与映射，不能与
            在途读者并发），且要在常规换血之前完成——换血决策按当前钉住集计算，
            扩缩落地后换血才不会重复迁移刚装入/换出的专家。

        Code Logic（这个函数做什么）:
            无待落地目标或缺钉住状态直接返回 False。每层目标 = clamp(k, floors[l],
            K_cap)（floors = 加载期钉住数：宿主冷压缩 bank 行数决定缩 K 下界；CPU
            解码层 floors = 0 自然保持不钉）。缩容目标传绝对值给 unpin_tail_experts；
            扩容 want 必须是 min(目标差值, K_cap - counts)（"装入个数"语义——2026-09-28
            真实 FP8 服务事故：grows 误存绝对目标，K=32/K_cap=61 收 pin_k=40 时以
            want=40 选点，install 在 32+29>=61 处每次拒绝第 30 个候选，目标因异常
            保留、每个 idle 原样重试，扩容永不收敛）。扩容选点热度源：窗口 EMA 已
            封口用 EMA；未封口退回全量累计热度（首批流量排空即有，不等待窗口）。
            扩容候选全无（热度全零）时目标保留、静默返回，等下一个 idle 安全点。
            否则 torch.cuda.synchronize 做安全栅栏，逐层先缩（尾部换出）后扩
            （install_pinned_experts 装入，选点见 _growth_candidates），再次
            synchronize 暴露异步错误。全部层落地（无 shortfall）才清空目标；窗口
            热度不足的余量保留目标、下个 idle 续传（幂等：counts 已更新，重算 grows
            只剩余量，不重复迁移已装入专家）。返回是否有变更。
        """
        if self._target_k is None or self._cache.pin_ids is None:
            return False
        target = self._target_k
        cache = self._cache
        floors = cache.pin_floors
        cap = cache.pin_capacity
        assert floors is not None and cap is not None
        counts = list(cache.pin_counts)
        targets = [min(max(target, floors[layer_id]), cap) for layer_id in range(cache.num_layers)]
        shrinks = {
            layer_id: targets[layer_id]
            for layer_id in range(cache.num_layers)
            if targets[layer_id] < counts[layer_id]
        }
        # 扩容量是"装入个数"（want 语义）：目标差值，且不超过剩余容量槽。shrinks
        # 的值是绝对目标数（unpin_tail_experts 的 new_count 参数即绝对值），两者
        # 语义不同，不能共用一种写法
        grows = {
            layer_id: min(targets[layer_id] - counts[layer_id], cap - counts[layer_id])
            for layer_id in range(cache.num_layers)
            if targets[layer_id] > counts[layer_id]
        }
        if not shrinks and not grows:
            # 已在目标上：无待办，消费目标（写重复 pin_k 不产生任何迁移）
            self._target_k = None
            self._defer_logged = False
            return False
        # 扩容选点的热度源：窗口 EMA 已封口用 EMA（对近期负载敏感）；未封口时退回
        # 全量累计热度（首批流量排空即有，无需等一个完整窗口）——扩容初始选点本就
        # 没有"更近"的信号可用，服务启动以来的 top 热点即最合理起点，其后由常规
        # EMA 换血继续修正。
        ema = None
        if grows:
            if self._hotness.has_window:
                ema = self._hotness.ema_counts()
            else:
                ema = self._hotness.cumulative_counts()
                if not self._defer_logged:
                    logger.info_rank0(
                        "dynamic pin-k: target K=%d growth picks from cumulative "
                        "hotness (first window not sealed yet)", target
                    )
                    self._defer_logged = True
        # 选点先于栅栏（纯宿主计算）：扩容候选全无（窗口热度不足）时目标保留、
        # 静默返回，等下一个 idle 安全点再试，不用日志刷屏
        grow_picks: dict[int, list[int]] = {}
        for layer_id in sorted(grows):
            pinned_set = set(cache.pinned_id_lists()[layer_id])
            grow_picks[layer_id] = self._growth_candidates(
                layer_id, grows[layer_id], pinned_set, ema
            )
        if not shrinks and all(not picks for picks in grow_picks.values()):
            if not self._defer_logged:
                logger.info_rank0(
                    "dynamic pin-k: target K=%d deferred -- window EMA has no growth "
                    "candidates (zero-traffic window?)", target
                )
                self._defer_logged = True
            return False
        t0 = time.perf_counter()
        if cache.device.type == "cuda":
            # idle 安全点栅栏（同 maybe_repin 的换血）：所有流同步后无在途读者
            torch.cuda.synchronize(cache.device)
        unpinned_total = 0
        pinned_total = 0
        shortfall_total = 0
        touched = 0
        details: list[str] = []
        for layer_id in sorted(set(shrinks) | set(grow_picks)):
            if layer_id in shrinks:
                out = cache.unpin_tail_experts(layer_id, shrinks[layer_id])
                unpinned_total += len(out)
                if out:
                    details.append(f"L{layer_id}: -{out} (K {counts[layer_id]}->{shrinks[layer_id]})")
            if layer_id in grow_picks:
                put = cache.install_pinned_experts(layer_id, grow_picks[layer_id])
                pinned_total += len(put)
                shortfall = grows[layer_id] - len(put)
                shortfall_total += shortfall
                note = f"，{shortfall} 个候选窗口热度不足未钉" if shortfall else ""
                details.append(
                    f"L{layer_id}: +{put} (K {counts[layer_id]}->{counts[layer_id] + len(put)}{note})"
                )
            touched += 1
        if cache.device.type == "cuda":
            # 应用后再次同步：行级拷贝是异步的，任何错误就地暴露
            torch.cuda.synchronize(cache.device)
        # 全部层落地才消费目标；窗口热度不足的余量保留目标，下个 idle 安全点续传
        # （counts 已推进，重算 grows 只剩余量——半应用状态靠幂等可续而非回滚）
        if shortfall_total == 0:
            self._target_k = None
            self._defer_logged = False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if touched:
            logger.info_rank0(
                "dynamic pin-k: target K=%d applied in %.1f ms -- %d pinned in, %d unpinned "
                "across %d layer(s): %s",
                target, elapsed_ms, pinned_total, unpinned_total, touched, "; ".join(details),
            )
        return bool(unpinned_total or pinned_total)

    def maybe_repin(self, now: float | None = None) -> bool:
        """
        Business Logic（为什么需要这个函数）:
            scheduler 的 idle 安全点每迭代都会到达，但重钉只能在"墙钟到点 && 已有
            完整热度窗口 && 确有迁移"三者同时成立时执行；执行前必须同步全部流，
            保证没有在途 GEMM/CPU GEMV/未完成 prefill chunk 读着即将改写的字节。
            目标 K（pin_k）不受墙钟与窗口门控：变更后的首个 idle 安全点即应用，
            先扩缩再换血。

        Code Logic（这个函数做什么）:
            先 apply_target_k（无目标时零开销返回 False）；墙钟未到 interval_s 直接
            返回其结果；到点即刷新（失败也不立刻重试，下一个窗口再看）。无完整
            窗口/无钉住时返回。否则取 EMA 与当前钉住集（GPU → host 一次拷贝）跑
            plan_repin_swaps；有迁移先在 CUDA 设备上 torch.cuda.synchronize 做安全
            栅栏，再逐层 swap_pinned_experts 执行，最后打一条汇总日志（层、换入/
            换出对、计数比、耗时）并返回 True。now 参数仅供测试注入时钟。
        """
        applied = self.apply_target_k()
        now = time.monotonic() if now is None else now
        if now - self._last < self._interval_s:
            return applied
        self._last = now
        if not self._hotness.has_window or self._cache.pin_ids is None:
            return applied
        ema = self._hotness.ema_counts()
        plan = plan_repin_swaps(ema, self._cache.pinned_id_lists(), gain=self._gain, max_swaps=self._max_swaps)
        if not plan:
            return applied
        t0 = time.perf_counter()
        if self._cache.device.type == "cuda":
            # idle 安全点栅栏：所有流同步后无在途 GEMM/CPU GEMV/prefill chunk，
            # 行级交换的 D2H/H2D 与 host-to-host 写回才不会与读者竞争。
            torch.cuda.synchronize(self._cache.device)
        total = 0
        for layer_id in sorted(plan):
            swaps = plan[layer_id]
            self._cache.swap_pinned_experts(layer_id, swaps)
            total += len(swaps)
        if self._cache.device.type == "cuda":
            # 交换后再次同步：行级拷贝是异步的，这里把任何交换期错误就地暴露，
            # 不让它以异步非法访问的形式归因到后续 forward。
            torch.cuda.synchronize(self._cache.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        detail = "; ".join(
            "L{}: {} -> {} (ema {:.1f} vs {:.1f}, x{:.2f})".format(
                layer_id, replaced, cand, ema[layer_id, cand], ema[layer_id, replaced],
                ema[layer_id, cand] / max(ema[layer_id, replaced], 1e-9),
            )
            for layer_id in sorted(plan)
            for cand, replaced in plan[layer_id]
        )
        logger.info_rank0(
            "dynamic repin: %d expert(s) swapped across %d layer(s) in %.1f ms: %s",
            total, len(plan), elapsed_ms, detail,
        )
        return True
