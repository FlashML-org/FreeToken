"""显存钉住热点专家的加载期装配：pin list/stats 读取、校验与 cold_row 构建。

本模块是选点工具（``freetoken.hotness.select``，纯 Python）与运行期
``OffloadMoeCache`` 之间的契约层：

* ``resolve_hot_pin_plan`` 读取 ``--hot-expert-list``（pin list JSON，或配合
  ``--hot-expert-slots`` 的热度统计 JSON）并产出每层钉住专家列表与
  ``cold_row [L, E] int32`` 冷行映射；
* CPU 解码层（``--moe-cpu-layers`` 命中的层）的钉住项在此剥离并告警——这些层的
  host bank 保持全量 ``[E]``（CPU executor 按原始专家 id 取行），只有 GPU 层参与
  冷压缩。

纯几何/纯 Python（torch 仅用于 cold_row 张量），可在无 GPU 环境单测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# pin list 文件 schema 与选点工具共用一个版本常量（单一事实来源）
from freetoken.hotness.select import SCHEMA_VERSION  # noqa: E402,F401

__all__ = ["SCHEMA_VERSION", "HotPinPlan", "resolve_hot_pin_plan", "cold_row_from_pins"]


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
        两者都未设置返回 None（不钉住）；hot_expert_slots 设置时把 hot_expert_list
        按热度统计 JSON 解读（复用 hotness.select.load_stats 校验 + select_pins 选
        每层 top-K），否则按 pin list JSON 解读。随后剥离 cpu_layer_ids 命中层的
        钉住项（告警一次，列出层与专家数），全部层为空时也返回 None（无可钉项）。
        文件校验失败抛 ValueError。
    """
    if hot_expert_list is None and hot_expert_slots is None:
        return None
    if hot_expert_slots is not None:
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
