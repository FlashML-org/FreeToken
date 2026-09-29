"""热点专家选点工具：读取热度统计 JSON，按每层 top-K 选出钉住专家并输出 pin list。

纯 Python 实现，可在无 GPU 环境运行；``--budget-gib`` 折算每专家字节数时才延迟导入
``freetoken.moe.offload_cache``（其顶层会引入 torch）。本模块与运行期计数器
``freetoken.moe.hotness`` 互不依赖：stats 文件的读取与校验在此独立实现。

调用形式（均可）::

    python -m freetoken.hotness select STATS.json --slots K [-o OUT.json]
    python -m freetoken.hotness select STATS.json --budget-gib G [--format nvfp4] [-o OUT.json]
    python -m freetoken.hotness.select select STATS.json --slots K
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Any

# stats 文件与 pin list 文件共用的 schema 版本（设计文档 §3.1/§3.2 契约）
SCHEMA_VERSION = 1

# 无 -o 时 stdout 上展示的示例层数
_EXAMPLE_LAYERS = 3

__all__ = [
    "SCHEMA_VERSION",
    "load_stats",
    "select_pins",
    "slots_from_budget",
    "coverage_report",
    "write_pin_list",
    "main",
]


def load_stats(path: str) -> dict[str, Any]:
    """
    Business Logic（为什么需要这个函数）:
        采集代理（--hot-stats-out）与选点工具之间靠 stats JSON 契约解耦；
        选点侧必须在计算前确信文件结构正确，否则静默选错专家比直接报错更糟。

    Code Logic（这个函数做什么）:
        读取并校验 stats JSON：schema_version 必须为 1；meta.num_layers /
        meta.num_experts 必须为正整数且与 counts 形状一致；counts 必须是
        L×E 的嵌套 list，元素为非负 int（bool 不算）。校验失败抛 ValueError
        并携带明确信息；成功返回解析后的 dict（含 meta 与 counts）。
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f"无法读取 stats 文件 {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"stats 文件 {path!r} 不是合法 JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"stats 文件顶层必须是 JSON object，实际为 {type(data).__name__}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version 必须为 {SCHEMA_VERSION}，实际为 {data.get('schema_version')!r}"
        )

    meta = data.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("stats 缺少 meta object 字段")
    num_layers = meta.get("num_layers")
    num_experts = meta.get("num_experts")
    if not _is_positive_int(num_layers) or not _is_positive_int(num_experts):
        raise ValueError(
            f"meta.num_layers / meta.num_experts 必须是正整数，"
            f"实际为 {num_layers!r} / {num_experts!r}"
        )

    counts = data.get("counts")
    if not isinstance(counts, list):
        raise ValueError(f"counts 必须是嵌套 list，实际为 {type(counts).__name__}")
    if len(counts) != num_layers:
        raise ValueError(
            f"counts 层数 ({len(counts)}) 与 meta.num_layers ({num_layers}) 不一致"
        )
    for layer_id, row in enumerate(counts):
        if not isinstance(row, list):
            raise ValueError(f"counts[{layer_id}] 必须是 list，实际为 {type(row).__name__}")
        if len(row) != num_experts:
            raise ValueError(
                f"counts[{layer_id}] 长度 ({len(row)}) 与 meta.num_experts ({num_experts}) 不一致"
            )
        for expert_id, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"counts[{layer_id}][{expert_id}] 必须是 int，实际为 {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"counts[{layer_id}][{expert_id}] 不能为负数 ({value})")
    return data


def select_pins(counts: list[list[int]], slots_per_layer: int) -> list[list[int]]:
    """
    Business Logic（为什么需要这个函数）:
        钉住显存预算以"每层 K 个专家"计，而路由偏斜逐层不同，因此每层独立
        选 top-K；平局必须有确定结果，否则同一份 stats 在不同机器上会选点
        不同，破坏可复现性，故平局一律取小 id。

    Code Logic（这个函数做什么）:
        对每层专家按 (-count, expert_id) 升序排序（等价于计数降序、平局小 id
        优先），取前 slots_per_layer 个专家 id。slots_per_layer < 1，或大于等于
        该层专家数（冷 bank 至少要留 1 行，钉满会在建 bank 时才失败）时抛
        ValueError。返回 [num_layers][K] 的专家 id 列表。
    """
    if not counts:
        raise ValueError("counts 不能为空")
    if slots_per_layer < 1:
        raise ValueError(f"slots_per_layer 必须 >= 1，实际为 {slots_per_layer}")
    num_experts = len(counts[0])
    if any(len(row) != num_experts for row in counts):
        raise ValueError("counts 各层长度必须一致（矩形 [L][E]）")
    if slots_per_layer >= num_experts:
        raise ValueError(
            f"slots_per_layer={slots_per_layer} 必须小于专家数 {num_experts}（冷 bank 至少 1 行）"
        )

    pins: list[list[int]] = []
    for row in counts:
        order = sorted(range(num_experts), key=lambda e: (-row[e], e))
        pins.append(order[:slots_per_layer])
    return pins


def slots_from_budget(total_budget_gib: float, bytes_per_expert: int, num_layers: int) -> int:
    """
    Business Logic（为什么需要这个函数）:
        用户更愿意按"我愿意花多少 GiB 显存"来指定钉住规模，而不是手算每层
        槽位数；工具必须给出不超预算的安全 K（向下取整），且至少钉 1 个，
        避免预算过小时静默退化为"不钉"。

    Code Logic（这个函数做什么）:
        计算 floor(total_budget_gib * 2**30 / bytes_per_expert / num_layers)
        作为每层可钉槽位数；结果至少为 1。参数非法（预算 <= 0、每专家字节
        < 1、层数 < 1）抛 ValueError。
    """
    if total_budget_gib <= 0:
        raise ValueError(f"total_budget_gib 必须为正数，实际为 {total_budget_gib}")
    if bytes_per_expert < 1:
        raise ValueError(f"bytes_per_expert 必须 >= 1，实际为 {bytes_per_expert}")
    if num_layers < 1:
        raise ValueError(f"num_layers 必须 >= 1，实际为 {num_layers}")
    slots = math.floor(total_budget_gib * 2**30 / bytes_per_expert / num_layers)
    return max(1, slots)


def coverage_report(counts: list[list[int]], pins: list[list[int]]) -> str:
    """
    Business Logic（为什么需要这个函数）:
        K 选多大是人工权衡：K 太小盖不住流量，太大挤占 LRU 区。用户需要
        看到覆盖率随当前 K 的实际数值（每层质量占比、全局加权命中率、零
        命中专家量）来判断 K 是否合理、是否值得调。

    Code Logic（这个函数做什么）:
        输入 counts [L][E] 与选点结果 pins [L][K]，返回多行文本：逐层列出
        "被选中专家计数和 / 该层总计数" 的百分比（层总计数为 0 时显示 n/a，
        不伪造 0%），随后是全局加权命中率估计（所有层选中计数总和 / 全部
        计数总和）与零命中专家数（count == 0 的专家个数 / 总专家数）。
    """
    if len(counts) != len(pins):
        raise ValueError(
            f"counts 层数 ({len(counts)}) 与 pins 层数 ({len(pins)}) 不一致"
        )
    lines: list[str] = ["每层 top-K 质量占比（选中专家计数和 / 该层总计数）:"]
    total_all = 0
    pinned_all = 0
    zero_hit = 0
    for layer_id, (row, chosen) in enumerate(zip(counts, pins)):
        layer_total = sum(row)
        layer_pinned = sum(row[e] for e in chosen)
        total_all += layer_total
        pinned_all += layer_pinned
        zero_hit += sum(1 for v in row if v == 0)
        share = f"{layer_pinned / layer_total * 100:.2f}%" if layer_total > 0 else "n/a"
        lines.append(f"  layer {layer_id:>4}: {share}")
    global_share = f"{pinned_all / total_all * 100:.2f}%" if total_all > 0 else "n/a"
    num_experts = len(counts[0]) if counts else 0
    lines.append(
        f"全局加权命中率估计: {global_share}（选中计数 {pinned_all} / 总计数 {total_all}）"
    )
    lines.append(f"零命中专家数: {zero_hit} / {len(counts) * num_experts}")
    return "\n".join(lines)


def write_pin_list(
    path: str,
    num_layers: int,
    num_experts: int,
    slots_per_layer: int,
    pins: list[list[int]],
) -> None:
    """
    Business Logic（为什么需要这个函数）:
        选点结果是加载期钉住（engine 侧）的输入契约，必须以设计文档 §3.2
        的 pin list JSON 固化到盘；引擎与选点可能先后崩溃，因此写入必须
        原子（tmp + os.replace），避免留下半截文件被引擎误读。

    Code Logic（这个函数做什么）:
        校验 slots_per_layer 小于专家数（冷 bank 至少 1 行），pins 为
        [num_layers][slots_per_layer] 的矩形、每层专家 id 在 [0, num_experts)
        内且不重复，然后组装 {"schema_version": 1,
        "num_layers", "num_experts", "per_layer_slots", "pins": [{"layer",
        "experts"}, ...]}，先写同目录临时文件再 os.replace 原子落盘。
        校验失败抛 ValueError。
    """
    if slots_per_layer >= num_experts:
        raise ValueError(
            f"slots_per_layer={slots_per_layer} 必须小于专家数 {num_experts}（冷 bank 至少 1 行）"
        )
    if len(pins) != num_layers:
        raise ValueError(f"pins 层数 ({len(pins)}) 与 num_layers ({num_layers}) 不一致")
    for layer_id, experts in enumerate(pins):
        if len(experts) != slots_per_layer:
            raise ValueError(
                f"pins[{layer_id}] 长度 ({len(experts)}) 与 slots_per_layer "
                f"({slots_per_layer}) 不一致"
            )
        if len(set(experts)) != len(experts):
            raise ValueError(f"pins[{layer_id}] 内专家 id 重复: {experts}")
        for expert in experts:
            if not 0 <= expert < num_experts:
                raise ValueError(
                    f"pins[{layer_id}] 含越界专家 id {expert}（num_experts={num_experts}）"
                )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "per_layer_slots": slots_per_layer,
        "pins": [
            {"layer": layer_id, "experts": list(experts)}
            for layer_id, experts in enumerate(pins)
        ],
    }
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".pin_list_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _is_positive_int(value: Any) -> bool:
    """
    Business Logic（为什么需要这个函数）:
        meta 里的层数/专家数决定后续所有形状校验，必须排除 bool（int 的
        子类）与非正数，否则 True 会被当成 1 蒙混过关。

    Code Logic（这个函数做什么）:
        当且仅当 value 是非 bool 的 int 且 > 0 时返回 True。
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _read_model_config_dims(model_path: str) -> tuple[int, int] | None:
    """
    Business Logic（为什么需要这个函数）:
        --budget-gib 折算每专家字节数需要 (hidden_size, moe_intermediate)，
        而 stats meta 里只有 model_path；从该目录的 config.json 直接读取是
        唯一免手填的来源，读不到时交由调用方要求显式传参。

    Code Logic（这个函数做什么）:
        model_path 是含 config.json 的本地目录时读取并返回
        (hidden_size, moe_intermediate_size 或 intermediate_size)；任何一步
        不可得（目录不存在、文件缺失、字段非法）返回 None，不抛异常。
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    hidden = config.get("hidden_size")
    intermediate = config.get("moe_intermediate_size", config.get("intermediate_size"))
    if not _is_positive_int(hidden) or not _is_positive_int(intermediate):
        return None
    return hidden, intermediate


def _resolve_bytes_per_expert(
    fmt: str,
    hidden_size: int | None,
    moe_intermediate: int | None,
    model_path: str,
) -> int:
    """
    Business Logic（为什么需要这个函数）:
        预算折算的唯一权威是 offload_cache._BANK_BYTES_PER_EXPERT（bank
        布局声明处），但它按 (hidden, intermediate) 参数化；本函数把"格式
        标签 + 维度来源"收敛为一处，维度优先 CLI 显式传参、其次模型
        config.json，都不满足时报可操作的错误。

    Code Logic（这个函数做什么）:
        延迟导入 freetoken.moe.offload_cache（其顶层引入 torch，保持选点
        工具在 --slots 路径下零 torch 依赖），按 fmt 查表（未命中则报出可
        选值），补齐维度后调用表中 lambda 得到每专家字节数并返回。
    """
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT  # 延迟导入：顶层会引入 torch

    per_expert = _BANK_BYTES_PER_EXPERT.get(fmt)
    if per_expert is None:
        raise ValueError(
            f"--format {fmt!r} 不在 _BANK_BYTES_PER_EXPERT 中，可选: {sorted(_BANK_BYTES_PER_EXPERT)}"
        )
    if hidden_size is None or moe_intermediate is None:
        dims = _read_model_config_dims(model_path)
        if dims is not None:
            if hidden_size is None:
                hidden_size = dims[0]
            if moe_intermediate is None:
                moe_intermediate = dims[1]
    if hidden_size is None or moe_intermediate is None:
        raise ValueError(
            "--budget-gib 需要 (hidden_size, moe_intermediate) 才能折算每专家字节数："
            "请显式传 --hidden-size/--moe-intermediate，或确保 stats meta.model_path "
            "指向含合法 config.json 的本地模型目录"
        )
    return int(per_expert(hidden_size, moe_intermediate))


def _build_select_parser() -> argparse.ArgumentParser:
    """
    Business Logic（为什么需要这个函数）:
        argparse 对象只在 main 内使用，独立成函数便于测试与 `python -m
        freetoken.hotness` / `...select` 两种入口复用同一套参数定义。

    Code Logic（这个函数做什么）:
        构造带 `select` 子命令的 ArgumentParser：位置参数 stats；--slots 与
        --budget-gib 互斥且必选其一；--format 默认 nvfp4；--hidden-size /
        --moe-intermediate 供预算折算显式补维度；-o 指定 pin list 输出路径。
    """
    parser = argparse.ArgumentParser(
        prog="python -m freetoken.hotness",
        description="热点专家选点工具：从热度统计 JSON 选出每层 top-K 专家",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    select_parser = subparsers.add_parser(
        "select", help="选每层 top-K 专家，打印覆盖率报告，可选写 pin list"
    )
    select_parser.add_argument("stats", help="热度统计 JSON 路径（--hot-stats-out 产出）")
    group = select_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--slots", type=int, metavar="K", help="每层钉住的专家数 K")
    group.add_argument(
        "--budget-gib", type=float, metavar="G", help="钉住显存总预算（GiB），按每专家字节数折算每层 K"
    )
    select_parser.add_argument(
        "--format",
        default="nvfp4",
        metavar="FMT",
        help="预算折算用的量化格式标签（_BANK_BYTES_PER_EXPERT 键，默认 nvfp4）",
    )
    select_parser.add_argument(
        "--hidden-size", type=int, default=None, help="预算折算用 hidden_size；缺省时从模型 config.json 读"
    )
    select_parser.add_argument(
        "--moe-intermediate",
        type=int,
        default=None,
        help="预算折算用 moe_intermediate；缺省时从模型 config.json 读",
    )
    select_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="pin list JSON 输出路径；缺省只打印前几层选点示例",
    )
    return parser


def _run_select(args: argparse.Namespace) -> int:
    """
    Business Logic（为什么需要这个函数）:
        `select` 子命令的执行体：把"读 stats → 定 K → 选点 → 报告 → 落盘"
        串成一条用户可见的流水线，错误统一以 ValueError 抛给 main 转退出码。

    Code Logic（这个函数做什么）:
        load_stats 校验读取；K 来自 --slots 或由 --budget-gib 经
        _BANK_BYTES_PER_EXPERT 折算。K 大于等于专家数直接拒绝（冷 bank 至少
        1 行），不再截成全量。select_pins 选点后向 stdout 打印头部信息与
        coverage_report；给了 -o 就 write_pin_list 落盘，否则打印前几层选点
        示例。成功返回 0。
    """
    stats = load_stats(args.stats)
    meta = stats["meta"]
    num_layers: int = meta["num_layers"]
    num_experts: int = meta["num_experts"]
    counts: list[list[int]] = stats["counts"]

    if args.slots is not None:
        slots_per_layer = args.slots
        source = f"--slots {slots_per_layer}"
    else:
        bytes_per_expert = _resolve_bytes_per_expert(
            args.format,
            args.hidden_size,
            args.moe_intermediate,
            meta.get("model_path", "") or "",
        )
        slots_per_layer = slots_from_budget(args.budget_gib, bytes_per_expert, num_layers)
        source = (
            f"--budget-gib {args.budget_gib} GiB × {args.format} "
            f"({bytes_per_expert} bytes/专家) / {num_layers} 层"
        )

    if slots_per_layer >= num_experts:
        raise ValueError(
            f"每层 K={slots_per_layer} 必须小于专家数 {num_experts}（冷 bank 至少 1 行）"
        )

    pins = select_pins(counts, slots_per_layer)
    print(f"stats: {args.stats}（layers={num_layers}, experts={num_experts}）")
    print(f"每层钉住 K={slots_per_layer}（来源: {source}）")
    print(coverage_report(counts, pins))

    if args.output:
        write_pin_list(args.output, num_layers, num_experts, slots_per_layer, pins)
        print(f"已写出 pin list: {args.output}")
    else:
        shown = pins[:_EXAMPLE_LAYERS]
        print(f"示例（前 {len(shown)} 层选点，未写文件；需要请加 -o OUT.json）:")
        for layer_id, experts in enumerate(shown):
            print(f"  layer {layer_id}: {experts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Business Logic（为什么需要这个函数）:
        命令行入口把校验失败转成非零退出码而非堆栈回溯，让采集-选点脚本
        可以可靠地判断成败。

    Code Logic（这个函数做什么）:
        解析 argv（缺省用 sys.argv[1:]）并分发到 `select` 子命令；
        ValueError（含 stats 校验失败）打印到 stderr 并返回 2，成功返回 0；
        参数用法错误沿用 argparse 的 SystemExit(2)。
    """
    parser = _build_select_parser()
    args = parser.parse_args(argv)
    try:
        return _run_select(args)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
