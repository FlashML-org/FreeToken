"""运行中调参文件（--tune-file）的解析与应用。

设计上"纯函数解析"与"线程轮询应用"分离：:func:`parse_tune_json` 只做文本 -> 合法
调参项的结构化转换（无 IO、无副作用，可直接单测）；:class:`TuneFilePoller` 负责
mtime 监听、读文件与应用到 ``OffloadMoeCache.set_fetch_params``（设备张量按指针
读取，值更新对已捕获 decode 图立即生效）。
"""

from __future__ import annotations

import json
import os
import threading
import time

from typing import TYPE_CHECKING

import torch

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)

# 轮询周期（秒）：调参文件按人类编辑粒度变化，2s 的 mtime 检查足够实时且开销可忽略。
POLL_INTERVAL_S = 2.0


def parse_tune_json(text: str) -> tuple[dict[str, float], list[str]]:
    """解析调参文件文本为（合法调参项, 告警文本列表）。

    Business Logic（为什么需要这个函数）:
        运行中调参要求"文件内容是否合法"的判定与应用解耦：非法 JSON 静默跳过、
        越界值忽略但告警一次、缺键不动现有值。纯函数形态让这些规则可以脱离文件
        系统与轮询线程直接单测，也保证轮询线程里不做任何隐式策略。

    Code Logic（这个函数做什么）:
        json.loads 解析（失败抛 ValueError，由调用方静默跳过），顶层必须是 object；
        只认两个键："fetch_fraction"（数值且在 [0, 1]，否则剔除并记一条告警）与
        "pin_k"（数值，交由动态重钉管理器在下一 idle 安全点应用）。返回
        (updates, warnings)：updates 只含合法键（fetch_fraction 为 float、pin_k 为
        float，整型语义由应用方校验）；warnings 是人读告警文本，由调用方负责打日志
        （每次文件变化至多一轮）。
    """
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tune file is not valid JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise ValueError("tune file must be a JSON object")
    updates: dict[str, float] = {}
    warnings: list[str] = []
    if "fetch_fraction" in doc:
        value = doc["fetch_fraction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            warnings.append(f"tune file: fetch_fraction {value!r} is not a number, ignored")
        elif not 0.0 <= float(value) <= 1.0:
            warnings.append(
                f"tune file: fetch_fraction {value!r} outside [0, 1], ignored"
            )
        else:
            updates["fetch_fraction"] = float(value)
    if "pin_k" in doc:
        value = doc["pin_k"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            warnings.append(f"tune file: pin_k {value!r} is not a number, ignored")
        else:
            updates["pin_k"] = float(value)
    return updates, warnings


class TuneFilePoller:
    """调参文件守护轮询线程：mtime 变化 -> 解析 -> 应用到 OffloadMoeCache。

    Business Logic（为什么需要这个类）:
        用户在服务运行中（模型已捕获 decode 图、无法重捕获调参）需要不改代码地
        调 hybrid 拉取比例；一个每 2s 看 mtime 的 JSON 文件是最小可用的带外交互
        入口。线程常驻而非挂进 scheduler 循环：scheduler 可能长时间阻塞在消息
        等待上，墙钟轮询必须独立于请求到达。

    Code Logic（这个类做什么）:
        start() 起 daemon 线程，每 interval_s 调一次 poll_once：stat mtime 未变
        （或文件暂不存在）静默返回；变化则读文本、parse_tune_json——非法 JSON /
        读失败静默跳过（mtime 已记录，避免每轮重读重刷日志），fetch_fraction 合法
        则先 torch.cuda.synchronize 做安全栅栏（同动态重钉：不在 in-flight 回放
        中途改值）再 cache.set_fetch_params(cache.hybrid_max_fetch, fraction)
        （cap 不变）。pin_k 经 cache.repin_manager（动态重钉管理器，engine 装配）
        set_target_k 记录最新目标——应用是异步的，管理器在下一 idle 安全点扩缩落
        地，文件里写多少次都只保留最新目标；非整数/越界值告警忽略；没有管理器
        （未启用动态重钉）时保持原行为只打一条忽略日志。stop() 置位事件并 join。
    """

    def __init__(
        self, cache: "OffloadMoeCache", path: str, interval_s: float = POLL_INTERVAL_S
    ) -> None:
        self.cache = cache
        self.path = path
        self.interval_s = float(interval_s)
        self._last_mtime: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """启动守护轮询线程（幂等；进程退出时 daemon 线程自动收尾）。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="tune-file-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """置位停止事件并等待线程退出（engine.shutdown 调用；未启动时空操作）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        """线程主循环：周期等待 + 单次检查；任何意外异常降级为告警日志，绝不离场。

        Code Logic（这个函数做什么）:
            首轮先打一条 alive 心跳（证明线程在跑、给出基线 mtime），意外异常以
            warning 记录（60s 节流：持续失败不刷屏，但绝不在 INFO 级别下静默）。
        """
        first = True
        last_error_log = 0.0
        while not self._stop.wait(self.interval_s):
            try:
                self.poll_once()
                if first:
                    logger.info_rank0(
                        "tune file poller alive (baseline mtime set on first poll)"
                    )
                    first = False
            except Exception as exc:  # noqa: BLE001 -- 轮询线程绝不能带崩引擎
                now = time.monotonic()
                if now - last_error_log >= 60.0:
                    last_error_log = now
                    logger.warning("tune file poll failed: %s: %s", type(exc).__name__, exc)

    def poll_once(self) -> None:
        """单次 mtime 检查与应用（单测可直接调用；mtime 未变 / 解析失败静默跳过）。"""
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            # 文件暂不存在：清掉记录，等它出现（或重建）时按新文件应用
            self._last_mtime = None
            return
        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        try:
            with open(self.path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return
        try:
            updates, warnings = parse_tune_json(text)
        except ValueError:
            return  # 非法 JSON：静默跳过
        for message in warnings:
            logger.warning(message)
        fraction = updates.get("fetch_fraction")
        if fraction is not None:
            if self.cache.device.type == "cuda":
                # 安全栅栏（同动态重钉）：等 in-flight 回放结束再改值，避免撕裂读
                torch.cuda.synchronize(self.cache.device)
            self.cache.set_fetch_params(self.cache.hybrid_max_fetch, fraction)
            logger.info_rank0(
                f"tune file: hybrid fetch fraction -> {fraction:.1%} "
                f"(cap {self.cache.hybrid_max_fetch} unchanged)"
            )
        if "pin_k" in updates:
            self._apply_pin_k(updates["pin_k"])

    def _apply_pin_k(self, value: float) -> None:
        """把 pin_k 交给动态重钉管理器记为最新目标（无管理器/坏值只告警，绝不离场）。

        Business Logic（为什么需要这个函数）:
            pin_k 的语义是"最新意图"而非"立即执行"：真实扩缩必须等 idle 安全点
            （HotExpertRepinManager.apply_target_k），轮询线程只做目标转发；管理器
            可能未装配（未启用动态重钉）或目标非法（越界），两种情况都以一条日志
            收场——轮询线程绝不能带崩引擎。

        Code Logic（这个函数做什么）:
            经 cache.repin_manager 查找管理器（engine 装配在 cache 上，晚于 poller
            启动也有效）：无管理器打忽略日志保持旧行为；非整数打告警；set_target_k
            抛 ValueError（越界/未钉住）转告警；成功则打"下一 idle 安全点生效"日志。
        """
        manager = getattr(self.cache, "repin_manager", None)
        if manager is None:
            logger.info_rank0(
                "tune file: pin_k 已忽略（未启用动态重钉：需要 --hot-expert-list 与 "
                "--hot-expert-repin-interval-s > 0）"
            )
            return
        if not float(value).is_integer():
            logger.warning(f"tune file: pin_k {value!r} 不是整数，忽略")
            return
        try:
            manager.set_target_k(int(value))
        except ValueError as exc:
            logger.warning(f"tune file: pin_k {value!r} 已忽略（{exc}）")
            return
        logger.info_rank0(f"tune file: pin_k -> {int(value)}（下一 idle 安全点生效）")
