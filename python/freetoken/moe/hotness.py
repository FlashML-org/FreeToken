from __future__ import annotations

import atexit
import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# stats 文件 schema 版本：counts 布局或 meta 字段变化时递增，选点工具按版本拒绝不认识的文件。
SCHEMA_VERSION = 1

# 动态重钉（设计 §10）的 EMA 衰减：半衰期 == 窗口时长 ⇒ 每封口一个窗口平滑热度
# 衰减为一半（ema = 0.5*ema + 0.5*window；恒定流量下稳态收敛到窗口计数本身）。
_WINDOW_DECAY = 0.5


class ExpertHotness:
    """MoE 专家路由热度计数器（offload 路径的选点数据源 + 动态重钉的窗口热度源）。

    每个 MoE 层前向在 ``fused_topk`` 之后、路由改写之前调用 ``record``，把原始专家
    id 累加进 device 侧计数器 ``counts [num_layers * num_experts] int64``。计数全程
    固定 shape、无 host 同步，因此 CUDA graph 捕获后每次 replay 都会按真实路由重复
    累加。宿主侧只在墙钟间隔到达时做一次 D2H 排空（``maybe_flush``），进程退出时经
    ``atexit`` 落盘为选点工具消费的 JSON（schema 见 SCHEMA_VERSION）。

    配置 ``window_interval_s``（动态重钉）时，同一次排空的增量额外喂给滑动窗口
    累计器：窗口边界到达即封口出"完整窗口"，经 EMA 平滑后供重钉决策查询
    （``window_topk``/``ema_kth``）。两种消费各自只增自己的宿主缓冲，互不破坏。
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        device: torch.device,
        out_path: str | None,
        flush_interval_s: float = 60.0,
        meta: dict[str, Any] | None = None,
        window_interval_s: float | None = None,
    ) -> None:
        """
        Business Logic（为什么需要这个函数）:
            显存钉住热点专家前需要离线采集每个（层, 专家）的真实路由热度，作为选点
            工具的输入；动态重钉还需要同一份计数的滑动窗口视图。采集必须近零开销
            （不能每步 D2H 同步），所以设备侧累加、宿主侧周期排空。

        Code Logic（这个函数做什么）:
            分配 [num_layers * num_experts] 的 int64 device 计数器（零初始化）、同尺寸
            的 numpy 宿主累计数组和 ones 常量缓存；记录排空间隔、输出路径（None =
            仅喂窗口、不落盘）与来源元信息，并初始化墙钟与 total_tokens 记账。
            window_interval_s 给定时启用滑动窗口模式：预分配窗口增量累计器，EMA/
            完整窗口在首个窗口封口时才生成（其前的查询以 has_window=False 拒绝）。
        """
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.device = device
        self.out_path = out_path
        self.flush_interval_s = flush_interval_s
        self.meta: dict[str, Any] = dict(meta or {})
        # device 侧累计计数：index = layer_id * num_experts + expert_id
        self.counts = torch.zeros(num_layers * num_experts, dtype=torch.int64, device=device)
        # 宿主侧累计数组（历次排空的累加结果），save 的数据源
        self._host_counts = np.zeros(num_layers * num_experts, dtype=np.int64)
        # 累计记录的 token 数（每次 record 加 topk_ids.shape[0]），写进 meta
        self.total_tokens = 0
        self._last_flush = time.monotonic()
        self._t0 = time.monotonic()
        # atexit 防重入：install 只注册一次，_exit_flush 只落盘一次
        self._exit_installed = False
        self._exit_done = False
        # ---- 动态重钉的滑动窗口（设计 §10）：窗口时长 = window_interval_s ----
        # None = 窗口模式关闭（maybe_flush 行为与原先逐字节一致）
        self.window_interval_s = float(window_interval_s) if window_interval_s else None
        # 当前窗口的宿主侧部分累计：一次排空可能横跨窗口边界，delta 先落这里，
        # 与 device 计数器共同组成"当前窗口"（device 部分 + 已排空部分），全量累计
        # （_host_counts）语义不受影响。
        self._window_acc = np.zeros(num_layers * num_experts, dtype=np.int64)
        self._window_started: float | None = None
        # 最近一个完整窗口的 D2H 快照（上一窗口保留为参照）与其 EMA 平滑热度
        self._window: np.ndarray | None = None
        self._ema: np.ndarray | None = None
        # save() 时取"当前钉住集"的回调（动态重钉落盘 pinned 字段用）；None = 不写
        self.pinned_provider: Callable[[], list[list[int]]] | None = None

    def record(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Business Logic（为什么需要这个函数）:
            统计必须在 ``ensure_experts`` 把 topk_ids 原地改写成 slot id 之前进行，
            这样拿到的才是真实专家 id；decode 走 CUDA graph 时统计也必须随 replay
            正确累加。

        Code Logic（这个函数做什么）:
            把该层本步的 topk 扁平化映射进全局平坦 id 空间（expert + layer_id *
            num_experts），用 index_add_ 累加同形全 1 增量；固定 shape、无 host
            同步，graph 可捕获。增量用 ``ones_like`` 就地生成而不是共享缓存 buffer：
            共享 buffer 按需扩容会释放旧存储，而已捕获的早期 decode graph 仍引用旧
            指针（use-after-free：释放后复用的 float 字节被当 int64 累加，真实
            NVFP4 服务中把计数器污染到 1e18 量级）。``ones_like`` 在 graph 捕获时
            落在各 graph 自己的内存池里，随池存活，replay 永远读到 1；eager 路径
            （prefill）每层每次一小块分配，量级可忽略。host 侧 total_tokens 按本步
            token 数递增（graph 捕获时只执行一次，属于已知的一次性 warm-up 偏差）。
        """
        flat = topk_ids.flatten() + layer_id * self.num_experts
        self.counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int64))
        self.total_tokens += int(topk_ids.shape[0])

    def maybe_flush(self) -> bool:
        """
        Business Logic（为什么需要这个函数）:
            device 计数器不能无限增长，也不能每步 D2H 同步拖慢推理；调度循环的每次
            迭代都会调用本方法，按墙钟间隔低频排空（98 KB 级 D2H，开销可忽略），
            崩溃时最多丢一个间隔的统计。窗口模式下同一份增量还要喂滑动窗口，而窗口
            边界可能早于落盘间隔——排空门控因此取两者较小值，保证窗口按时封口。

        Code Logic（这个函数做什么）:
            距上次排空未满间隔（窗口模式 = min(flush_interval_s, window_interval_s)，
            否则 flush_interval_s）直接返回 False；否则把 device 计数器 D2H 为增量
            delta 并清零 device 侧，累加进宿主累计，窗口模式再交给 _feed_window，
            刷新墙钟后返回 True。
        """
        now = time.monotonic()
        interval = self.flush_interval_s
        if self.window_interval_s is not None:
            interval = min(interval, self.window_interval_s)
        if now - self._last_flush < interval:
            return False
        # 先取增量的独立宿主副本再清零：CPU 设备上 .cpu() 返回同一存储的视图，
        # 顺序反了会把 delta 一起清成零
        delta = self.counts.detach().cpu().numpy().copy()
        self.counts.zero_()
        self._last_flush = now
        self._host_counts += delta
        if os.getenv("FREETOKEN_REPIN_DEBUG"):
            nz = np.nonzero(delta)[0]
            logger.info(
                "REPIN_DEBUG flush: delta.sum=%d max=%d nonzero=%d first=%s "
                "counts.dtype=%s shape=%s",
                int(delta.sum()), int(delta.max()), int(nz.size),
                [(int(i), int(delta[i])) for i in nz[:8]],
                self.counts.dtype, tuple(self.counts.shape),
            )
        if self.window_interval_s is not None:
            self._feed_window(delta, now)
        return True
    def _feed_window(self, delta: np.ndarray, now: float) -> bool:
        """
        Business Logic（为什么需要这个函数）:
            动态重钉需要"完整窗口"粒度的热度：device 计数与窗口边界不同相（排空
            可横跨边界），增量必须先累计进窗口缓冲、边界到达时才封口，否则窗口会
            系统性丢掉边界前的流量。

        Code Logic（这个函数做什么）:
            把一次排空的 delta 并入当前窗口累计器；首个 delta 记窗口起点。未到
            window_interval_s 返回 False；到达则封口：完整窗口 = 累计器快照，EMA =
            首窗直接取窗口计数（热启动）、其后 ema = decay*ema + (1-decay)*窗口
            （半衰期 = 窗口时长），随后清空累计器并重置窗口起点，返回 True。
        """
        self._window_acc += delta
        if self._window_started is None:
            self._window_started = now
        assert self.window_interval_s is not None
        if now - self._window_started < self.window_interval_s:
            return False
        window = self._window_acc
        if self._ema is None:
            self._ema = window.astype(np.float64)
        else:
            self._ema = _WINDOW_DECAY * self._ema + (1.0 - _WINDOW_DECAY) * window
        self._window = window
        self._window_acc = np.zeros_like(self._window_acc)
        self._window_started = now
        return True

    @property
    def has_window(self) -> bool:
        """是否已有至少一个封口的窗口（首个窗口之前的重钉查询应直接跳过）。"""
        return self._ema is not None

    def ema_counts(self) -> np.ndarray:
        """平滑热度 EMA 的 [num_layers, num_experts] float64 视图（无完整窗口时抛 RuntimeError）。"""
        if self._ema is None:
            raise RuntimeError("尚无封口的热度窗口：先让 maybe_flush 在窗口模式下跑满一个 window_interval_s")
        return self._ema.reshape(self.num_layers, self.num_experts)

    def window_topk(self, layer_id: int, k: int) -> list[int]:
        """
        Business Logic（为什么需要这个函数）:
            动态重钉决策的输入是"按当前窗口热度排序的专家"：与选点工具同一规则
            （计数降序、平局取小 id）保证在线窗口视图与离线选点语义一致。

        Code Logic（这个函数做什么）:
            对指定层的 EMA 计数按 (-count, expert_id) 升序排序取前 k 个专家 id
            （等价计数降序、平局小 id 优先）；k 截断到专家数。无完整窗口抛
            RuntimeError。
        """
        ema = self.ema_counts()[layer_id]
        keep = max(0, min(int(k), self.num_experts))
        order = sorted(range(self.num_experts), key=lambda e: (-ema[e], e))
        return order[:keep]

    def ema_kth(self, layer_id: int, k: int) -> float:
        """
        Business Logic（为什么需要这个函数）:
            重钉迟滞比较需要"当前第 K 名的 EMA 计数"这类按名次取值的查询；
            决策方（hot_pin.plan_repin_swaps）与测试都需要一个确定性的名次语义。

        Code Logic（这个函数做什么）:
            返回该层 EMA 计数第 k 名（k 从 1 计，降序、平局取小 id）的值；k 越界
            抛 ValueError。无完整窗口抛 RuntimeError。
        """
        if not 1 <= k <= self.num_experts:
            raise ValueError(f"名次 k 必须在 [1, {self.num_experts}]，实际为 {k}")
        ema = self.ema_counts()[layer_id]
        order = sorted(range(self.num_experts), key=lambda e: (-ema[e], e))
        return float(ema[order[k - 1]])

    def save(self) -> None:
        """
        Business Logic（为什么需要这个函数）:
            选点工具消费的是落盘 JSON（统计 → 选点的契约文件）；进程退出或用户主动
            导出时必须把宿主累计写成原子、schema 稳定的文件。配置了 pinned_provider
            （动态重钉）时附带当前钉住集，供事后分析窗口漂移（schema_version 仍为
            1，pinned 为可选键，选点工具只读已知键、向后兼容）。

        Code Logic（这个函数做什么）:
            out_path 为 None（纯窗口模式）时直接返回；否则先强制排空一次 device
            计数器（不落未排空数据），组装 schema_version=1 的 payload（meta 附
            num_layers/num_experts/total_tokens/duration_s/created_at，counts 为嵌套
            list[int]，pinned_provider 给定时附 pin-list 风格的 pinned 条目），写
            out_path + ".tmp" 后 os.replace 原子替换。
        """
        if not self.out_path:
            return
        self._host_counts += self.counts.cpu().numpy()
        self.counts.zero_()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                **self.meta,
                "num_layers": self.num_layers,
                "num_experts": self.num_experts,
                "total_tokens": self.total_tokens,
                "duration_s": round(time.monotonic() - self._t0, 3),
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            "counts": self._host_counts.reshape(self.num_layers, self.num_experts).tolist(),
        }
        if self.pinned_provider is not None:
            payload["pinned"] = [
                {"layer": layer_id, "experts": list(pins)}
                for layer_id, pins in enumerate(self.pinned_provider())
            ]
        parent = os.path.dirname(self.out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = self.out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, self.out_path)

    @staticmethod
    def load(path: str) -> dict:
        """
        Business Logic（为什么需要这个函数）:
            选点工具与钉住启动路径都要读统计文件；契约（版本、维度）必须在读入口处
            统一校验，坏文件要在选点时报错，而不是产出错误的 pin list。

        Code Logic（这个函数做什么）:
            读 JSON 并校验 schema_version、meta 里的 num_layers/num_experts 与 counts
            的嵌套维度一致；通过则原样返回整个 payload dict，任何不符抛 ValueError。
        """
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported hotness stats schema_version {version!r} in {path} "
                f"(expected {SCHEMA_VERSION})"
            )
        meta = payload.get("meta")
        counts = payload.get("counts")
        if not isinstance(meta, dict) or not isinstance(counts, list):
            raise ValueError(f"malformed hotness stats file {path}: meta/counts missing")
        num_layers = meta.get("num_layers")
        num_experts = meta.get("num_experts")
        if not isinstance(num_layers, int) or not isinstance(num_experts, int):
            raise ValueError(f"malformed hotness stats file {path}: meta dimensions missing")
        if len(counts) != num_layers or not all(
            isinstance(layer_counts, list) and len(layer_counts) == num_experts for layer_counts in counts
        ):
            raise ValueError(
                f"hotness stats dimension mismatch in {path}: counts must be "
                f"[{num_layers}][{num_experts}]"
            )
        return payload

    def install_exit_flush(self) -> None:
        """
        Business Logic（为什么需要这个函数）:
            采集结束依赖进程退出落盘；若不在退出钩子里兜底，一次 SIGTERM 就丢掉全部
            统计。注册动作可能被调用多次（幂等），落盘也必须只发生一次。

        Code Logic（这个函数做什么）:
            首次调用时用 atexit.register 注册 _exit_flush；重复调用直接返回。
        """
        if self._exit_installed:
            return
        self._exit_installed = True
        atexit.register(self._exit_flush)

    def _exit_flush(self) -> None:
        """
        Business Logic（为什么需要这个函数）:
            atexit 回调在解释器关闭阶段执行，任何异常都不能向外抛；且 save 可能已被
            显式调用过，重复写文件没有意义。

        Code Logic（这个函数做什么）:
            防重入标志保证只执行一次；调用 save 把统计原子落盘，失败时记录 warning
            并吞掉异常（退出路径不能因统计文件失败而报错）。
        """
        if self._exit_done:
            return
        self._exit_done = True
        try:
            self.save()
        except Exception:
            logger.warning("failed to flush expert hotness stats to %s", self.out_path, exc_info=True)
