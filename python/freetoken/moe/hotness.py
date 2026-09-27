from __future__ import annotations

import atexit
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# stats 文件 schema 版本：counts 布局或 meta 字段变化时递增，选点工具按版本拒绝不认识的文件。
SCHEMA_VERSION = 1


class ExpertHotness:
    """MoE 专家路由热度计数器（offload 路径的选点数据源）。

    每个 MoE 层前向在 ``fused_topk`` 之后、路由改写之前调用 ``record``，把原始专家
    id 累加进 device 侧计数器 ``counts [num_layers * num_experts] int64``。计数全程
    固定 shape、无 host 同步，因此 CUDA graph 捕获后每次 replay 都会按真实路由重复
    累加。宿主侧只在墙钟间隔到达时做一次 D2H 排空（``maybe_flush``），进程退出时经
    ``atexit`` 落盘为选点工具消费的 JSON（schema 见 SCHEMA_VERSION）。
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        device: torch.device,
        out_path: str,
        flush_interval_s: float = 60.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Business Logic（为什么需要这个函数）:
            显存钉住热点专家前需要离线采集每个（层, 专家）的真实路由热度，作为选点
            工具的输入；采集必须近零开销（不能每步 D2H 同步），所以设备侧累加、宿主
            侧周期排空。

        Code Logic（这个函数做什么）:
            分配 [num_layers * num_experts] 的 int64 device 计数器（零初始化）、同尺寸
            的 numpy 宿主累计数组和 ones 常量缓存；记录排空间隔、输出路径与来源元信息
            （model_path / moe_strategy / quant_format / top_k 等），并初始化墙钟与
            total_tokens 记账。
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
        # record 用的全 1 常量缓存：预分配 + 按需扩展，避免每次 forward 分配
        self._ones_buf: torch.Tensor | None = None
        # 累计记录的 token 数（每次 record 加 topk_ids.shape[0]），写进 meta
        self.total_tokens = 0
        self._last_flush = time.monotonic()
        self._t0 = time.monotonic()
        # atexit 防重入：install 只注册一次，_exit_flush 只落盘一次
        self._exit_installed = False
        self._exit_done = False

    def _ones(self, n: int) -> torch.Tensor:
        """
        Business Logic（为什么需要这个函数）:
            record 每层每步都要给 index_add_ 提供全 1 增量，逐次分配会在 MoE 热路径上
            制造分配器压力；固定 buffer 也是 CUDA graph 可捕获的前提。

        Code Logic（这个函数做什么）:
            返回缓存 buffer 的前 n 个元素的视图；buffer 不存在或不够长时按 n 重新分配
            （ones 只写一次，之后只读，replay 语义安全）。
        """
        if self._ones_buf is None or self._ones_buf.numel() < n:
            self._ones_buf = torch.ones(n, dtype=torch.int64, device=self.device)
        return self._ones_buf[:n]

    def record(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Business Logic（为什么需要这个函数）:
            统计必须在 ``ensure_experts`` 把 topk_ids 原地改写成 slot id 之前进行，
            这样拿到的才是真实专家 id；decode 走 CUDA graph 时统计也必须随 replay
            正确累加。

        Code Logic（这个函数做什么）:
            把该层本步的 topk 扁平化映射进全局平坦 id 空间（expert + layer_id *
            num_experts），用 index_add_ 累加预分配的全 1 增量；固定 shape、无 host
            同步，graph 可捕获。host 侧 total_tokens 按本步 token 数递增（graph 捕获
            时只执行一次，属于已知的一次性 warm-up 偏差）。
        """
        flat = topk_ids.flatten() + layer_id * self.num_experts
        self.counts.index_add_(0, flat, self._ones(topk_ids.numel()))
        self.total_tokens += int(topk_ids.shape[0])

    def maybe_flush(self) -> bool:
        """
        Business Logic（为什么需要这个函数）:
            device 计数器不能无限增长，也不能每步 D2H 同步拖慢推理；调度循环的每次
            迭代都会调用本方法，按墙钟间隔低频排空（98 KB 级 D2H，开销可忽略），
            崩溃时最多丢一个间隔的统计。

        Code Logic（这个函数做什么）:
            距上次排空未满 flush_interval_s 直接返回 False；否则把 device 计数器 D2H
            累加进宿主 numpy 数组并清零 device 侧，刷新墙钟后返回 True。
        """
        now = time.monotonic()
        if now - self._last_flush < self.flush_interval_s:
            return False
        self._host_counts += self.counts.cpu().numpy()
        self.counts.zero_()
        self._last_flush = now
        return True

    def save(self) -> None:
        """
        Business Logic（为什么需要这个函数）:
            选点工具消费的是落盘 JSON（统计 → 选点的契约文件）；进程退出或用户主动
            导出时必须把宿主累计写成原子、schema 稳定的文件。

        Code Logic（这个函数做什么）:
            先强制排空一次 device 计数器（不落未排空数据），组装 schema_version=1 的
            payload（meta 附 num_layers/num_experts/total_tokens/duration_s/created_at，
            counts 为嵌套 list[int]），写 out_path + ".tmp" 后 os.replace 原子替换。
        """
        self._host_counts += self.counts.cpu().numpy()
        self.counts.zero_()
        payload = {
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
