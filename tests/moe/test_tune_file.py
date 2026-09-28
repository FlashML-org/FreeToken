"""--tune-file 运行中调参的解析与轮询应用。

覆盖 moe/tune_file.py 的两半：parse_tune_json 的纯解析规则（合法/非法 JSON、越界、
缺键、非数值）与 TuneFilePoller 的 mtime 门控应用（变化才应用、解析失败静默跳过、
fetch_fraction 应用 / pin_k 只日志）。
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from freetoken.moe.tune_file import POLL_INTERVAL_S, TuneFilePoller, parse_tune_json


class _FakeCache:
    """记录 set_fetch_params 调用的最小 OffloadMoeCache 替身（无 CUDA 依赖）。

    Business Logic（为什么需要这个替身）:
        轮询应用的单元测试只关心"何时以何值调 set_fetch_params"，不关心设备张量
        细节；用替身把 poller 的应用行为从 CUDA 依赖中剥离，保持测试纯 CPU 可跑。

    Code Logic（这个替身做什么）:
        记录每次 set_fetch_params 的 (max_fetch, fetch_fraction) 到 calls；device
        伪造成 type="cpu" 使 poller 跳过 torch.cuda.synchronize 栅栏。
    """

    def __init__(self) -> None:
        self.hybrid_max_fetch: int = 32
        self.hybrid_fetch_fraction: float = 0.0
        self.calls: list[tuple[int, float]] = []
        self.device = SimpleNamespace(type="cpu")

    def set_fetch_params(self, max_fetch: int, fetch_fraction: float) -> None:
        """与真实 cache 同签名：cap 原样透传，只应用新 fraction（供断言）。"""
        self.calls.append((max_fetch, fetch_fraction))
        self.hybrid_max_fetch = max_fetch
        self.hybrid_fetch_fraction = fetch_fraction


def test_poll_interval_default():
    """轮询周期默认 2 秒（--tune-file 的 mtime 检查频率契约）。"""
    assert POLL_INTERVAL_S == 2.0


def test_parse_tune_json_valid_and_missing_keys():
    """合法键被解析、类型归一；缺失键不产生任何 updates 与告警。"""
    updates, warnings = parse_tune_json('{"fetch_fraction": 0.4, "pin_k": 16}')
    assert updates == {"fetch_fraction": 0.4, "pin_k": 16.0}
    assert warnings == []
    # 整数 0/1 合法并归一为 float
    updates, warnings = parse_tune_json('{"fetch_fraction": 0}')
    assert updates == {"fetch_fraction": 0.0}
    assert warnings == []
    updates, warnings = parse_tune_json('{"fetch_fraction": 1}')
    assert updates == {"fetch_fraction": 1.0}
    assert warnings == []
    # 空对象与无关键：合法但无操作
    updates, warnings = parse_tune_json("{}")
    assert updates == {} and warnings == []
    updates, warnings = parse_tune_json('{"other": 1}')
    assert updates == {} and warnings == []


def test_parse_tune_json_out_of_range_and_bad_types():
    """越界 fetch_fraction 忽略并给出告警文本；非数值（含 bool）同样忽略。"""
    updates, warnings = parse_tune_json('{"fetch_fraction": 1.5}')
    assert updates == {}
    assert len(warnings) == 1 and "outside [0, 1]" in warnings[0]
    updates, warnings = parse_tune_json('{"fetch_fraction": -0.1}')
    assert updates == {}
    assert len(warnings) == 1 and "outside [0, 1]" in warnings[0]
    # 字符串数字不是数值；true/false 是 bool（int 子类）也必须拒绝
    updates, warnings = parse_tune_json('{"fetch_fraction": "0.4"}')
    assert updates == {} and len(warnings) == 1
    updates, warnings = parse_tune_json('{"fetch_fraction": true}')
    assert updates == {} and len(warnings) == 1
    # pin_k 非数值：忽略并告警；合法 pin_k 单独出现不触发 set_fetch_params 语义
    updates, warnings = parse_tune_json('{"pin_k": "16"}')
    assert updates == {} and len(warnings) == 1
    # 越界与合法键混合：合法键保留、越界键剔除
    updates, warnings = parse_tune_json('{"fetch_fraction": 5.0, "pin_k": 8}')
    assert updates == {"pin_k": 8.0}
    assert len(warnings) == 1 and "outside [0, 1]" in warnings[0]


def test_parse_tune_json_invalid():
    """非法 JSON 与非 object 顶层抛 ValueError（由 poller 静默跳过）。"""
    with pytest.raises(ValueError):
        parse_tune_json("not json")
    with pytest.raises(ValueError):
        parse_tune_json("[1, 2, 3]")
    with pytest.raises(ValueError):
        parse_tune_json("null")


def test_poller_applies_only_on_mtime_change(tmp_path):
    """mtime 门控：文件出现/变化才应用；mtime 未变、非法 JSON、缺键都静默跳过。

    Business Logic（为什么需要这个测试）:
        轮询线程每 2s 醒一次，应用语义必须严格 mtime 门控——否则会反复应用同一
        内容并每轮重刷告警日志；同时非法内容绝不触发 set_fetch_params。

    Code Logic（这个测试做什么）:
        用固定 os.utime 精确控制 mtime（规避文件系统时间戳粒度），以 _FakeCache
        断言 calls 序列：文件不存在静默；出现且 mtime 变化 -> 应用 (cap, fraction)；
        mtime 未变不再应用；mtime 回退视为未变；改成非法 JSON 静默；越界值不应用；
        pin_k-only 不应用；fetch_fraction 与 cap 组合时 cap 透传不变。
    """
    path = tmp_path / "tune.json"
    cache = _FakeCache()
    poller = TuneFilePoller(cache, str(path))

    def touch(payload: str, mtime: float) -> None:
        path.write_text(payload, encoding="utf-8")
        os.utime(path, (mtime, mtime))

    # 文件不存在：静默跳过，不崩溃、不应用
    poller.poll_once()
    assert cache.calls == []
    # 文件出现（首见 mtime）：应用一次，cap 透传不变
    touch(json.dumps({"fetch_fraction": 0.3}), 1000.0)
    poller.poll_once()
    assert cache.calls == [(32, 0.3)]
    # mtime 未变：不再重复应用
    poller.poll_once()
    assert cache.calls == [(32, 0.3)]
    # 内容变化但 mtime 未变：跳过（mtime 是唯一门控）
    touch(json.dumps({"fetch_fraction": 0.9}), 1000.0)
    poller.poll_once()
    assert cache.calls == [(32, 0.3)]
    # mtime 变化：应用新值
    touch(json.dumps({"fetch_fraction": 0.9}), 2000.0)
    poller.poll_once()
    assert cache.calls == [(32, 0.3), (32, 0.9)]
    # 非法 JSON：静默跳过（且不因每轮重读而反复报错）
    touch("not json", 3000.0)
    poller.poll_once()
    poller.poll_once()
    assert cache.calls == [(32, 0.3), (32, 0.9)]
    # 越界值：不应用
    touch(json.dumps({"fetch_fraction": 1.5}), 4000.0)
    poller.poll_once()
    assert cache.calls == [(32, 0.3), (32, 0.9)]
    # pin_k-only：本阶段只解析，不触发 set_fetch_params
    touch(json.dumps({"pin_k": 16}), 5000.0)
    poller.poll_once()
    assert cache.calls == [(32, 0.3), (32, 0.9)]
    # 文件被删除：静默；重建后按新文件应用
    path.unlink()
    poller.poll_once()
    assert cache.calls == [(32, 0.3), (32, 0.9)]
    touch(json.dumps({"fetch_fraction": 0.1}), 6000.0)
    poller.poll_once()
    assert cache.calls == [(32, 0.3), (32, 0.9), (32, 0.1)]


def test_poller_cap_follows_current_cache_value(tmp_path):
    """应用时 cap 取 cache 当前 hybrid_max_fetch（只改 fraction，cap 不被覆写）。

    Business Logic（为什么需要这个测试）:
        --tune-file 的 fetch_fraction 语义是"比例分流替换固定 cap"，应用时必须以
        cache 现行 cap 调 set_fetch_params（真实 cache 的 set_fetch_params 会同步
        两侧），poller 无自身 cap 状态、不得引入第二个事实源。

    Code Logic（这个测试做什么）:
        改替身的 hybrid_max_fetch 后投递新 fraction，断言 poller 透传的是改动后的
        cap 值而非构造时的旧值。
    """
    path = tmp_path / "tune.json"
    cache = _FakeCache()
    poller = TuneFilePoller(cache, str(path))
    cache.hybrid_max_fetch = 7  # 模拟运行中已把 cap 改为 7（如显式 --moe-hybrid-max-fetch）
    path.write_text(json.dumps({"fetch_fraction": 0.6}), encoding="utf-8")
    os.utime(path, (1000.0, 1000.0))
    poller.poll_once()
    assert cache.calls == [(7, 0.6)]
