"""热度统计采集负载：向 FreeToken 服务打多样化代码类请求，用于专家使用频率统计。

Business Logic（为什么需要这个函数）:
    钉住热集的质量取决于热度统计对目标工作负载（编码类）的代表性与覆盖量；
    需要一个可复现的驱动器，把多语言/多任务类型/多长度档位的代码请求以受控
    并发打入服务，使路由统计收敛到稳定的专家使用分布。

Code Logic（这个函数做什么）:
    以 ThreadPoolExecutor 并发向 /v1/chat/completions 发送内置提示池中的请求
    （temperature/max_tokens 分档制造路由多样性，长 prefill 用例覆盖预填路径），
    统计吞吐与错误，输出摘要 JSON。用法：
    python collect_hot_stats_workload.py --host 127.0.0.1 --port 1936 \
        --model NAME --total 300 --concurrency 6 --summary-out /tmp/summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

# 多样化代码类提示池：算法/调试/审查/重构/正则/SQL/并发/测试/多语言/长 prefill
_SHORT_PROMPTS: list[str] = [
    "用 Python 实现一个 LRU 缓存，要求 O(1) 的 get/put，并写清楚每个方法的复杂度。",
    "写一个 Rust 函数，统计字符串中出现次数最多的 UTF-8 字符，处理 Unicode 边界。",
    "帮我 review 这段代码的风险点：`with open(path) as f: data = json.load(f)`（生产环境读取用户上传的配置）。",
    "实现一个线程安全的环形缓冲区（Python），说明锁的粒度选择。",
    "写一个正则表达式，匹配 IPv4 但排除 10.0.0.0/8 段，并解释每个部分。",
    "用 C++20 写一个无锁的单生产者单消费者队列骨架，并指出内存序的关键点。",
    "SQL：一张 10 亿行的订单表，查询每个用户最近 3 笔订单，写出最优写法并解释索引设计。",
    "用 pandas 把嵌套 JSON 展平成宽表，列名用点号路径，处理数组展开。",
    "写一个 Python 装饰器：记录异步函数耗时，超过阈值时打 warning，支持嵌套。",
    "Go：实现一个带 jitter 的指数退避重试函数，可取消。",
    "调试：下面的递归为什么栈溢出？给出两种修法并比较。\ndef walk(node):\n    return walk(node.next) + node.val",
    "写一个 shell 脚本：找出最近 24 小时内被修改且包含 TODO 的 .py 文件，输出为 markdown 清单。",
    "实现布隆过滤器的 Python 类，参数化误判率，说明内存估算公式。",
    "TypeScript：给一个深层嵌套对象写类型安全的 get(obj, 'a.b[0].c') 函数。",
    "解释 Python GIL 对多线程数值计算的影响，并给出三种绕过方案的取舍。",
    "写一个 pytest 参数化用例集，覆盖一个解析 'key=value;key2=value2' 的函数的全部边界。",
    "把下面的回调风格代码重构成 async/await，保持错误传播语义：`fs.readFile(p, (e, d) => cb(parse(d)))`。",
    "实现一个简易的令牌桶限流器（Python），支持并发，说明时间来源的选择。",
    "Java：解释 HashMap 在 JDK8 之后树化的条件与原因。",
    "用 Python 的 ctypes 调用 libc 的 qsort 对 numpy 数组排序，指出这个做法的坑。",
    "写一个 git pre-commit 钩子：阻止大文件与密钥模式（AKIA/sk-）提交。",
    "给定二叉树前序+中序，写出重建树的 Python 实现并证明正确性要点。",
    "设计一个幂等的支付回调接口（HTTP 语义 + 存储），列出边界情况。",
    "解释 CAPM 不存在于分布式系统的说法从何而来——转成严肃版：解释 CAP 定理与最终一致性的工程取舍。",
    "用 Python 实现一个跳表（skip list），接口对齐 dict 的子集。",
    "写一个零依赖的 Python HTTP 健康检查探针，带指数退避与抖动。",
    "Rust：解释 Send/Sync 的区别，并各给一个编译失败的例子。",
    "把 O(n²) 的两数之和暴力解优化到 O(n)，并说明哈希冲突最坏情况。",
    "写一个 Makefile 目标：构建、测试、lint 三阶段，带并行与依赖。",
    "解释数据库连接池的泄漏检测思路，并在 Python/SQLAlchemy 下落地。",
    "实现一个支持通配符 * 和 ? 的匹配函数，要求 O(nm) DP 而不是回溯。",
    "Kubernetes：写一个 Deployment + PDB 的 YAML，滚动更新时保证至少 2 副本可用。",
    "用 Python 的 struct 模块解析一个自定义二进制协议头（magic/len/crc），含 CRC 校验。",
    "解释 UTF-8 的自同步性质，并写一个验证函数判断字节流是否为合法 UTF-8（不用标准解码器）。",
    "实现一个简单的中文分词逆向前缀树（最大正向匹配），给出复杂度。",
    "写一个 Python 上下文管理器：临时修改环境变量并保证恢复（含异常路径）。",
    "解释一致性哈希，并用 30 行 Python 实现带虚拟节点的环。",
    "给出一个无锁计数器的错误实现（count += 1），分析竞态窗口并修复。",
    "写一个 SQL 迁移脚本框架的 Python 雏形：版本表、前向执行、失败回滚点。",
    "C++：解释移动语义下 vector 扩容的异常安全保证（noexcept 与 swap）。",
    "实现一个滑动窗口最大值（单调队列），并解释为什么双端队列是必要的。",
    "把同步的 requests 爬虫改成 httpx 异步版，带并发上限与限速。",
    "写一个基于 difflib 的配置对比工具，输出 unified diff 并高亮敏感键。",
    "设计一个本地文件缓存层：key 规范、原子写、TTL、容量淘汰，给出核心实现。",
    "解释 epoll 的边缘触发与水平触发区别，写一个 Python selectors ET 模式 echo server。",
    "用 Python 实现矩阵链乘法的动态规划解，输出最优加括号方式。",
    "写一个代码生成器：从 JSON Schema 生成 pydantic v2 模型（处理嵌套/枚举/默认值）。",
    "实现一个简单的位图索引：set/clear/count/rank，用 int 当位板。",
    "解释零拷贝（sendfile/splice）在静态文件服务中的作用，给出 Python 可行的近似方案。",
    "写一个从 traceback 日志中聚合 top-N 异常的脚本（按 去参数化 的异常签名分组）。",
    "用 Python 实现 Damerau-Levenshtein 距离，剪枝后用于拼写检查。",
    "解释 Redis 分布式锁 Redlock 的争议点，并给一个单实例足够时的简化方案。",
    "写一个抽象：把同步函数包装成 async（线程池 vs 子进程的取舍矩阵）。",
    "实现一个简易解释器：支持 + - * / 与括号的中缀表达式求值（词法+递归下降）。",
    "C++：写一个 RAII 的文件描述符包装，禁止拷贝、允许移动。",
    "把一个 2000 行的 God Class 拆分：给出识别内聚边界的步骤清单与 Python 示例。",
    "写一个 JSONL 流式处理器：逐行读取、按 schema 校验、坏行进死信文件。",
    "实现 raft 选举的纸面演练：给出 5 节点在分区下的任期变化时间线。",
    "解释 TLS 握手中证书链验证的步骤，用 openssl s_client 的输出对照。",
    "Python：multiprocessing 与 threading 在 CPU 密集/IO 密集任务下的选型基准设计。",
    "写一个支持断点续传的分块下载器（Range + 状态文件）。",
    "实现一个 LRU+LFU 混合缓存（先 LFU 筛、同频 LRU），说明退化场景。",
    "写一个用于代码评审的 checklist 生成器：输入 diff，输出按风险分类的问题清单骨架。",
    "解释程序分析里的指针分析为什么是 NP-hard，给出流不敏感近似的样子。",
]

_LONG_PROMPTS: list[str] = [
    # 长 prefill 用例（贴大段代码让预填路径吃到真实流量）
    "Review 下面这段 Python 服务代码，按 正确性/并发/资源泄漏 分类给出问题清单：\n\n"
    "```python\nimport threading, sqlite3, json\n"
    "DB = sqlite3.connect('app.db', check_same_thread=False)\n"
    "lock = threading.Lock()\n"
    "def handle(msg):\n    data = json.loads(msg)\n"
    "    with lock:\n        cur = DB.execute('insert into events values (?)', (data['id'],))\n"
    "    if data.get('retry'):\n        threading.Timer(5, handle, args=[msg]).start()\n"
    "    return cur.lastrowid\n"
    "def cleanup():\n    DB.execute('delete from events where ts < date(\"now\", \"-30 day\")')\n"
    "    DB.commit()\n```\n逐条给出修法。",
    "下面是一个配置文件加载器的实现，指出它与 12-factor 的偏离点：\n\n"
    "```python\nclass Config:\n    def __init__(self, path='config.json'):\n        self.__dict__.update(json.load(open(path)))\n"
    "    def get(self, key, default=None):\n        try: return getattr(self, key)\n        except AttributeError: return default\n```\n"
    "重写它以支持环境变量覆盖与类型校验。",
    "这个 bash 备份脚本有什么问题？\n\n```bash\n#!/bin/bash\nSRC=$1\nDST=/backup/$(date +%F)\n"
    "cp -r $SRC $DST\nfind $DST -mtime +30 -delete\ngzip $DST/*\n```\n给出加固后的版本。",
]

_PROMPTS: list[str] = _SHORT_PROMPTS + _LONG_PROMPTS


def _build_jobs(total: int, seed: int) -> list[dict[str, Any]]:
    """生成请求作业列表：温度/长度分档 + 随机洗牌，制造路由多样性。"""
    rng = random.Random(seed)
    jobs: list[dict[str, Any]] = []
    for i in range(total):
        prompt = _PROMPTS[i % len(_PROMPTS)]
        max_tokens = rng.choice([256, 384, 512, 768, 1024])
        temperature = rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])
        jobs.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "idx": i})
    return jobs


def run(args: argparse.Namespace) -> None:
    """并发执行请求并输出摘要。"""
    jobs = _build_jobs(args.total, args.seed)
    client = httpx.Client(base_url=f"http://{args.host}:{args.port}", timeout=httpx.Timeout(600.0))
    ok = failed = 0
    comp_tokens = prompt_tokens = 0
    errors: list[str] = []
    t0 = time.monotonic()
    done = 0

    def one(job: dict[str, Any]) -> dict[str, int]:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": job["prompt"]}],
                "max_tokens": job["max_tokens"],
                "temperature": job["temperature"],
            },
        )
        resp.raise_for_status()
        usage = resp.json().get("usage", {})
        return {
            "completion": int(usage.get("completion_tokens", 0)),
            "prompt": int(usage.get("prompt_tokens", 0)),
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                usage = fut.result()
                ok += 1
                comp_tokens += usage["completion"]
                prompt_tokens += usage["prompt"]
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if len(errors) < 10:
                    errors.append(f"#{job['idx']}: {type(exc).__name__}: {exc}")
            done += 1
            if done % 25 == 0:
                elapsed = time.monotonic() - t0
                print(f"进度 {done}/{len(jobs)}  ok={ok} fail={failed}  "
                      f"{comp_tokens} comp tok  {elapsed:.0f}s", flush=True)

    elapsed = time.monotonic() - t0
    summary = {
        "total": len(jobs), "ok": ok, "failed": failed,
        "completion_tokens": comp_tokens, "prompt_tokens": prompt_tokens,
        "elapsed_s": round(elapsed, 1),
        "aggregate_tok_s": round(comp_tokens / elapsed, 1) if elapsed else 0.0,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    """
    Business Logic（为什么需要这个函数）:
        把命令行参数装配成一次可复现的热度采集负载。

    Code Logic（这个函数做什么）:
        解析 host/port/model/total/concurrency/seed/summary-out 并执行 run。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1936)
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
