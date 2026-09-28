# 设计：显存钉住热点专家（VRAM Hot Expert Pinning）

- 分支：`feat/vram-hot-expert-pin`（基于上游 main `0d652e7`）
- 目标：消除当前 offload/hybrid 架构中"host tier 全量专家 + 显存 LRU 缓存副本"的冗余——把高频专家**单副本**常驻显存（不参与 LRU 驱逐与 host 换入换出），host tier 只装其余冷专家，从而把 host RAM 需求从 `全量专家` 降到 `冷专家`，使 180B 级 FP8 checkpoint（专家 112.5 GiB）能在 125 GiB 内存的机器上以真实裕度加载与服务。
- 附带能力：专家热度统计采集（NVFP4 模型跑大量目标域推理后导出），以及离线选点工具。

## 1. 背景与现状（本仓库代码事实）

| 事实 | 位置 |
|---|---|
| decode 路由 top-k：`fused_topk` 产出 `topk_ids [num_tokens, top_k] int32` | `layers/moe.py:227-237`（decode）、`:240-251`（prefill） |
| `ensure_experts` 会**原地**把 `topk_ids` 改写成 slot id；统计必须在改写前 | `layers/moe.py:285`、`moe/offload_kernels.py:19-40` |
| slot cache 状态全部在 GPU：`slot_for_id [L,E] int32`、`id_of_slot [cache_size] int32`、`usage [cache_size] int64`、`step () int64` | `moe/offload_cache.py:169-191` |
| offload 路径走 flashlib `lru_ensure`（外部包，**不可改**）；受害槽 = `argmin(usage)`，且 `usage == step`（本次调用刚命中）的槽**不可驱逐**（`_packed_keys` 打包为 INT64_MAX） | flashlib `slot_cache/triton/lru_ensure.py:94-107,371-441` |
| hybrid 路径的 `_ensure_experts_hybrid_kernel` 是**本仓库自己的** triton kernel，受害槽 argmin(usage) 且 active(owner) 排除 | `moe/offload_kernels.py:290-410` |
| prefill 两条变体都假设 host bank **按专家 id 全量**存放：overlap 用整层 bank 拷贝双缓冲（`buffer[buffer_id].copy_(per_layer[layer_id])`），非 overlap 用 `materialize_layer` 把整层写进 slots `[0,E)`（position == expert id 直通 GEMM） | `moe/offload_cache.py:668-701`、`layers/moe.py:347-384`、`moe/offload_kernels.py:249-285` |
| host bank 布局：`dict[role → list[L]]`，每层 `[num_experts, *row]` pinned 张量；专家字节数表 `_BANK_BYTES_PER_EXPERT` | `moe/expert_banks.py:35-37`、`moe/offload_cache.py:36-96` |
| 专家权重加载完成后**只读**（全库无运行期写路径） | Explore 检索报告 §8 |
| hybrid 的 CPU executor 通过 `data_ptr` 表直读 host banks | `moe/cpu_executor.py:335-401` |
| `lru_step` 在 kernel 入口自增（flashlib 与 hybrid kernel 均是 `step = load+1; store`） | `offload_kernels.py:329-330`、flashlib 同构 |

推论：
- **钉住机制可以零 flashlib 改动实现**：把钉住专家**附加到每层每次 `lru_ensure` 的 query** 里 → phase1 命中写 `usage[s]=step` → phase2 `_packed_keys` 判为不可驱逐 → 每次调用自我续期，永驻显存。
- **hybrid 路径可直接改 kernel**：受害槽扫描加"槽位 ≥ `cache_size - P` 则排除"的范围保护，更干净（无 query 膨胀）。
- 冷压缩 bank（host 只装冷专家）必须同步改造三处消费方：hybrid kernel 的 `src_indices`、`materialize_layer`（prefill 非 overlap）、CPU executor 的行映射；overlap 双缓冲路径 v1 不支持（断言拒绝），v2 再做三源组装。

## 2. 总体架构

```
NVMe checkpoint（真源，只读）
   │  加载期一次性分流
   ├─ 热专家（pin list，每层 top-K）──→ GPU slot cache 顶部固定区 [cache_size-P, cache_size)
   └─ 冷专家 ──────────────────────→ host bank（每层 [E-K, *row]，行号 = cold_row）
                                        │ 运行期按需 H2D（miss）
                                        ▼
        GPU slot cache LRU 区 [0(或2E), cache_size-P)
```

副本语义（按专家类别区分，修正表述）：

- **热专家（钉住）**：**显存单副本**——host tier 不装它、LRU 不管理它，这是本设计节省 host 内存的全部来源；
- **冷专家**：host 是唯一**权威驻留**；显存 LRU 区持有的是可丢弃的**运行期缓存副本**（沿用 FreeToken 原有 offload 语义）——即冷专家在被缓存期间**同时存在于显存与内存**，驱逐只是丢弃显存那份；
- **NVMe checkpoint**：全量只读真源，任何层级的副本丢失都可从它重建。

不变式是"**权威驻留唯一**"而非"物理单副本"：每个专家恰有一个权威所在（热专家=显存固定区，冷专家=host bank），其余出现均为可丢弃缓存。权重只读 ⇒ 任何方向的迁移/驱逐都无需写回：LRU 驱逐丢的是显存副本（host 仍在），降级/升级都是单向覆盖。

## 3. 组件设计

### 3.1 热度统计采集（`--hot-stats-out FILE`）

- **计数点**：`layers/moe.py` 的 `decode_forward` 与 `prefill_forward` 中，`fused_topk` 之后、`_decode_routed/_prefill_routed` 之前（此刻 `topk_ids` 还是原始专家 id）。`OffloadMoELayer` 是各层共享对象、`self.layer_id` 可用。
- **计数器**：新模块 `moe/hotness.py`，`ExpertHotness` 类持有 `counts [L*E] int64`（device）。开启时每个 MoE 层前向执行：
  `counts.index_add_(0, topk_ids.flatten() + layer_id * num_experts, ones_like)`。
  - 固定 shape、无同步 → **CUDA graph 可捕获**（whole-graph replay 时同一 op 重复累加，语义正确）。
  - 仅当 `hot_stats_out` 配置时启用；未配置时零开销（不进入前向路径）。
- **排空与落盘**：无每步同步。宿主侧周期钩子放在 `Scheduler._process_last_data`（每 loop 迭代调用，`scheduler.py:419-428`）：按墙钟时间间隔（默认 60s，`--hot-stats-interval-s`）做一次 `counts.cpu()` 累加到宿主数组（98 KB D2H，可忽略），**每次排空顺带原子写一次 JSON**（tmp + `os.replace`）——崩溃/SIGKILL 最多丢一个间隔的统计。`Scheduler.shutdown()` 在 CUDA 仍健康时显式 `save()` 收口（此时设备排空保证成功）；`atexit` 兜底时 device 排空是 best-effort（解释器退出阶段 CUDA 上下文可能已失效，失败则退回宿主累计照常写盘，实测该路径曾因强制 `counts.cpu()` 丢掉整场统计，见 6fb25ee）。
- **文件 schema**（统计 → 选点的契约）：
  ```json
  {"schema_version": 1,
   "meta": {"num_layers": 48, "num_experts": 512, "top_k": 10,
            "total_tokens": 12345678, "moe_strategy": "hybrid",
            "quant_format": "nvfp4", "model_path": "...",
            "duration_s": 3600.2, "created_at": "2026-09-27T18:00:00+08:00"},
   "counts": [[512 个 int], × 48]}
  ```

### 3.2 选点工具（`python -m freetoken.hotness select`）

- 输入：stats JSON；`--slots K`（每层钉 K 个）或 `--budget-gib G`（按 `_BANK_BYTES_PER_EXPERT[fmt]` 折算）；`--format`（默认 nvfp4）。
- 规则：每层按 count 降序取 top-K，平局取小 id。**每层独立**（路由偏斜逐层不同）。
- 输出 pin list JSON：
  ```json
  {"schema_version": 1, "num_layers": 48, "num_experts": 512, "per_layer_slots": 113,
   "pins": [{"layer": 0, "experts": [7, 331, ...]}, ...]}
  ```
- 同时打印覆盖率报告：每层 top-K 质量占比、全局加权命中率估计、零命中专家数——供人工判断 K 是否合理。

### 3.3 加载期钉住（核心）

新配置（`EngineConfig` + `args.py` 注册，沿用 moe_* 字段惯例）：
`hot_expert_list: str | None`、`hot_expert_slots: int | None`（与 list 二选一，直接吃 stats 文件内部选点）、`hot_stats_out: str | None`、`hot_stats_interval_s: float = 60.0`。

未配置任何 hot_* 时**行为与上游逐字节一致**（验收标准）。

#### 3.3.1 slot 布局

- 钉住区 = slot 顶部 `[cache_size - P, cache_size)`，`P = L × K`（每层 K 个，层 l 的第 j 个钉住专家 → slot `cache_size - P + l*K + j`）。
- 选顶部原因：prefill 双缓冲借用底部 `[0, 2E)`（`prefill_hit_compact` 阈值、`_invalidate_prefill_buffer`）；`materialize_layer` 写 `[0,E)`。顶部互不干扰。
- 约束（init 校验，错误信息明确）：
  - LRU 区剩余 `cache_size - P ≥ max(2*E, 512)`，否则拒绝（防 LRU 退化）；
  - `hot_expert_list` 的 K 超预算时截断并告警。
  - v1 曾要求 `prefill_overlap` 必须为 False；扩展二（§9）的三源组装已解除该互斥，钉住 + `--moe-prefill-overlap`/hit-d2d 为合法配置。

#### 3.3.2 初始化（`engine.py::_init_offload_moe_cache` 扩展）

1. 读 pin list → `pin_ids [L, K] int32`（GPU）、`pin_slots [L, K] int32`（GPU，= 顶部区地址）。
2. **预填映射**：`slot_for_id[l, e] = pin_slot`、`id_of_slot[pin_slot] = l*E + e`、`usage[pin_slot] = 0`。
3. **钉住权重入显存**：从 checkpoint safetensors 直接读 pinned 行（复用 `nvfp4_expert_spec`/`Nvfp4DiskIndex` 的行读取器或 `expert_pieces` 流式读取）→ 临时 pinned 暂存 → H2D 写入 slot cache 对应槽 → 释放暂存。**不经过 host bank**。
4. **冷压缩 host bank**：`build_expert_banks` 增加 `skip: dict[layer → set[int]]`；每层 bank 形状 `[E-K, *row]`，行号 = `cold_row`（按专家 id 升序压缩）。同时产出 `cold_row [L, E] int32` GPU 常驻表（pinned → -1）。
5. **显存预算记账**：钉住字节 = `P × bytes_per_expert`，计入 moe cache 预算内（`resolve_moe_cache_auto` 不变，`cache_size` 含钉住区；LRU 区 = `cache_size - P` 自然变小）。日志打印：钉住字节数、host 节省字节数、LRU 区大小。

#### 3.3.3 hybrid kernel 改造（`moe/offload_kernels.py`）

`_ensure_experts_hybrid_kernel` 增加参数 `pin_base: tl.constexpr`（= cache_size - P）、`cold_map_ptr`：

- Phase 2 受害槽排除：`u = tl.where(owner_active | (~c_mask) | (off_c >= pin_base), INT64_MAX, u)`。
- Phase 2 `src_indices` 改写冷行号：`e` 选出后 `src = tl.load(cold_map_ptr + base + e)`（取到的必为冷专家，≥0）；`slot_for_id/base+e` 写 victim 的逻辑不变。
- Phase 1/3 不变：pinned 专家 `slot_for_id` 已预填 → 走 hit → `out = pinned slot` → GEMM 直读顶部区。CPU executor 永远收不到 pinned 专家（hybrid 的 `on_gpu = topk_ids >= 0` 恒真）。
- **CPU executor 行映射**：`cpu_executor.py` 的 data_ptr 表按 `cold_row` 重排（每层传 cold bank + 映射，GEMV 按 expert→cold_row 取数）。

#### 3.3.4 offload/flashlib 路径（合并查询技巧，零 flashlib 改动）

`offload_kernels.ensure_experts` 当 `cache.pin_ids is not None`：

```
comb = pin_query_buffer[layer_id]        # [K_r + K, int32]，按 (layer, bs) 预分配
comb[:K_r] = expert_ids; comb[K_r:] = pin_ids[layer_id]
lru_ensure(comb, slot_for_id.view(-1), id_of_slot, usage, step, comb, src, dst, n, id_base=...)
expert_ids.copy_(comb[:K_r])             # 写回 slot id
```

- pinned 每次都被 query → phase1 命中 → `usage=step` → 本次不可驱逐；逐调用续期 ⇒ 永驻。
- pinned 是命中 ⇒ 不产生 copy plan 条目；即使某冷专家与 pinned 重复，phase1 去重折叠，两个位置都写同一 slot。
- `src_indices` 是"专家 id（层内）"语义，而 bank 已冷压缩 ⇒ 在 `copy_missing` 前插一个固定 shape 的 remap kernel：`src = cold_row[layer, src]`（num_indices 之外的垃圾项 remap 无害，fused copy 按设备侧长度读取）。
- prefill 非 overlap 的 `materialize_layer` 改造（我们的 kernel）：冷专家照旧（slot = 专家 id，`usage=step`）；**pinned 专家跳过**（不写 id_of_slot/slot_for_id/usage，不产生 copy 计划项）；随后 prefill GEMM 前把 `topk_ids` 经 `slot_for_id` 行 gather 成 slot 索引，`views` 用全量 slot cache、`n=None`（与 decode 同形）。`num_indices = E - K`，src 经 remap。

#### 3.3.5 与既有机制的交互

| 机制 | 交互 | 处理 |
|---|---|---|
| prefill overlap 双缓冲 | 与冷压缩 bank 曾不兼容 | 扩展二三源组装已解除：miss 经 cold_row remap 进 batch、钉住行经顶部槽 D2D gather / 组合填充（bank 冷行 + cache 顶部槽 + misses 三源） |
| `_invalidate_prefill_buffer` | 只写 `[0,2E)` | 顶部区无碰撞 |
| `_reset_cache_kernel`（rebuild/reset） | 清空全部映射 | v1：带 pin 时 `rebuild`/reset 后重钉（调 init 的预填步骤）；至少断言并告警 |
| `decode_log_interval`/`decode_miss_stats` | 统计语义 | miss 统计不含 pinned（它们恒命中），报告里注明 |
| `moe_cache_auto` | 预算 | 钉住字节在预算内，自动解算自然缩小 LRU |
| `--moe-cache-size` 显式值 | 用户自管 | 文档提示需包含钉住区 |

## 4. 文件级改动清单

| 文件 | 改动 |
|---|---|
| `moe/hotness.py`（新） | `ExpertHotness` 计数器 + 排空 + JSON 读写 + schema 常量 |
| `hotness/__init__.py`、`hotness/select.py`（新） | 选点工具（纯 python，CPU 可测） |
| `layers/moe.py` | decode/prefill 计数插桩；非 overlap prefill 的 slot 重映射 GEMM |
| `moe/offload_kernels.py` | hybrid kernel 范围保护 + src 冷行 remap；`ensure_experts` 合并查询；`materialize_layer` pinned 跳过；remap kernel |
| `moe/offload_cache.py` | pin 状态字段（pin_ids/pin_slots/cold_row/pin_query_buffer）、init 预填、`copy_missing` remap 接入、断言 |
| `moe/expert_banks.py` | `skip` 冷压缩构建 |
| `moe/cpu_executor.py` | data_ptr 表按 cold_row 重排 |
| `engine/engine.py` | `_init_offload_moe_cache` 钉住编排（读 list、入显存、预算记账、日志） |
| `engine/config.py` + `server/args.py` | 四个新字段/flag |
| `tests/moe/test_hotness.py`、`tests/hotness/test_select.py`、`tests/moe/test_hot_pin.py`（新） | 见 §6 |

## 5. 真实采集工作流（交付后用户执行）

```bash
# 1) 采集（NVFP4，hybrid，代码类负载：claude-code/codex 实战数小时）
ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 --moe-strategy hybrid \
  --hot-stats-out ~/hotstats/code-$(date +%m%d).json --memory-ratio 0.95 ...
# 2) 选点（看覆盖率曲线决定 K）
python -m freetoken.hotness select ~/hotstats/code-0927.json --budget-gib 24 -o pins.json
# 3) 钉住运行（对比基线：tok/s、miss 率、host RSS）
ft serve ... --hot-expert-list pins.json
```

## 6. 测试与验收

1. **纯 CPU 单测**（模仿 `tests/moe/test_offload.py` 的 CPU cache 惯例）：
   - 选点：top-K 排序/平局/覆盖率/预算折算；
   - stats schema 读写往返；
   - 合并查询 ensure：CPU 镜像（`_ensure_experts_hybrid_cpu` 同款）验证 pinned 永不被驱逐、冷专家 LRU 正常；
   - 冷行 remap 的正确性（含 pinned→-1）。
2. **GPU 单测**（skipif no CUDA）：真实 `lru_ensure` + 合并查询，反复随机 query 数千步后断言 pinned 槽 `id_of_slot` 不变、冷专家命中正常；hybrid kernel 范围保护同断言。
3. **不变式回归**：无 hot_* 参数时全部现有测试不变绿→红。
4. **冒烟**：`--dummy-weight` + 合成 pin list 启动 qwen4_exp（NVFP4 路径），验证日志输出钉住字节/LRU 区、host RSS 低于全量、一次 chat completion 正常；随后真实 NVFP4 checkpoint 短负载。
5. **显存/RAM 记账断言**：加载完成后 host bank 总字节 == 冷专家字节（对照 pin list 计算）。

## 7. 风险与后续（v2）

- 磁盘冷层（#337 式）串联成三级：热集进一步释放 host；
- flashlib 上游化：给 `lru_ensure` 提 range-exclusion 提案，去掉合并查询的 query 膨胀。

## 8. 扩展一：FP8 CPU executor（FP8 获得 hybrid 能力）

**目标**：`fp8_block` 格式进入 CPU executor 的 `WFmt` 集合，解除 #534（`cpu_format=None` → offload 解析到 CPU 解码即崩）的限制，使 FP8 也能 `--moe-strategy hybrid`（PCIe 取数 + CPU 溢出分流）。

**改动面**：
- `csrc/cpu_moe/cpu_moe_ext.cpp`：`WFmt` 新增 `fp8_block`。GEMV 内核：权重 e4m3 字节（gate_up `[E,2I,H]`、down `[E,H,I]`）→ 256 项 LUT（进程级一次构建，float）解量化 → fp32 FMA；块 scale bf16（`[E, 2I/128, H/128]`、`[E, H/128, I/128]`，行宽经 `fp8_block_scale_pad` 填充，取数时按 pad 步长读）。累加结构：每输出行按 K 方向分块，块内 FMA、块尾乘 scale 后并入——与 GPU triton fp8_block kernel 的解量化语义一致（无激活量化，fp32 计算 bf16 输出）。
- `moe/cpu_executor.py`：`_WFMT_IDS` 注册 `"fp8_block"`；`_resolve_banks` 增加分支（按 role 给出 weights/scales 指针与 (H, I)）；cold_row 重排对本格式同样生效。
- **#534 修复**：`layers/quantization/moe/fp8_block.py` 为 `TritonFp8BlockMoEKernel` 声明 CPU 执行格式（cpu_format 绑定到新 WFmt），使策略解析在"显存不足 → offload/hybrid"时不再抛 `KernelSelectionError`；同时核对策略解析处（engine/config 的 decode_target 解析）允许 `fp8_block + hybrid`。
- **测试**：CPU 单测——随机小矩阵（H=256, I=64, 128×128 块对齐）上 fp8 GEMV 对照 float64 解量化参考（容差按 bf16 输出量化），含 padded scale 步长用例；参数化覆盖 avx2 路径。

**验收**：本机真实 FP8 checkpoint（`Qwen3.6-35B-A3B-...-fp8` 或 `Qwen3.8-Flash-Next-FP8`）以 `--moe-strategy hybrid` 启动并完成一次 chat completion（此前 #534 形态直接崩溃）；吞吐数据记录入报告。

**性能预期（写进报告，不是验收线）**：FP8 每权重 1 字节，CPU GEMV 带宽受限——每 token 活跃字节 2.2 GiB（NVFP4 1.24 GiB），50 GB/s 内存下 CPU 侧理论上限 ~22 tok/s。定位是"8-bit 质量档"，速度不与 NVFP4 竞争。

## 9. 扩展二：prefill overlap 三源组装（解除 v1 的 overlap 关闭约束）

**目标**：钉住与 `--moe-prefill-hit-d2d`/双缓冲共存，恢复 prefill 吞吐。三源 = 冷专家的 bank 行、钉住专家的顶部槽位、LRU 命中槽位。

**改动面**（全部在 `moe/offload_cache.py` 的双缓冲族 + `moe/offload_kernels.py::prefill_hit_compact`）：
- `prefill_hit_compact` 语义已天然兼容：阈值 `slot >= 2*num_experts` 只排除双缓冲借用区，钉住槽（顶部）与 LRU 命中槽都会被收集为 hit → gather 进 buffer。无需改动或仅加注释。
- `_prefetch_split` 的 **miss 侧**：宿主 run-list 从快照构建后，专家 id → `cold_row` remap（快照 host 数学内完成，miss 集合不含 pinned）。
- **整层拷贝回退路径**（`copy()` 的 `buffer.copy_(per_layer[layer_id])`，hit-d2d 不可用时）：改为组合填充——冷行用 `index_select(bank, cold_row_table)`、钉住行用 fast_index_copy 从顶部槽 gather，两次固定 shape 拷贝；或统一走 gather 内核。实现者按最小 diff 选择，验收以数值一致为准。
- **小行宽 bank**（< 256KB）在 hit-d2d 路径既不进 batch 也不进命中 gather：其整行集合（冷行自 bank + 钉住行自顶部槽）由同一组合填充覆盖，避免 batch 混入子 256KB 条目而整体退化为同步拷贝（实现时确定的关键细节：小 bank 的整层 batch 条目在冷压缩下行数不足 E，无法保留）。
- 解除 §3.3.1 的 `prefill_overlap=False` 断言：钉住与 overlap 同时启用成为合法配置；`_hit_d2d_usable` 的各回退原因补一条钉住相关检查（顶部区必须存在）。
- 快照一致性：`begin_prefill` 的 snapshot 已 fence 在前一次 decode 之后，钉住映射在 chunk 之间不变（动态重钉只在 idle 安全点发生，见 §10），无需额外同步，但需在注释中写明该前提。

**验收**：dummy-weight 下同一批 prefill 请求，钉住+overlap 与 钉住+无overlap 与 无钉住基线 三者输出逐 token 一致（确定性内核）；真实 NVFP4 短负载 prefill 吞吐对比入报告。

## 10. 扩展三：动态重钉（热集随负载演进）

**目标**：运行期按滑动窗口热度自动调整热集成员，域漂移（代码 ↔ 长上下文 agent）时热集不失效。

**机制**：
- **窗口统计**：钉住模式下 `ExpertHotness` 常开（不再仅 `--hot-stats-out` 时）。双缓冲计数：每次墙钟排空（间隔 = min(落盘间隔, T)）把 device `counts` 增量 D2H 后清零，增量先并入"当前窗口"宿主累计器（device 部分 + 已排空部分合起来才是完整窗口，排空可横跨窗口边界）；窗口边界到达即封口出"完整窗口"，上一窗口保留为参照；EMA（半衰期=窗口长，每窗衰减 0.5，首窗直取）作为平滑热度，提供 `window_topk` 与 `ema_kth`（第 K 名计数）查询。`--hot-stats-out` 的全量累计仍吃同一份增量，语义不变。
- **重钉决策**（宿主侧纯计算，`--hot-expert-repin-interval-s` 默认 0=关闭，>0 时钉住模式下计数器自动常开）：每层比较 EMA top-K 与当前钉住集；仅当"候选 ∈ 当前冷集 且 EMA(候选) ≥ EMA(被替换者) × `--hot-expert-repin-gain`"（默认 1.5，迟滞防抖；EMA=0 的候选不换）才交换；每周期每层最多换 `--hot-expert-repin-max-swaps`（默认 8）个；候选/受害排序平局取小 id。
- **迁移执行**（行级交换，每对 ≤ 4.69 MiB，`OffloadMoeCache.swap_pinned_experts`）：
  1. 新热冷专家 c（bank 行 r_c）、被替换钉住专家 h（槽 s_h）；
  2. `s_h --D2H--> host scratch`；`bank[r_c] --H2D--> s_h`（c 上位——必须先于写回，写回会覆盖 c 的原始字节）；`scratch --> bank[r_c]`（h 回填 host，全库唯一运行期 bank 写者）；
  3. 更新映射：`pin_ids/cold_row/slot_for_id/id_of_slot`（usage 刷成当前 step 衔接 flashlib 不可驱逐语义），并刷新三源组装的预建 gather 索引与 cold_row 宿主镜像；全部为既有张量的值改写，shape 不变 → **与 CUDA 图兼容**，flashlib 合并查询读的 `pin_ids` 缓冲同理。
- **执行点**：`Scheduler` 的 idle 安全点（`_execute_pending_rebuild` 先例，scheduler.py:230-233）——所有流同步、无在途 GEMM/CPU GEMV 时做交换；由此 **host bank 从"加载后只读"变为"仅在重钉安全点可写"**，需在 `host_banks.py`/注释与并发假设中显式记录该约定。
- **与 §8/§9 的交互**：重钉只交换行身份，不改任何 shape/指针基址；CPU executor 的 data_ptr 表不变（bank 首址与行宽不动，变的只是行内容与 cold_row 值）——这是冷行号间接层带来的额外红利。
- **可观测**：每次重钉打印迁移对（layer, 换入, 换出, 计数比）汇总行；`--hot-stats-out` 落盘时附当前钉住集。

**验收**：CPU 单测（重钉决策：迟滞/上限/平局；映射交换不变式：slot_for_id 与 cold_row 互为逆映射、pin 槽无重复）+ GPU 单测（合成负载 A→B 域切换，T 调小，断言热集按窗口迁移且迁移期间一次 decode 输出与基线一致）。

## 11. 实施顺序（全部纳入本轮）

```
C 钉住主功能 ──→ E 三源组装（解除 overlap 约束）
      ├──→ D FP8 CPU executor（与 E 并行，文件面不相交）
      └──→ F 动态重钉（在 E 之后，共享 offload_cache 状态字段）
```
每步交付即跑测试；D/E/F 各自独立成 commit。

## 12. 扩展四：运行中调参（--tune-file 的 fetch_fraction 与 pin_k）

**目标**：服务运行中（decode 图已捕获、无法重捕获）不改代码地调 hybrid 拉取比例与每层活跃钉住数 K。

### 12.1 值更新图安全机制（Phase B 铺垫）

CUDA graph 捕获冻结的是内核标量实参的**值**、张量实参的**指针**。因此运行时可变的参数一律走"cache 持有、指针稳定、只改值"的设备张量：`fetch_params`（[2] int32：cap + Q16 定点比例）与 `pin_base_dev`（int32 标量，钉住排除区边界）。已捕获 decode 图在下一次 replay 即读到新值。

### 12.2 固定容量布局 + 活跃计数动态

钉住区从"每层前缀偏移的可变布局"改为**每层固定容量槽位**：

- `pin_base = cache_size - L × K_cap`，层 l 的容量槽 = `[pin_base + l*K_cap, pin_base + (l+1)*K_cap)`，静态不变（rebuild 重解算时同步刷 `pin_base_dev`）；
- `pin_ids [L, K_cap]`：前 `pin_counts[l]` 个为活跃钉住专家，尾部按**首钉 dup** 约定填充（合并查询每次拷贝整条容量行，填充项重复命中首钉的钉住槽——无 fetch、无越界、无需内核感知每层计数）；
- 活跃 K 变化只改前缀长度与映射值（shape 不变 ⇒ 图安全）：**扩** = 冷专家 H2D 装入该层第 count 个槽（`install_pinned_experts`，同时解除其 LRU 残留副本槽的 id 映射）；**缩** = 尾部活跃钉住 D2H 换出到宿主空闲行（`unpin_tail_experts`），空槽盖 `_PIN_USAGE_SENTINEL`（2^62，flashlib 全 cache argmin 永不选中；hybrid 内核按 `off_c >= pin_base` 整段排除，不依赖该值）；
- 已知取舍：缩 K 留空的容量槽不参与 LRU（在排除区内）；
- 每层动态下界 `pin_floors[l]` = 加载期钉住数：宿主冷压缩 bank 行数 E - floors[l] 固定，缩 K 换出的行取自空闲行池（深度 = K_active - floors[l]），因此 **K_active ∈ [floors[l], K_cap]**。

### 12.3 --tune-file 键语义

| 键 | 生效时机 | 途径 |
|---|---|---|
| `fetch_fraction`（[0,1] 数值） | 即时（下一次 decode replay） | poller 同步栅栏后 `cache.set_fetch_params`（cap 不变） |
| `pin_k`（整数） | 下一 idle 安全点 | poller 经 `cache.repin_manager.set_target_k` 记录**最新目标**（写多次只保留最新），scheduler 的 idle 点由 `HotExpertRepinManager.apply_target_k` 执行扩缩，先于常规 EMA 换血 |

pin_k 应用规则：每层目标 = clamp(k, floors[l], K_cap)（CPU 解码层 floors=0 自然不钉）；扩容按窗口 EMA 选点（EMA 降序、平局小 id、跳过已钉与零热度），候选不足时保留目标静默等下个 idle；缩容免窗口。无管理器（未启用动态重钉）时 pin_k 打日志忽略；非整数/越界告警忽略。

### 12.4 --hot-expert-slots 与容量

- stats 模式（--hot-expert-list 指向热度统计 JSON）：slots = 加载期选点数，容量同值 ⇒ 静态；
- **pin list 模式**（--hot-expert-list 指向 pin list JSON）：list 定初始 K（= floors），slots = 容量 K_cap，为运行中调 K 预留扩容空间（须 >= list 每层钉住数）；
- 不传 slots：K_cap = 初始 pin_counts 的 max，行为与固定 K 完全一致（向后兼容）。

### 12.5 守卫

- 容量显式值 < 初始每层钉住数 → 拒绝；
- K_cap > E → 拒绝；
- `cache_size - L×K_cap >= max(2E, 512)`（LRU 地板；rebuild 同式校验）。

**验收**：几何/管理器 CPU 单测（映射不变式、空闲行池、哨兵、floor clamp、窗口门控）；GPU 图安全测试（捕获带钉 ensure_experts_hybrid → idle 扩容 → replay 新钉恒命中、旧钉不变、残留副本槽去映射；缩容 replay 按恢复冷行换入）；tune 文件集成（pin_k → 管理器目标转发、坏值告警、无管理器旧行为）。
