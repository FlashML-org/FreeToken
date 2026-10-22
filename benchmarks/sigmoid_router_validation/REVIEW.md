# Sigmoid router：深化验证与审阅入口

> 最新 main 整合复验：121 项路由测试通过；分支/main 相关回归均 92 通过、1 跳过；四组消融重跑完成。见 [本轮报告](main-integration/README.md)。下文保留历史结果。

> 2026-10-22 WSL 后续：Linux 环境与 FreeToken 编译成功，117 项局部测试和 46 项相关回归通过；真实权重路由复验完成。见 [Linux 验证记录](linux/README.md)。下文的 Windows 实验保留原始环境与数值；完整模型 A/B 仍未完成。


更新：2026-10-22。状态：本地补丁，未 commit/push，未创建 PR。本文记录已执行的深化工作，以及当前条件下不能完成的项目。

## 1. Problem & Motivation

[Issue #730](https://github.com/FlashML-org/FreeToken/issues/730) 提议将五类 MoE 的 sigmoid 路由融合，减少小算子启动与逐步权重转换开销。目标是降低真实推理成本；微基准的加速比不能直接转换成生成 tokens/s。

该 Issue 的原始测量使用 RTX PRO 6000 与量化 checkpoint；本实验使用 RTX 4060 Laptop、BF16 前缀。因此没有直接比较其吞吐目标，也不以本机数据声称完成了该目标。

## 2. Implementation

共享内核 [sigmoid_router.py](../../python/freetoken/kernel/triton/sigmoid_router.py) 接受 `[tokens, experts]` logits、每专家偏置、top-k、分组参数、归一化/缩放参数及可选设备有效行数，输出连续 FP32 权重与 int32 专家编号，形状均为 `[tokens, topk]`。

每个 token 一个 Triton program，在片上完成 sigmoid、偏置选取、按每组最高两项之和选组、反复取最大值选专家、提取未加偏置的权重、归一化和缩放。选择 Triton 是为了沿用项目现有融合路由实现方式；反复最大值选择便于核对同分规则，适合目前较小的 top-k，但约有 O(kE) 的选择工作量，不能推断适合任意大 k。

输入按 stride 寻址，因此支持非连续 logits/bias；对非二次幂维度使用掩码。有效行数从 GPU 标量读取，不用 `.item()`，可在 CUDA Graph 重放间变化；无效行写入权重 0、编号 -1。推理场景限定有限 logits/bias，不提供反向传播实现。

调用位置在 gate 矩阵乘之后、专家计算之前。五个模型的 `_route` 已接入；共享专家依旧先于可能覆写输入的路由专家计算。

| 模型实现 | gate 精度处理 | 路由规则 |
|---|---|---|
| [glm4_moe](../../python/freetoken/models/glm4_moe/moe.py) | 原加载舍入后常驻 FP32 | 可选分组、epsilon 1e-20、原缩放 |
| [glm_moe_dsa](../../python/freetoken/models/glm_moe_dsa/moe.py) | 同上 | 同上 |
| [glm5_next](../../python/freetoken/models/glm5_next/moe.py) | 同上 | 同上 |
| [minimax_m2](../../python/freetoken/models/minimax_m2/moe.py) | 保留原 BF16 gate 路径；没有原 GLM 式的逐步 FP32 权重转换 | epsilon 0，不新增缩放 |
| [minimax_m3](../../python/freetoken/models/minimax_m3/moe.py) | 原实现已经常驻 FP32，不重复改动 | epsilon 1e-20、原缩放 |

三个 GLM 使用 [Fp32RouterLinear](../../python/freetoken/layers/linear.py)。不把 M2 强行改成 FP32，是为了避免这次融合额外改变 gate 计算精度；不表示它未经测试就满足任何新的 FP32 精度要求。

**Padding 的范围：** 内核的 mask 功能和 Graph 重放已测试，五模型调用目前没有传入有效行数。现有 softmax MoE 调用也未传入该可选参数。本补丁不能据此声称完成了调度器到模型的 padding 集成；完整服务的 Graph/padding 行为仍待验证。

## 3. Correctness Validation

本轮扩展测试后 **117 项通过**，新增覆盖关闭归一化、M2 的 BF16 logits/bias，以及经父级 `load_state_dict` 加载后 FP32 gate 的 Graph 重放、输入变化和权重地址稳定性。见 [测试日志](deepening/targeted-tests.log) 与 [模型测试源码](../../tests/models/test_sigmoid_routing.py)。这些仍是局部测试，不是完整模型加载测试。

此前回归的三个 FTW 失败，均在未修改基线上复现，原因是 Windows 缺少 `mmap.PROT_READ`。原始对照：[补丁日志](history/related-regression-tests.log)、[基线日志](history/baseline-ftw-tests.log)。没有为消除这些失败引入无关平台修复。

### 已有四行顺序差异的复查

用原种子 20261020、原三层权重、相同形状和输入生成顺序重放 2,385 行，复现恰好四行：layer 1 / std 3 / batch 256 的行 73、166、179，以及 layer 45 / std 0.1 / batch 256 的行 86（均为零起始索引）。

四行交换位置的专家，在 PyTorch FP32 的 `sigmoid(logit)+bias` 中分数**精确相等**。四行专家集合均相同，且第 8 与第 9 名的分数间隔均大于零。说明这次观测是入选集合内部的同分排序差异，不是观测到了边界换专家。详细专家编号、分数和边界间隔见 [tie-audit.json](deepening/tie-audit.json)。

这不能证明所有排序差异都无害：权重与专家配对正确时，数学上的加权和不变，但浮点归约不满足结合律，下游按槽位累加时仍可能发生舍入变化。这里没有执行四行对应的完整专家后端，不能宣称其输出逐位相同。

### 已知语义差异，单独列示

- 新实现对精确同分选择较小专家/组编号；不能保证与 PyTorch 的任意同分顺序一致。
- 若同分跨越 top-k 或选组边界，候选集合理论上也可能不同；近同分还可能受 sigmoid/归约舍入影响。
- [构造的边界同分例子](deepening/boundary-tie-example.json)在本机观测到顺序不同、集合相同；不将其写成已观测到集合不同的证据。
- 先前真实前缀的三个末 token 层输出一致，只覆盖那三个样本，不能否定以上风险。

## 4. Performance Evaluation

新增 2×2 消融使用 GLM-4.5-Air layer 1 的真实 gate、真实数学提示词的隐藏状态；测量包含输入转换、gate 矩阵乘和路由，排除专家计算。batch=1 使用提示词最后一个输入 token，不是完整自回归 decode。

| token 数 | 原实现 | 仅融合 | 仅 FP32 常驻 | 两项同时启用 |
|---:|---:|---:|---:|---:|
| 1 | 26.820 μs | 13.894 μs | 20.737 μs | 8.089 μs |
| 93 | 55.459 μs | 36.569 μs | 49.638 μs | 30.513 μs |

两项同时启用的完整路由局部加速约为 3.32× / 1.82×。数据支持两项均有贡献，其中融合节省更多；不能把两个单项加速倍数相加，也不能推出整模型收益。

每组轮换测量顺序，共五次 CUDA Graph 计时，TF32 关闭。另用 PyTorch profiler 捕获单次 eager 调用，GPU 执行事件分别为：

| token 数 | 原实现 | 仅融合 | 仅 FP32 常驻 | 两项同时启用 |
|---:|---:|---:|---:|---:|
| 1 | 13 | 4 | 12 | 3 |
| 93 | 14 | 5 | 13 | 4 |

捕获的事件包含转换 kernel、矩阵乘及路由 kernel；详细名称已保存。这不是 CUDA Graph 的主机 launch 次数，也不是整个模型每 token 的事件数。93-token 矩阵乘本身包含两个 GPU kernel。

完整原始数据见 [ablation.json](deepening/ablation.json)、[执行日志](deepening/ablation-run.log)。[TIMINGS.md](deepening/TIMINGS.md)同时提供已有 20 组微基准和本轮消融的中位数、最小/最大值、样本标准差；[CSV](deepening/microbenchmark-distribution.csv)可直接分析。五次测量的分布不是统计置信区间。

首次消融复用了省略 scale=1 乘法的简化参考路径。逐行核对旧代码后已恢复该操作并重新运行；`ablation-simplified-baseline-superseded.json` 与 `first-attempt.log` 只保留作历史，本文数字全部使用修正后的基线。

## 5. Reproducibility

原始代码基线：`cc56b6bdcf9188eb632197d6ec719c61bff11798`。本次深化没有重新拉取或移动到更新的 main，因此这些不是对最新 main 的端到端数据。提交前需在选定 main 上重验。

硬件：RTX 4060 Laptop 8 GiB；Windows 11；Python 3.12.15、PyTorch 2.14.1+cu130、Triton Windows 3.8.0.post29。完整依赖见 [环境清单](environment-current.txt)。权重版本：`zai-org/GLM-4.5-Air@a24ceef6ce4f3536971efe9b778bdaa1bab18daa`。

从仓库根目录，用已配置的 CUDA Python 执行（PowerShell）：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'python'
python -m pytest tests/kernels/test_sigmoid_router.py tests/models/test_sigmoid_routing.py -q -ra --tb=short
python benchmarks/sigmoid_router_validation/deepen_validation.py --router-cache D:/AI/FreeToken-validation/GLM-4.5-Air-router --prefix-cache D:/AI/FreeToken-validation/GLM-4.5-Air-prefix --output benchmarks/sigmoid_router_validation/deepening
python benchmarks/sigmoid_router_validation/summarize_deepening.py
```

缓存路径可替换；深化脚本读取此前固定版本的权重切片和激活，不进行下载。首次获取前缀缓存见 [README](README.md)，三层 gate 权重的下载来源与哈希见 [原实验 JSON](history/real-glm-router-results.json)，原下载脚本保留在 `history/test_real_glm_router.py`，需要按其说明调整本机路径。原始结果与本轮新结果分别存放，没有覆盖历史测试计数。

核心改动为上述一个 kernel、一个 linear 类及五个模型接入；测试与微基准在 `tests/`、`benchmarks/`，无专家后端改动。相关源码校验见 [本轮清单](deepening/source-manifest.json)，证据文件校验见 [checksums.json](checksums.json)。

## 6. Limitations & 下一步条件

| 项目 | 目前判断 | 解除条件 |
|---|---|---|
| 同分调查、原始时延分布、2×2 消融、GPU 事件计数 | 本机已完成，结果如上 | 外部审阅与其他 GPU 复验 |
| 完整 FreeToken 加载与多步生成 | 当前未完成；局部数据不能替代 | Linux 运行环境及能容纳受影响 checkpoint 的内存配置 |
| tokens/s、TTFT、峰值显存、完整质量 | 当前无结果 | 同硬件、同权重、同设置的 main/branch A/B；保留输出和完整日志 |
| 其他四类模型真实权重与多 GPU/TP | 未验证 | 对应权重与硬件；不因接口测试通过就宣称全支持 |
| padding 服务路径、Graph 下真实专家后端 | 内核能力已有，服务行为未验证 | 实际调度、padding、专家后端联合测试 |
| Issue 指定硬件上的吞吐目标 | 无法用本机确认 | 相同或可解释的目标环境；不能按微基准比例外推 |
| PR 提交与评审 | 按用户要求尚未进行，不是硬件阻塞 | 用户决定提交时再操作；本文不构成提交授权 |

“当前无法”不等于永久不可行。现选 BF16 模型超出本机容量；低位量化或其他受影响的小模型是否可行，仍需在具备 Linux 环境后核对精确 checkpoint 和加载方式。没有验证这些替代方案前，不承诺安装 WSL 就一定能跑通，也不把申请云资源作为已经完成的步骤。

后续最高价值工作仍是一次真实引擎的完整 A/B，而不是继续增加类似的随机路由样本。现有深化已补上本机可测的归因与数值解释，但系统级有效性仍未证明。
