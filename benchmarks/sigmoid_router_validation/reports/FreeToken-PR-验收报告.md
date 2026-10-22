# FreeToken #730：本机实验验收报告

> 2026-10-22 深化结果：扩展测试 117 项通过，完成四行同分追查与完整路由 2×2 消融、profiler 事件计数。详见[深化验证与审阅说明](../REVIEW.md)。下文保留前一轮结果。

> WSL 后续结果：CUDA 运算、pinned memory、FreeToken 编译通过；Linux 下 117 项局部测试与 46 项相关回归全部通过。Linux 完整路由局部消融为单行 33.293 → 9.313 μs、93 行 72.042 → 38.172 μs。详见 [Linux 验证记录](../linux/README.md)。以下 Windows 历史记录不代表最新 Linux 状态；完整模型 A/B 仍未完成。

日期：2026-10-21。状态：已完成下述本机局部验证；完整模型端到端性能与质量尚未验证。未 commit、push 或创建 PR。

## 1. 解决的问题

五个模型的 sigmoid 路由原来由多次 PyTorch 操作完成。共享 Triton 内核把 sigmoid、偏置排名、可选分组、top-k、取原始分数、归一化与缩放合并，减少 GPU 启动和中间张量。三个 GLM 模型还把原本每次转换的路由权重改为加载后常驻 FP32。

偏置只用于选专家，不进入最终混合权重。MiniMax-M2 的零 epsilon、各模型的缩放和分组规则均保留。此补丁没有改变模型结构或专家参数。

## 2. 环境与范围

- Windows 11、RTX 4060 Laptop GPU 8 GiB、驱动 616.92、Python 3.12.15。
- PyTorch 2.14.1+cu130，Triton Windows 3.8.0.post29（模块版本 3.8.0）。项目正式支持环境为 Linux；本次不能代替 Linux 验证。
- 工作基线 `cc56b6bdcf9188eb632197d6ec719c61bff11798`；本次检查远端 main 为 `fe5a5ec9c5644e266527de5afb9f1156b9bdcabb`，所检查的模型路由和 linear 目标文件没有上游变更。尚未移动工作基线。
- 真实权重来自 `zai-org/GLM-4.5-Air`，固定 revision `a24ceef6ce4f3536971efe9b778bdaa1bab18daa`。

## 3. 测试结果

| 检查 | 结果 | 证据与边界 |
|---|---|---|
| 新增内核与五模型路由测试 | 111 通过 | 标准仓库 pytest 入口和 fixture；含 CUDA Graph、设备有效行数、分组、同分、近同分、非连续输入等 |
| 相邻功能回归 | 43 通过、3 失败 | DeepSeek 路由、GLM5 配置、MiniMax-M3、FTW 加载器 |
| 未修改基线 FTW 对照 | 1 通过、同样 3 失败 | 同为 `mmap.PROT_READ` 在 Windows 不存在；不是本次新增回归 |
| 真实三层 gate 权重＋合成隐藏状态 | 27 组、2,385 行 | 专家集合全部一致，4 行顺序不同；非真实完整前向 |
| 真实提示词＋真实模型前缀 | 3 条、137 个 token | 第一处 MoE 的专家集合和顺序全部一致 |
| 真实专家权重输出对照 | 3 个最终输入 token | 第一处 MoE 加共享专家及残差后的 BF16 输出逐位一致 |
| 扩展微基准 | 20 组均通过数值断言并加速 | 1/8/256/1,024/4,096 token；只测 logits 之后的路由 |

日志：`validation/standard-fixtures-tests.log`、`validation/final-targeted-tests.log`、`validation/related-regression-tests.log`、`validation/baseline-ftw-tests.log`。

### 真实提示词测试究竟做了什么

使用固定版本的官方 tokenizer/chat template、真实词嵌入行、完整第 0 层、以及第 1 层 attention/norm，得到第一处 MoE 的真实输入。前缀由已安装 Transformers 的 GLM 实现执行；路由调用补丁的 FreeToken 实现，并分别对比原 PyTorch 表达式和 HF 路由。

| 提示词 | token 数（含模板） | 选专家/顺序差异 | 最大权重绝对误差（对原实现） | 最后 token 层输出最大误差 |
|---|---:|---:|---:|---:|
| 水箱 A/B 进水、C 排水的数学题 | 93 | 0 / 0 | 5.96e-8 | 0 |
| Python 返回首个不重复字符并解释复杂度 | 26 | 0 / 0 | 2.98e-8 | 0 |
| 解释蓝天和红色日落 | 18 | 0 / 0 | 5.96e-8 | 0 |

每条提示词只执行最后一个输入 token 的入选专家，用真实 gate/up/down 权重计算，再以参考 FP32 加权求和、BF16 输出组合。没有执行 FreeToken 的实际专家后端，没有计算后续 44 层、LM head 或生成任何回答。因此不能宣称“回答正确率不变”或“完整模型输出一致”。

额外 HF 对照中，137 行的专家集合也完全一致，但排列顺序全部不同；安装的 HF 实现使用 `topk(sorted=False)`。按专家编号对齐后的权重最大误差仍为 5.96e-8。上表的顺序一致专指对比 FreeToken 原实现，不代表与 HF 的排列逐位相同。

脚本：`validation/real_prompt_prefix.py`、`validation/checkpoint_slices.py`。结果：`validation/real-prompt-results.json`。权重按 HTTP Range 精确获取，缓存与 SHA256 清单、实际激活保存在 `D:/AI/FreeToken-validation/GLM-4.5-Air-prefix`，没有下载整个模型。

### 性能

每组先校验专家编号与权重，再交替进行旧/新路径各 5 次 CUDA Graph 计时，报告中位数，保存全部样本。20 组路由加速为 **2.30–8.81 倍**；4,096 token 下仍为 **2.30–3.27 倍**，但优势随批次增大而下降。

示例：单 token、192 专家分组路由从 33.623 μs 降到 4.004 μs；4,096 token 同配置从 547.162 μs 降到 167.105 μs。

这些数字排除了 gate 矩阵乘、权重转换、专家计算、调度及通信，不能写成“生成速度快 8 倍”。Windows 桌面上的时延会受系统负载、频率和温度影响。原始数据见 `validation/extended-benchmark.json`。

## 4. 会不会损害质量

目前没有在上述真实前缀样本中发现专家选择变化，证据支持通常输入下的局部数值一致性。但还不能保证完整质量不变：

1. 精确同分时新内核取较小编号，PyTorch topk 的同分顺序没有稳定保证；遇到 top-k 边界同分，可能选不同专家。
2. sigmoid 和归约的浮点舍入略有差异，边界近同分也可能改变选择；深层累积及自回归放大未测。
3. 只用了一个模型、三个短提示词、第一处 MoE；其余模型只有接口/合成输入测试，没有真实 checkpoint 完整推理。
4. FP32 常驻增加路由显存，相比 BF16 每层多 `专家数 × 隐藏维度 × 2` 字节；GLM-4.5-Air 的 45 个 MoE 层合计约 45 MiB。端到端显存峰值尚未测。

需要补做固定提示词、多步生成的旧/新对照，以及一小批固定数据上的 loss/perplexity 或任务得分。逐字输出不一致时，先分析路由边界和数值误差，不应自动判定质量下降，也不能直接忽略。

## 5. 本机复现

在 `C:/Users/1/Documents/ChatGPT/UCB/FreeToken` 执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'python'
..\validation\.venv-cuda\Scripts\python.exe -m pytest tests/kernels/test_sigmoid_router.py tests/models/test_sigmoid_routing.py -q -ra
..\validation\.venv-cuda\Scripts\python.exe -m pytest tests/kernels/test_dsv4_router_select.py tests/models/test_glm5_next_config.py tests/models/test_minimax_m3.py tests/checkpoint/test_ftw_weights.py -q -ra
..\validation\.venv-cuda\Scripts\python.exe benchmarks/bench_sigmoid_router.py --tokens 1 8 256 1024 4096 --output ../validation/extended-benchmark.json
```

在 `C:/Users/1/Documents/ChatGPT/UCB` 执行真实前缀测试（固定路径脚本，缓存已有，首次需联网）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\validation\.venv-cuda\Scripts\python.exe validation/real_prompt_prefix.py
```

基线 FTW 对照使用 `git archive HEAD` 导出的独立副本 `validation/baseline`，没有重置或改动当前工作树。

