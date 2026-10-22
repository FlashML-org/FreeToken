# CUDA 实测结果

日期：2026-10-20（America/New_York）；日志目录使用 UTC 时间 `cuda-20261021T011532Z`。

## 环境

- NVIDIA GeForce RTX 4060 Laptop GPU，8 GB，计算能力 sm_89，驱动 616.92，Windows 11 / WDDM。
- 独立环境 `validation/.venv-cuda`，Python 3.12.15，torch 2.14.1+cu130，CUDA runtime 13.0，triton-windows 3.8.0.post29（模块版本 3.8.0）。
- Windows Triton 的版本配套依据：[项目安装说明](https://github.com/triton-lang/triton-windows)。
- 基线及补丁文件 SHA256 记录于本轮日志目录 `source-manifest.json`。这次没有修改内核或模型代码，没有提交 PR。

## 结果

| 项目 | 最终结果 |
|---|---|
| CUDA 可用性 | True，成功识别 RTX 4060 |
| 实际 Triton 内核测试 | **105 passed，21.99 秒** |
| FP32 加载及五个模型的合成路由接入测试 | **6 passed，12.27 秒** |
| CUDA Graph 路由微基准 | 12 组完成，全部通过测速前的结果比较 |

内核测试直接加载未改动的实际 kernel 文件，避开完整引擎的包初始化；模型接入测试使用真实模块导入。两者使用 `--confcutdir` 绕过仓库全局 fixture，不能当作完整仓库测试套件通过。

模型测试首次因缺少 huggingface_hub、第二次因缺少 flashlib 停在导入阶段。补齐依赖后 6 项全部通过，初次失败及重试日志均保留。模型测试使用合成数据，没有加载完整模型 checkpoint。

## 性能

单位：微秒。同一组输入，CUDA Graph 模式，新旧交替测试 5 次取中位数。仅测 logits 之后的路由计算，不含前置矩阵运算、权重加载或整个模型生成。

| tokens | 专家数 / top-k | 分组数 / 保留组数 | 原实现 | 融合实现 | 加速 |
|---:|---|---|---:|---:|---:|
| 1 | 128 / 4（eps=0, scale=1） | 1 / 1 | 13.820 | 1.913 | 7.225x |
| 1 | 128 / 4（eps=1e-20, scale=2） | 1 / 1 | 13.947 | 1.863 | 7.487x |
| 1 | 192 / 8 | 8 / 4 | 34.013 | 3.864 | 8.802x |
| 1 | 288 / 8 | 1 / 1 | 15.467 | 2.864 | 5.400x |
| 8 | 128 / 4（eps=0, scale=1） | 1 / 1 | 13.702 | 1.878 | 7.298x |
| 8 | 128 / 4（eps=1e-20, scale=2） | 1 / 1 | 16.135 | 1.879 | 8.585x |
| 8 | 192 / 8 | 8 / 4 | 34.124 | 3.854 | 8.854x |
| 8 | 288 / 8 | 1 / 1 | 16.708 | 2.861 | 5.840x |
| 256 | 128 / 4（eps=0, scale=1） | 1 / 1 | 21.938 | 4.563 | 4.808x |
| 256 | 128 / 4（eps=1e-20, scale=2） | 1 / 1 | 24.051 | 4.546 | 5.291x |
| 256 | 192 / 8 | 8 / 4 | 67.084 | 11.716 | 5.726x |
| 256 | 288 / 8 | 1 / 1 | 33.663 | 8.315 | 4.049x |

这些是当前 Windows 笔记本环境下的微基准，未控制 GPU 时钟，不能推广为整机 tokens/s 提升，也不能替代目标 Linux 环境的端到端 A/B。尚未执行完整模型质量验证、真实 checkpoint 加载、Compute Sanitizer 或完整测试套件。

## 重跑与日志

在工作区根目录运行：

```powershell
.\validation\.venv-cuda\Scripts\python.exe .\validation\run_cuda_validation.py
```

每次创建新的带时间戳目录，旧 CPU 和 GPU 记录都保留。逐项退出码、完整命令与输出见 [results.json](cuda-20261021T011532Z/results.json)，测速原始输出见 [benchmark.log](cuda-20261021T011532Z/benchmark.log)，模型最终通过记录见 [model-retry-2.log](cuda-20261021T011532Z/model-retry-2.log)，最终依赖版本见 [packages-final.log](cuda-20261021T011532Z/packages-final.log)。
