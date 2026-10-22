"""Collect completed WSL validation without archiving or publishing anything."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
evidence = root / 'FreeToken/benchmarks/sigmoid_router_validation'
source = Path('D:/AI/FreeToken-validation/wsl-setup')
target = evidence / 'linux'
target.mkdir(exist_ok=True)
steps = ['cuda-resume', 'torch-resume', 'dependencies', 'cuda-smoke', 'nvcc',
         'build', 'targeted-tests', 'regression-tests', 'deepening-verified']
for step in steps:
    assert (source / f'{step}.exit').read_text().strip() == '0', step
    for suffix in ['log', 'exit']:
        shutil.copy2(source / f'{step}.{suffix}', target)
for name in ['validate-linux.sh', 'deepen-linux.sh', 'smoke.py', 'env.sh',
             'environment-linux.txt', 'requirements-linux-resolved.txt']:
    shutil.copy2(source / name, target)
shutil.copytree(source / 'deepening-verified', target / 'deepening', dirs_exist_ok=True)
a = json.loads((target / 'deepening/ablation.json').read_text())
t = json.loads((target / 'deepening/tie-audit.json').read_text())
assert sum(c['set_mismatch'] for c in t['cases']) == 0
rows = {(r['tokens'], r['variant']): r for r in a['results']}
table = '| 激活行数 | 原实现 | 仅融合 | 仅 FP32 常驻 | 两项同时 | 局部加速 |\n|---|---:|---:|---:|---:|---:|\n'
for n in [1, 93]:
    values = [rows[n, name]['median_us'] for name in ['baseline','fusion_only','fp32_only','both']]
    table += f'| {n} | ' + ' | '.join(f'{v:.3f}' for v in values) + f' | {values[0]/values[3]:.2f}× |\n'
report = '''# WSL Linux 验证记录

日期：2026-10-22。环境建立、FreeToken 原生扩展编译及相关局部测试已成功；完整模型加载、多步生成和端到端 A/B 仍未完成。

## 环境与范围

- Ubuntu 24.04 / WSL 2，内核 6.18.54.1；RTX 4060 Laptop 8188 MiB，Windows NVIDIA 驱动 616.92。
- Python 3.12.3、PyTorch 2.14.1+cu130、Triton 3.8.0、CUDA Toolkit / nvcc 13.0.88。依赖见 [环境清单](environment-linux.txt) 和 [解析后的依赖](requirements-linux-resolved.txt)。
- WSL 当前约 15 GiB RAM、4 GiB swap；没有扩大内存，没有安装 Linux NVIDIA 显示驱动。WSL 虚拟磁盘位于 D 盘。
- 基线 `cc56b6bdcf9188eb632197d6ec719c61bff11798` 加本地补丁；不是最新 main 的完整模型 A/B。Linux 工作树是 Windows 工作树副本。
- 使用核心依赖及 Triton 路径；没有安装可选 `[accel]` 全套组件，不能由此声称所有原生加速后端已验证。

## 已完成结果

| 检查 | 结果 | 原始记录 |
|---|---|---|
| CUDA 算术与 pinned memory 拷贝 | 通过 | [日志](cuda-smoke.log) |
| FreeToken 原生扩展编译、editable 安装 | 通过，4 分 28 秒；CLI 返回 0.1.3 | [日志](build.log) |
| 融合路由与五模型接入 | 117 passed，28.05 秒 | [日志](targeted-tests.log) |
| 相关回归 | 46 passed，10.47 秒 | [日志](regression-tests.log) |

回归覆盖 DSV4 router、GLM5 Next 配置、MiniMax M3、FTW 权重加载。此前 Windows 的 3 项 FTW 失败在 Linux 中通过。此处是相关测试子集，并非全仓库测试通过。

## 真实权重路由复验

使用此前固定版本 GLM-4.5-Air 的三层 gate 权重，以及真实数学提示词前缀的已保存激活；没有重新执行整个模型。

27 组真实 gate＋合成激活用例，共 2,385 行：专家集合差异为 0，专家顺序差异为 4，与此前观察一致。详情见 [同分调查](deepening/tie-audit.json)。单行和 93 行真实提示词激活的四个变体均通过专家 ID 和权重容差断言。

以下包含输入转换、gate 矩阵乘与路由，不含专家计算或生成；单位 μs，五次 CUDA Graph 测量的中位数，TF32 关闭。

''' + table + '''
原始五次样本、标准差和 eager GPU 事件见 [ablation.json](deepening/ablation.json)。这些数据仅比较本次 WSL 内的新旧路由，不用于推断 Windows 与 WSL 谁更快，更不能外推完整模型 tokens/s。

## 复现与记录

[validate-linux.sh](validate-linux.sh) 包含依赖安装、CUDA 检查、编译及两个 pytest 命令；[deepen-linux.sh](deepen-linux.sh) 包含真实权重路由复验命令。脚本中的本机工作目录和 D 盘缓存路径需按复现机器调整。权重获取与版本见 [主说明](../README.md)。

安装日志与每步 `.exit` 均已保留，归档的九个步骤退出码均为 0。首次路由复验完成计算后，外层命令因变量传递错误未保存正确退出码；该次记录留在本机 `D:/AI/FreeToken-validation/wsl-setup/deepening.log`，本文只引用改成脚本后成功重跑的 `deepening-verified` 结果。

## 仍未解决

Linux 环境障碍已解除，但原 GLM-4.5-Air BF16 完整权重仍超出本机容量。受影响量化 checkpoint 的格式支持与内存需求仍需逐一确认。当前没有完整模型输出、tokens/s、TTFT、整模型峰值显存或质量结论；没有提交、推送或创建 PR，也没有生成压缩包。
'''
(target / 'README.md').write_text(report, encoding='utf-8')
print(table)
print('Collected Linux evidence:', target)
