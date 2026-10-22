# 本机验证记录

> 最新 GPU 实测：RTX 4060 上 105 项内核测试 + 6 项模型接入测试通过；12 组路由微基准加速 4.049x–8.854x，不代表整模型加速。详见 [CUDA 实测报告](CUDA-RESULTS.md)。下方早期 CPU 限制与未执行状态保留作历史记录。

验证日期：2026-10-20（America/New_York）。本文件记录本地草稿的检查，不代表 PR 已具备提交条件。

## 已配置

独立环境：`validation/.venv`，Python 3.12.15、PyTorch 2.14.1+cpu、pytest 9.1.1、ruff 0.17.1、NumPy 2.5.4。uv 0.13.4 保存在此目录。未修改系统 Python，未安装 CUDA/显卡驱动，未启用 WSL，未启动云资源。

Windows 查询显示 AMD Radeon 780M Graphics，驱动 31.0.22032.1。硬件记录见 `hardware.json`。本环境为 CPU PyTorch，`torch.cuda.is_available()` 为 False；这项软件检查本身不用于证明机器绝对没有其他 GPU。

## 实际结果

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 10 个修改/新增 Python 文件语法编译 | 通过 | Python 语法有效；不等于 Triton 编译成功 |
| Ruff E9/F 检查 | 通过 | 所选静态错误规则未发现问题 |
| git diff --check | 通过 | 已跟踪补丁无空白格式错误；Windows 换行提示非失败 |
| CPU 参考计算 | 8 passed | 原 PyTorch 对照公式满足手算案例和边界检查 |
| GPU 内核测试 | 105 skipped | 没有执行，不能算通过 |
| GPU 模型接入测试 | 6 skipped | 没有执行，不能算通过 |
| GPU 微基准 | 退出码 1，要求 CUDA 环境 | 没有测速结果，不能报告加速比 |

CPU 检查只执行 `tests/kernels/test_sigmoid_router.py` 中的 reference 函数，未执行新的 Triton 内核，也未验证 Fp32RouterLinear 的加载逻辑。它不能证明新补丁正确或更快。GPU 测试使用 `--confcutdir` 隔离仓库的全局 fixture，因此在仅有 torch/pytest 的环境中能够收集并报告跳过；并未安装完整 FreeToken 依赖。

## 重跑

在工作区根目录用 PowerShell：

```powershell
.\validation\.venv\Scripts\python.exe .\validation\run_validation.py
```

脚本会覆盖本目录的各项 `.log` 和 `results.json`，记录命令、退出码和完整输出。该脚本完成日志收集后的退出码不代表全部验证通过，请看逐项结果。

`results.json` 为机器可读记录，`packages.log` 为实际安装版本。后续 GPU 环境和完整模型 A/B 步骤见工作区的 `FreeToken-VALIDATION.md`。当前 CPU 环境不能直接升级成 NVIDIA 硬件；在没有可用 GPU 的情况下，真正的内核验证仍未完成。
