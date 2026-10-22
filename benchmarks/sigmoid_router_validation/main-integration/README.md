# 最新 main 整合与复验（2026-10-22）

- 上游 main：`546c5fe2e4468e2f3f8a574c64021a47016c35f7`；合并提交 `62e8a38f9a8a1b3560ba3d56e24281d88be9958a`。
- 复验代码快照：`7fd095282f2e9280546d6f90929df5ea87634a5e`。运行时为合并提交加同内容的未提交测试/审计脚本；随后提交为此 SHA，最终证据提交只追加文档与日志。
- 原分支在 `codex/backup-pr730-before-main` 保留。补全浅克隆历史后正常 merge，无冲突，未 reset/clean。
- WSL 本地工作树 `/home/freetoken/freetoken-work/FreeToken-integrated`；独立 main 比较树 `/home/freetoken/freetoken-work/FreeToken-main-baseline`。复验在 Linux 文件系统运行。

## 实际验证

| 项目 | 修改版 | 最新 main | 日志 |
|---|---|---|---|
| 原生扩展构建 | 成功，FreeToken 0.2.0，4m05s | 复用同源原生构建 | [build.log](build.log) |
| 路由/模型测试 | 121 passed，9.23s | 新测试依赖新内核，不在 main 运行 | [targeted-tests.log](targeted-tests.log) |
| 相关及扩展回归 | 92 passed / 1 skipped，7.71s | 92 passed / 1 skipped，7.83s | [分支](FreeToken-integrated-regression.log)、[main](FreeToken-main-baseline-regression.log) |

main 与分支的原生扩展源码和 setup.py 一致，因此独立 main 树复用本次编译产物；Python 路径指向各自工作树。不是两个不同版本二进制之间的性能对照。

扩展回归包括原 46 项及 engine cache-budget、CPU-layer 配置、scheduler status。两边首次均 90 passed、2 failed、1 skipped，失败源于缺少 FlashInfer；安装项目指定 `flashinfer-python[cu13]==0.6.18.post1` 后通过。保留 [原分支失败日志](regression-before-flashinfer.log)、[原 main 失败日志](baseline-before-flashinfer.log) 和 [安装日志](flashinfer-install.log)。跳过项明确为 WSL pinning 限制；未运行全仓测试及完整 FlashInfer 推理。PyTorch/Triton 版本未改变。

## 正确性与路径审查

- 复用五家族原表达式对照，selection-only bias、group top-2 sum、M2 BF16/epsilon、加载后 FP32 舍入保持既有语义。
- [tie-audit.json](tie-audit.json) 保存 4 行完整 logits、bias、unbiased scores、两侧 IDs/weights 与 selection scores；2,385 行专家集合差异为 0。
- 四行样本固化到 `tests/kernels/data/sigmoid_router_ties.json`，新增回归检查专家集合、权重-ID 配对、合成专家输出归约（rtol=1e-5，atol=1e-6）。这是归约次序检查，不是实际专家完整计算或模型质量评估。
- 精确边界同分仍可能改变专家集合；采用最低索引确定性规则，未声称复现 PyTorch 不保证的 tie 顺序。保留在 PR 中供维护者审阅。
- 五个模型 forward 静态追踪均进入 `_route` 和共享 kernel。`GraphRunner.pad_batch` 补 dummy requests，`replay` 返回 `logits[:batch.size]`；不接入新的有效行计数，不改 scheduler。完整专家/服务联合运行仍未验证。

## A/B/C/D 消融

单位 μs，五次旋转顺序的 CUDA Graph 计时中位数，含 input cast 和 gate GEMM；复用真实前缀的 1/93 行激活。A 的原表达式与本轮 main 的 GLM 路由一致；不是 main/PR 服务进程的端到端对照。

| Rows | A: eager + cast | B: eager + FP32 | C: fused + cast | D: fused + FP32 | A/D |
|---|---:|---:|---:|---:|---:|
| 1 | 28.353 | 23.724 | 14.628 | 8.945 | 3.17x |
| 93 | 57.070 | 51.074 | 37.208 | 30.857 | 1.85x |

单行 eager GPU 事件 A/B/C/D 为 13/12/4/3；93 行为 14/13/5/4。这是独立 eager profiler 的设备事件，不是 graph 的主机 launch 数。融合贡献主要收益，FP32 常驻仍有额外收益；不据此自动拆分 PR。

单层实测新增 FP32 张量分配 2,097,152 字节；替换原 BF16 后净增 1,048,576 字节由张量字节数计算。45 层约 45 MiB 为估算，不是整模型峰值显存实测。所有原始样本、min/max、标准差、事件名在 [ablation.json](ablation.json)。

## 状态与剩余边界

- [x] 保存旧分支、合并最新 main、重建、正确性回归、四组消融、main 同环境相关回归。
- [!] 完整真实模型生成、tokens/s、TTFT、峰值显存和质量仍无结果；没有假造启动失败日志。现有候选只做文件头/源码可行性检查。
- [?] 精确/近边界 tie 的集合差异策略、其他家族 checkpoint 和多 GPU，需维护者进一步 review/验证。
- [ ] 完整模型性能验收仍未满足，PR 必须保持 Draft。

最新 main 新增 locked-layer fetch，但 engine 的混合格式 + CPU strategy/CPU-layer 拒绝仍在；不把新特性误写为已解除当前候选限制。旧候选统计包含 MTP，不是精确运行时预算。

复现：使用 [validate-integrated.sh](validate-integrated.sh) 和 [regression-integrated.sh](regression-integrated.sh)，本机路径需替换；[baseline-integrated.sh](baseline-integrated.sh) 展示独立 main 树和同源扩展复用。环境见 [environment.txt](environment.txt)。最终脚本执行遇到错误时原始日志仍保留；以各测试日志与最终回归退出码为准。
