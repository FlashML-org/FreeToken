# 完整模型可行性检查（2026-10-22）

这是远端 GGUF 文件头检查，不是完整权重加载或生成实验。没有下载完整 checkpoint，也没有运行小型随机模型的完整引擎测试。

| 候选 | 文件大小（GiB） | 文件中专家张量合计（GiB） | 专家层格式组合数 |
|---|---:|---:|---:|
| unsloth/GLM-4.5-Air-GGUF / UD-IQ1_S | 35.86 | 31.67 | 4 |
| bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF / IQ1_M | 27.38 | 23.57 | 3 |

数值来自 [candidates.json](candidates.json)，包含固定 revision、文件 SHA256 和张量类型统计。专家统计包括文件中附带的 MTP 张量，不能当成精确运行时内存预算；还未计入加载临时内存、系统与 GPU 缓存。原文件列表见 [Unsloth](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/tree/506d64aa8c5cfe9dbbf00bc7a15739438f83204d) 和 [Bartowski](https://huggingface.co/bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF/tree/d4ba84e54f918101e1b024a9632f319a2e907445)。

本机约 31.3 GiB 系统内存、8 GiB 显存；WSL 当前约 15.3 GiB 内存。现有 `engine.py::_pin_budget_bytes` 对 WSL 默认按内存的 40% 预算锁页，`_init_offload_moe_cache` 明确拒绝混合专家格式搭配 CPU strategy 或 CPU 分层。`models/gguf/experts.py` 的默认卸载路径把专家加载为 host banks，并非按 token 从磁盘流式取专家。因此当前已核对的配置不能靠增大 swap 直接跑通。

结论仅限这些候选与当前基线实现，不声称穷尽所有模型、量化和配置。继续可能需要更大内存、其他兼容 checkpoint 或独立的后端工作；不将这种扩展混入路由优化 PR。GLM-4.7-Flash 是 Glm4MoeLiteForCausalLM，当前改动不覆盖该架构，不能替代本 PR 的端到端验证。
