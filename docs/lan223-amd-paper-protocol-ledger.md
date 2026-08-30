# LAN-223 paper protocol ledger

This ledger records which FreeToken paper fields are available before a result
is called a strict replication. The primary paper is `2608.16157v1.pdf` in the
project root. The upstream summary is `docs/upstream-qwen-paper-protocol.md`.

| Field | Paper evidence | State | LAN-223 consequence |
| --- | --- | --- | --- |
| Models | Qwen3.6-35B-A3B, DeepSeek-V4-Flash, GLM-5.2 | Confirmed | Qwen is primary AMD qualification model |
| RTX 4060 row | RTX 4060 Laptop 8 GB, Core i9-13900H, LPDDR5 32 GiB, PCIe 4.0 x8 | Confirmed | Hardware reference only |
| RTX 5090 desktop | RTX 5090 32 GB, Ryzen 9 9950X3D, DDR5 192 GiB, PCIe 5.0 x16 | Confirmed | Capacity and performance reference only |
| Qwen precision | BF16 on most paper systems, NVFP4 on 8 GB laptop | Confirmed | Q4 GGUF cannot claim parity |
| DSV4 precision | Native MXFP4 experts | Confirmed | Requires separate AMD capacity and correctness program |
| Workloads | AIME, OpenCode plus SWE, Claude Code plus SWE, OpenClaw email/calendar | Confirmed | Recreate as paper-inspired until exact fixtures recovered |
| Metric | Per-request mean decode TPS and per-request mean TTFT | Confirmed | Harness records client SSE timings separately |
| Tail claim | Worst FreeToken agent turn below 44 seconds | Confirmed | Requires complete multi-turn matrix |
| Exact prompt corpus | Not published in paper | Missing | Blocks strict replication |
| Exact output caps and stops | Not published in paper | Missing | Blocks strict replication |
| Warmup and scored sequence | Only partially described | Missing | Blocks strict replication |
| Exact cache and KV allocation | Not published in paper | Missing | Blocks strict replication |
| Exact commit, driver, CUDA stack | Not fully published | Missing | Blocks strict replication |

## Decision rule

Until every missing row is resolved from released artifacts or the authors,
call the result `LAN-223 paper-inspired`, never `paper replication`.

