# 重放区域计数：26 调用定向筛选

状态：30/30 Rust lib 测试，fmt、最终全部 targets strict Clippy、release 均通过；80 次标准加 16 次定向独立 naive 完整轨迹一致；26/26 计时轨迹一致。二进制 SHA：`a4c391afb260d1163f0f9535d1631b2848bc88d173c1535aa58b9cabb20e9129`。完整结论见[研究报告](../../ADAPTIVE_REPLAY_REPORT.md)。

本窗口同一 binary 比较 chain、原 Vec replay、replay-inline。自然 EN/ZH 为 256 KiB/512 规则，各模式 W4×n2；另外 inline W1×n2、直接串行 W1×n2，AB 64 KiB 各模式 W4×n2，总计 26 调用。CPU 预算、固定参数、完整计时口径与上一窗口相同。不能用上一窗口的 chain W1 拼本窗口的扩展性。

| 模式 | EN W4 ms | ZH W4 ms | AB W4 ms |
|---|---:|---:|---:|
| chain | 51.56 | 26.32 | 11.09 |
| Vec replay | 66.58 | 30.58 | 7.83 |
| replay-inline | 50.78 | 21.28 | 7.52 |

inline 自身 W1→W4 为 EN/ZH 1.20×/1.51×；同窗直接串行/W4 为 0.74×/1.12×。计数 heap 容量峰值降约 65%/82%，整进程 HWM 未一致下降。n=2、EN 原 replay 波动很大，幅度不能作稳定承诺。ZH 和 AB 支持继续保留候选，EN 没有确认净胜出生链。

`screen.jsonl` 与 `summary.json` 保留原始两轮值、CPU、HWM、源码/输入/binary SHA、CLI、全部计数和相同比值。`frozen-config.json` 对门控、模式及 binary 校验。标准 oracle 使用上一归档的通用脚本，本目录 `run_directed.py` 覆盖两种计数布局、跨区、过滤、u64、长 token、W>N、fallback 并检查语义及物理计数。

`measured-source-snapshot.tar.gz`、`measured-source-hashes.json` 原样保留计时使用的源码与设计文档；environment 的 `source_snapshot_sha256` 与 `source_sha256` 指向它们。测后仅修正 DESIGN 中一句累计 fresh pair 统计说明：每个 pair 只在创建批次出生。修正后的最终源码另存 `new-source-*`，Rust/Cargo 文件逐字未变；environment 明确记录旧/新文档 SHA 与原因。不要把文档修正理解为算法重测或隐含代码改动。

共享基座为 c1d8fc3，26 个共享文件逐字未变。按 manifest release 构建，再使用新输出目录执行 oracle、源码封存、`run_screen.py`、`summarize.py`；重建环境变化须显式记录新 binary SHA，不覆盖本归档。没有新增 unsafe，没有外存或几十核实验。
