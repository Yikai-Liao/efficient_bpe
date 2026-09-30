# 冻结后执行的限定验证

共享基座固定为 `c1d8fc3bf4fb31bfa59afd545b37a7697c15ec1f`。
`snapshot_sources.py` 只在两个作者明确 freeze、最终源码及文档稳定后运行，
封存两个新 crate 的 Rust 文件、Cargo 文件和设计说明，并逐字核对共享
`rust/src/**`、Cargo 根文件与 `aa_parity.rs`。`oracle_standard.py`
以最终二进制 SHA 配置运行每个 crate 的 20 标准小输入×控制/实验×W1/W4，
各 80 次独立 naive 全规则及最终 token 轨迹对照。额外定向用例在最终
CLI 和激活计数确认后加入各自门控，不增加旧版笛卡尔积。

全部 Cargo、oracle 和作者只读审查结束后，`run_screen.py` 才可用最终
`frozen-config.json` 运行恰好 40 次：adaptive 与 replay 各
控制/实验×EN/ZH 256 KiB×W1/W4×两轮（各 16 次），同窗口
CF32/aHash/checked 直接串行 EN/ZH×两轮（4 次），以及 replay 在
AB 64 KiB 的控制/实验 W4×两轮（4 次）。两轮按 seed 20260930
随机交错，第二轮反转相对次序。W1 限 CPU 5；W4 限 CPU 0、1、2、5。
输出包含完整训练 call wall、进程 CPU、训练后/trace 前 VmHWM、完整
轨迹和 fingerprint、实际 CLI 与二进制/输入/source SHA。AB 另与
上一轮冻结 fingerprint 核对。失败则停止，不发布性能结论。

本计划已执行：最终 `frozen-config.json`、源码封存、Cargo 门控、
两 crate 的完整 oracle 与全部 40 次计时均完成。结果及原始两轮值
见 README、screen.jsonl 与 summary.json。随后根据重放的元数据
成本追加独立 `owned_replay_counts` 原型和 26 调用筛选，归档在
`../radical-replay-counts-v1`，没有扩大为旧版完整矩阵。
