# 区域分片规划：有限小测

同一 `owned_region_fused` 二进制比较 `dynamic` 与 `region`。固定 tagged-fused
端点、aHash、lazy heap、chunk 4096，EN/ZH 连续语料各 256 KiB、512 条规则，
W1 绑定 CPU5，W4 绑定 CPU0/1/2/5。每格仅一次；这是 **n=1 筛选**，
不构成稳定排序或 4 MiB 结论。

debug 库测试 16/16、strict Clippy 与 release 构建通过。
[独立 Python naive oracle](../radical-region-fused-gate-v1/differential.json) 的
160 个标准和 8 个定向完整轨迹全部匹配。定向用例覆盖跨切点长 token、
加权 AA、空 region、worker 数多于位置及 tagged 域 fallback。本次 8 个调用的
完整规则与最终 token 轨迹、fingerprint 在同一输入内也全部一致。

| 输入 | 模式 | W1 call / CPU | W4 call / CPU | W4 训练 HWM |
| --- | --- | ---: | ---: | ---: |
| EN | dynamic | .06327 / .06191 s | .03824 / .10314 s | 8.89 MiB |
| EN | region | .06336 / .06235 s | .04004 / .12972 s | 8.93 MiB |
| ZH | dynamic | .02910 / .02894 s | .02187 / .07176 s | 8.34 MiB |
| ZH | region | .03024 / .02966 s | .02540 / .07634 s | 8.33 MiB |

区域模式的 W4 规划阶段，EN 为 .01545 s（dynamic .01388），ZH 为
.00735 s（dynamic .00536）；频率归约 EN .00793/.00654、ZH
.00426/.00327 s（region/dynamic）。区域分割 worker 用时总和分别
.00116/.00058 s，**它是各 worker 时间之和，不是额外的墙钟阶段**。
EN/ZH 分别做 4096 次切点搜索，跨 region 出生边为 4/0 条；AA 重组时间
.000011/.000032 s。这些阶段可能嵌套，不能简单相加为完整调用。

区域 W4 将 EN flat tasks 从 514 降至 276，ZH 从 512 降至 348；
总 posting visits 仍分别为 202877、29670，总合法合并为 144309、25947。
逐批最忙 region（逻辑任务）的 visits 求和为 55015、11460；总量与该和之比分别
3.69、2.59。合法合并相同口径的比值分别 3.62、2.60。这只是在
**假设单位工作等成本**下由计数得出的静态负载上界，既不是实际加速比，
也不是调度或内存开销后的性能预测。此次 W4 完整调用没有区域模式净收益。
HWM 是训练返回时读取的**进程高水位**，包含启动与输入解析，不能当作
backend 独占分配量。

[原始 8 行](quick.jsonl)、[执行环境](quick.jsonl.environment.json)、
[检查记录](checks.json)保留全部阶段、容量和 fingerprint。
[逐文件哈希](new-source-hashes.json)、[新源码快照](new-source-snapshot.tar.gz)
及[共享依赖证明](shared-source-provenance.json)可还原此次构建；共享 Rust
文件与 commit `705bea3` 逐字一致。大二进制仅在 ignored
`rust/target/reruns/radical-region-fused-gate-v1/`，SHA256 记录在检查文件中。
