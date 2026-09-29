# Fused endpoint：有限 4 MiB 复核

这是 [256 KiB 小测](../radical-fused-bitmap-quick-v1/README.md) 后唯一获授权的
4 MiB 复核，**不是完整矩阵**。两种 endpoint 模式在同一个新二进制内比较；
另用已冻结的直接串行 CF32 aHash checked 作绝对参照。输入为 EN/ZH 连续文本，
各 3000 条规则；每配置两次，先以 seed 20260930 乱序，再反向执行同一顺序。
W1 固定 CPU5，W4 固定 CPU0/1/2/5。20 次完整 merge 规则及最终 token
JSON 两两一致，且所有 fingerprint 与此前独立归档的同输入值一致。

以下完整调用秒数按每配置两次取中位数，括号内为实际 min–max：

| 输入 | 模式 | W1 | W4 | 自身 W1/W4 |
| --- | --- | ---: | ---: | ---: |
| EN | two-pass | 1.282（1.275–1.289） | .492（.465–.519） | 2.61× |
| EN | tagged-fused | 1.245（1.211–1.279） | .549（.451–.648） | 2.27× |
| EN | 直接串行 CF32 aHash | 1.066（1.060–1.072） | — | — |
| ZH | two-pass | .559（.552–.566） | .328（.301–.355） | 1.70× |
| ZH | tagged-fused | .553（.549–.556） | .311（.281–.342） | 1.78× |
| ZH | 直接串行 CF32 aHash | .648（.632–.663） | — | — |

融合的 W1 在两种输入都略短；W4 的 EN 两次从 .451 到 .648 秒，
与 two-pass 范围交叠，不能据中位数宣称融合稳定更快或更慢。
ZH W4 的中位数差也只有约 5%。EN 直接串行仍短于两种 W1 endpoint。
n=2 只说明短输入结果尚未稳定延伸到较长输入。

阶段计时同样按字段分别取中位数。EN W4 two-pass/fused 的 plan 为
.201/.234 秒，apply 为 .0188/.00016 秒；ZH W4 plan 为 .0799/.0748 秒，
apply 为 .0101/.00042 秒。融合把端点写入规划任务，故 apply 近零是
计时边界移动；EN W4 的 plan 已多于两阶段路径，不能单看 apply 推断收益。
EN 的非 AA 融合覆盖 240 批、2,889,948 个实际 merge，ZH 为 249 批、
638,508 个 merge。有效起点临时载荷峰值代理分别从 1,243,208 B、
113,260 B 降为零；AA 仍保留 plan，所以 `peak_task_starts` 的融合值是
EN 1627、ZH 2402。EN 自然输入实际发生 1 次 `decoder_zero_rereads`，
ZH 为零。

W4 `call_cpu_seconds/call_seconds`（分别取 CPU 与 wall 中位数后相除）
为 EN two-pass 3.04 核、fused 2.64 核；ZH 2.50/2.64 核。
这只是进程 CPU 时间的数学比值，包含自旋和内存等待的影响，不能视为
有效工作并行度。训练后、trace 序列化前的 `train_vm_hwm_mib` 中位数：
EN W1 two-pass/fused 78.92/80.03 MiB，W4 94.75/92.66 MiB；
ZH W1 82.88/82.21 MiB，W4 99.85/97.28 MiB。该进程高水位也包含
输入解析，不等于纯后端载荷。

[原始 20 行](screen.jsonl)、[各字段中位数及范围](summary.json)、
[执行环境与二进制哈希](screen.jsonl.environment.json)、
[完整性检查](checks.json)可复核上述数字。运行脚本在 [run_screen.py](run_screen.py)。
新 crate 的逐文件哈希和源码包复用[小测归档](../radical-fused-bitmap-quick-v1/README.md)
的 `new-source-hashes.json` / `new-source-snapshot.tar.gz`；共享 Rust
依赖与 commit `5536655` 逐字相同。大二进制仅在被忽略的 `rust/target/reruns/`
中，不作为 Git 归档内容。
