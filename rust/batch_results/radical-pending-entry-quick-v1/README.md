# Pending owner Entry：限定 16 次调用小测

同一 `owned_pending_entry` 二进制比较 `staged`、`fused-direct-combined`
与 `fused-direct-pending`。EN/ZH 连续语料各 256 KiB、512 条规则，
输入 piece weight 均为 1，通常 minimum=2；三模式 × W1/W4 共 12 次。
另做 EN minimum=16 下两个 direct 模式的 W4 对照 2 次，及冻结串行
CF32/aHash/checked 的 EN/ZH W1 参考各 1 次，总计 **16 次计时调用、
每格 n=1**。高阈值还另跑 1 次**不纳入计时表**的同 minimum 串行
正确性 oracle；它与两条高阈值结果的完整轨迹和 fingerprint 匹配。
W1 固定 CPU5、W4 固定 CPU0/1/2/5，均为相同进程 CPU 配额。

[正确性门控](../radical-pending-entry-gate-v1/README.md)通过 14/14 库测试、
strict Clippy、release 和 240+24 次完整轨迹。此次 16 行全部与同输入、
同 minimum 串行完整规则及最终 token 轨迹相同。

| 输入，minimum=2 | 模式 | W1 call / CPU | W4 call / CPU | W4 训练 HWM |
| --- | --- | ---: | ---: | ---: |
| EN | staged | .06560 / .06468 s | .04428 / .13274 s | 8.37 MiB |
| EN | combined | .06383 / .06360 s | .03112 / .08822 s | 8.80 MiB |
| EN | pending | .06919 / .06867 s | .03617 / .08880 s | 8.80 MiB |
| ZH | staged | .03173 / .03170 s | .01998 / .06643 s | 7.82 MiB |
| ZH | combined | .03117 / .03078 s | .02235 / .06649 s | 7.99 MiB |
| ZH | pending | .03686 / .03672 s | .01975 / .06374 s | 7.79 MiB |

串行 CF32 W1 的 EN/ZH call 为 .04794/.02403 s。EN/ZH 常规阈值的
direct 两模式累计新键 B/E 分别为 33681/19258、15364/5338，
并非全部新键合格；staged 路径不填这组 direct 专属计数。
EN W4 的 staged 频率归约加出生填充阶段分别 .00823+.00418 s；
combined/pending 的融合提交为 .00759/.00876 s。ZH 相应为
.00298+.00173 s 对 .00466/.00368 s。这些是不同代码路径的阶段，
不能把阶段数字简单相加成完整调用或推断因果收益。

EN `minimum=16` 的 direct 两模式均真实产生 B=33681、E=3306：
combined/pending W4 call 分别 .02425/.04036 s，CPU .08062/.08291 s，
训练 HWM 7.070/7.145 MiB。`owner_capacity_after_retire_peak` 的公开
可容纳元素总数（公开 capacity）分别 3261/7162，`fresh_scratch_payload_bytes_peak` 分别
43008/32768 B。这是完整训练中高阈值低合格率和容量差异的一个实例；
**HashMap 的公开 capacity 不等于底层 bucket 数或常驻物理字节**，
且两模式归约前公开容量峰值已不同；插入也可能只复用初始 retain 留下的
tombstone。这里没有证明新分配了 bucket 或产生长期物理内存污染。
HWM 包含输入解析，n=1 的调用
差异不证明稳定性能排序。此输入与既有 4 MiB weight=2 语料不同。

[原始 16 行](screen.jsonl)、[环境与哈希](screen.jsonl.environment.json)、
[不计时高阈值串行 oracle](high-min-serial-oracle.json)、[检查清单](checks.json)
保留完整阶段、B/E/D、容量、CPU 与训练 HWM。
[逐文件源码哈希](new-source-hashes.json)、[新源码快照](new-source-snapshot.tar.gz)
和[共享依赖证明](shared-source-provenance.json)可还原，根 Rust 共享文件
与 commit `af7f05d` 逐字一致。大二进制仅保留在 ignored `rust/target/reruns/`。
