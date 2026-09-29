# Fused endpoint 与 AA bitmap：有限小输入门控

这是两条独立实验路线的 **n=1 筛选**，并非完整性能矩阵。新二进制均以
`cargo build --offline --release` 构建，release profile 为 thin LTO、1 个
codegen unit、debug=1。每次训练进程固定 W1=`CPU5` 或
W4=`CPU0,1,2,5`；调度顺序由 seed 20260930 打乱。`call_seconds` 是完整
训练调用，`call_cpu_seconds/call_seconds` 是平均进程占用核数，不能解释为
有效计算利用率。`train_vm_hwm_mib` 在训练调用后、生成完整 trace 前采样，
仍含进程启动及 JSON 输入解析造成的高水位。

两个新 crate 的 debug 库测试分别 14/14 与 13/13 通过，
`clippy --all-targets -- -D warnings` 与 release 构建通过。
[fused oracle](../radical-fused-endpoint-gate-v1/differential.json) 检查了标准
20 小输入 × 3 模式 × 2 hash × W1/W4 的 240 次完整轨迹，另对 ABAB16384、
混合相邻规则、加权 piece 边界，以 chunk=3/7、W4、各重复 3 次检查了
18 次交错敏感轨迹。[bitmap oracle](../radical-aa-bitmap-gate-v1/differential.json)
检查标准 160 次；另以加权 4095/4096/4097 个 A 的三 piece、密集 4 MiB
单 run 进行了 6 次定向完整轨迹检查。三 piece 的 W1/W4 均实际经历 bitmap
与 fallback，合并 token 长度超过 255。**4 MiB 只用于正确性 oracle，未计时。**
本次 30 个训练调用的完整规则轨迹、最终 token 和 fingerprint 均一致。

| 输入 | W | fused 两阶段 | tagged 两阶段 | tagged 融合 | 原 owner aHash | 直接串行 CF32 aHash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EN 256 KiB | 1 | .05926 s | .05528 s | .05285 s | .06079 s | .03778 s |
| EN 256 KiB | 4 | .04589 s | .05378 s | .04332 s | .03972 s | — |
| ZH 256 KiB | 1 | .03272 s | .02974 s | .02439 s | .02910 s | .01911 s |
| ZH 256 KiB | 4 | .02124 s | .02757 s | .01704 s | .03061 s | — |

同一个 fused 二进制内，融合在这四格均短于原两阶段控制；但 EN W4 的
原 owner aHash .03972 s 仍短于 fused .04332 s。不同二进制只是绝对参照，
不能用于单项机制归因。融合实际覆盖 EN 68/69、ZH 76/87 个批次；两阶段
非 AA 有效起点临时载荷峰值代理为 EN 59796 B、ZH 8932 B，融合为 0。
`peak_task_starts` 融合后仍有 EN 99、ZH 131，来自保留的 AA 路径。
本次 `decoder_zero_rereads` 均为 0，复杂发布交错仅由定向测试覆盖。
融合模式把端点写放入规划任务，故其 `apply_seconds` 近零是计时边界移动，
不能单独称为省下原 apply 时间。

| AA 输入 | W | sort | adaptive bitmap | bitmap 轮 / fallback 轮 | bitmap 峰值 | sort AA plan 容量峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unary 64 KiB | 1 | .01033 s | .01003 s | 4 / 12 | 8200 B | 1048576 B |
| unary 64 KiB | 4 | .01086 s | .00814 s | 4 / 12 | 8200 B | 1048576 B |
| AB 64 KiB | 1 | .01009 s | .00916 s | 3 / 12 | 8200 B | 524288 B |
| AB 64 KiB | 4 | .00810 s | .01085 s | 3 / 12 | 8200 B | 524288 B |
| EN 自然 256 KiB | 1 | .05458 s | .05362 s | 0 / 1 | 0 B | 4096 B |
| EN 自然 256 KiB | 4 | .05524 s | .03725 s | 0 / 1 | 0 B | 4096 B |

EN 自然输入是明确的**负控制**：adaptive 没有执行任何 bitmap 轮，只走原
sort fallback。因此 W4 的 .05524→.03725 s 差异不能归因于 bitmap；
是 n=1 短调用的波动警报。AA-heavy 的 W4 两种输入方向相反，也不能宣称
bitmap 普遍加速。容量列是特定临时结构的 capacity 代理，不是进程总内存。
bitmap 的实际分配容量在 `try_reserve_exact` 后才检查；如果分配器超配，
回退前可短暂超过 posting 字节守卫，该守卫不是严格的进程内存预算。

完整数据在 [quick.jsonl](quick.jsonl)、[summary.json](summary.json)；
[sidecar](quick.jsonl.environment.json) 记录 CPU 亲和、二进制与输入哈希。
[checks.json](checks.json) 复核矩阵、轨迹标记及来源。两个新 crate 的源码
位于 [snapshot](new-source-snapshot.tar.gz) 并有[逐文件哈希](new-source-hashes.json)；
[共享依赖证明](shared-source-provenance.json) 确认 `rust/src/**`、根 Cargo
文件和 `aa_parity.rs` 与 commit `5536655` 逐字一致。可执行文件留在被忽略
的 `rust/target/reruns/radical-{fused-endpoint,aa-bitmap}-gate-v1/`，未提交。
