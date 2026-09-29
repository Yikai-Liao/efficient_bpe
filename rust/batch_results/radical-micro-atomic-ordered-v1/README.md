# 微区、原子旧键、有序 posting：限定复核

这轮只检验三个新机制，没有扩大到 4 MiB 或重跑旧矩阵。三个新 crate
分别通过其[微区门控](../radical-region-tasks-gate-v1/README.md)、
[原子旧键门控](../radical-atomic-old-gate-v1/README.md)和
[有序 posting 门控](../radical-ordered-posting-gate-v1/README.md)。
[56 次完整训练的原始记录](screen.jsonl)全部匹配同输入的完整 merge 规则、
最终 token 轨迹与 fingerprint；[逐组结果](summary.json)保留每次原值。

在英中 256 KiB、512 规则、权重 1、最低频 2 的屏测中，微区数从
`T=W` 增至 `T=4W` 虽提高按访问次数计算的静态负载上界，但 W4
完整调用在四个同模式对照中均变慢。原子旧键归约在 EN W4 变慢，
在 ZH W4 小幅变快。有序 posting 的自然语料 W4 单次调用较快，
但原模式 AA 排序只占 7–95 µs，不能将多毫秒的完整调用差异归因于
省略排序。以下都是 n=1/2 的机制诊断，不构成稳定排序。

## 计时与同源检查

- 输入：`quick-en-continuous-262144`、`quick-zh-continuous-262144`，
  以及有序 posting 的单篇 `AB` 65536 输入；`AB` 只测 W4、n=2。
  aHash、lazy heap、chunk 4096；有序 posting 使用 atomic region k1。
- W1 进程固定 CPU 5；W4 固定 CPU 0、1、2、5。CPU 秒是整个训练调用的
  **进程** CPU 时间，CPU/墙钟只是平均占用核数，不等于有效计算利用率。
  HWM 是训练调用后、构造完整 trace 前采集的进程高水位，含启动和
  输入解析，不是 allocator 的净分配。
- n=2 的第二轮反转第一轮相对次序。表中的时间单位为毫秒，`原值`
  按两轮顺序列出；中位数对 n=2 是两个数的算术中点。CPU、HWM 与
  各阶段指标分别取中位数，故不同列不能逐行相加作为一轮实际时间。
- `serial` 是同窗口 CF32/aHash/checked W1 直接串行参考。所有自然
  语料 fulltrace 与该参考一致；`AB` 与同窗口 `ordered_region` 和
  之前冻结的 combo fingerprint 一致。[runner](run_screen.py)检查了
  fixture/source/binary SHA、有效模式与全部完整轨迹。

| 语料 | 模式 W4 | 完整调用原值 ms | 中位 ms | 进程 CPU ms | 训练 HWM MiB | 自身 W1/W4 | 串行/W4 |
|---|---|---:|---:|---:|---:|---:|---:|
| EN | micro region k1 | 48.172 / 38.130 | 43.151 | 115.407 | 8.96 | 1.47 | 0.87 |
| EN | micro region k4 | 49.936 / 58.912 | 54.424 | 144.094 | 9.59 | 1.18 | 0.69 |
| EN | micro snapshot k1 | 44.043 / 33.085 | 38.564 | 100.940 | 9.04 | — | 0.97 |
| EN | micro snapshot k4 | 55.773 / 47.641 | 51.707 | 133.213 | 9.68 | — | 0.73 |
| EN | atomic owner | 35.585 / 30.862 | 33.223 | 92.867 | 8.67 | 1.75 | 1.13 |
| EN | atomic producer | 49.461 / 33.138 | 41.299 | 102.385 | 8.51 | 1.42 | 0.91 |
| EN | ordered region | 39.242 | 39.242 | 111.648 | 9.07 | 1.63 | 0.96 |
| EN | ordered global | 38.337 | 38.337 | 112.314 | 9.02 | 1.44 | 0.98 |
| ZH | micro region k1 | 20.930 / 22.181 | 21.556 | 63.940 | 8.25 | 1.60 | 0.92 |
| ZH | micro region k4 | 23.322 / 27.413 | 25.367 | 73.434 | 8.26 | 1.36 | 0.78 |
| ZH | micro snapshot k1 | 17.729 / 17.411 | 17.570 | 58.960 | 8.12 | — | 1.13 |
| ZH | micro snapshot k4 | 42.275 / 21.816 | 32.046 | 80.121 | 7.97 | — | 0.62 |
| ZH | atomic owner | 25.545 / 22.342 | 23.944 | 70.704 | 8.03 | 1.60 | 0.83 |
| ZH | atomic producer | 22.902 / 22.053 | 22.478 | 58.288 | 7.99 | 1.36 | 0.88 |
| ZH | ordered region | 22.400 | 22.400 | 55.421 | 8.31 | 1.60 | 0.89 |
| ZH | ordered global | 20.796 | 20.796 | 59.881 | 8.05 | 1.44 | 0.95 |

EN 串行 W1 两次为 39.024/36.077 ms，中位 37.550 ms、HWM 7.92 MiB；
ZH 为 20.951/18.741 ms，中位 19.846 ms、HWM 5.79 MiB。各并行模式的
W1 原值、CPU/HWM 和自身比值的精确值均在 `summary.json`。
AB W4 的 region 为 8.722/14.085 ms，global 为 11.631/10.862 ms；
两范围交叠，不足以判定全局有序版本的完整调用优势。

## 工作量与容量

微区 EN W4 非 AA posting 访问共 202627 次，k1/k4 的按批最大区访问
下界分别为 54935/50698 次；ZH 共 28281 次，下界 10708/7943 次。
在**每次访问同成本**的假设下，静态可并行访问比 EN 从 3.69 到 4.00、
ZH 从 2.64 到 3.56。这是计数下界，不是实测 speedup。k1→k4 的
路由头容量峰值由 1408→5632 B，EN region W4 的 partition searches
4096→16384、route delta capacity 7168→14336 项、plan 17.93→21.32 ms；
ZH 的 partition searches 同样四倍，plan 5.98→7.08 ms。snapshot
EN W4 k1/k4 的边界查询为 21/76 次、延迟写 2/12 次；ZH 为 2/41 次、
延迟写 0/8 次。descriptor capacity 上界 600→3000 B。`partition_worker_seconds`
是 worker 时间之和，不能与完整调用的墙钟时间直接相减。

原子旧键 EN W4 producer 模式记录约 32871 次旧 route、29753 次
原子旧键调用、4299 个退休标记；旧键 flush 的 worker 时间和约
3.51 ms，嵌在 plan 的 16.98 ms 中。owner 对照的 plan 为 12.05 ms；
producer/owner 的 fused commit 为 8.85/8.00 ms。ZH producer 有
15258 次旧 route、9577 次原子调用、3748 个退休标记，flush worker
时间和约 1.93 ms；producer/owner plan 为 6.17/5.74 ms，fused
commit 为 3.63/4.99 ms。记录的 `preflush_route_capacity_peak`
EN/ZH 为 4592/2268 项；这些公开容量计数不是物理 bucket 字节数。

有序 posting 在 EN W4 反转 50812 段/269984 个 birth 位置，省略
1 批 AA sort；原 region 排序仅 0.007 ms。ZH 反转 11654 段/40975
位置、省略 11 批，原排序 0.095 ms。global 的 birth group fill 分别
比 region 多约 0.571/0.295 ms，且 hash/调度及 n=1 波动参与完整调用。
在专门 `AB` 输入中，global 省略 15 批、65519 个 AA 位置的 sort；
region 的 sort 累计约 1.093 ms，global 的反转为 56 段/65519 位置。

## 可复现来源

新 crate 的[源码 SHA 列表](new-source-hashes.json)和
[98 KiB 源码快照](new-source-snapshot.tar.gz)固定了这轮二进制来源；
[共享依赖来源](shared-source-provenance.json)记录基座提交
`e2cfd89950f6b80cd4cad229737979417f4ff04a`、`rust/src/**`、root
Cargo 文件与 `aa_parity.rs` 的逐字 SHA、Rust 工具链版本。复原时检出
基座提交，再叠加源码快照。release 二进制只留在 ignored
`rust/target/reruns/`，不提交。`verify_archive.py` 的[核查结果](checks.json)
已对 20 个新文件、26 个共享文件、源码 tar 成员和四个二进制 SHA 逐项
复核；它不重新运行训练。
