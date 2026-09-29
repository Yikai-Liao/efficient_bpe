# Owner accumulator：复用最大 producer map 的小测

独立 crate `owned_reuse_accumulator` 以同一二进制比较 `staged`、
`fused-fresh` 与 `fused-reuse`。固定 aHash、lazy heap、chunk 4096；
EN/ZH 连续 256 KiB、512 规则，W1 固定 CPU5、W4 固定 CPU0/1/2/5。
每格仅一次，属于 **n=1 机制筛选**，未做 4 MiB 计时。

debug 库测试 12/12、strict Clippy、release 构建通过。
[Python 独立 oracle](../radical-reuse-accumulator-gate-v1/differential.json) 的
20 case × 3 commit 模式 × std/aHash × W1/W4 共 240 次完整规则轨迹及
最终 token 匹配。本次 12 个训练调用的完整轨迹和 fingerprint 也一致。

| 输入 | W | staged | fused-fresh | fused-reuse |
| --- | ---: | ---: | ---: | ---: |
| EN 256 KiB | 1 | .05658 s | .08140 s | .05281 s |
| EN 256 KiB | 4 | .05746 s | .04542 s | .04555 s |
| ZH 256 KiB | 1 | .02727 s | .04649 s | .02830 s |
| ZH 256 KiB | 4 | .02330 s | .01756 s | .03155 s |

完整调用方向不一致；尤其 ZH W4 中复用模式慢于 staged，不能把下面的
减少哈希入口次数直接当作端到端收益。W4 确定工作量如下：

| 输入 | fresh 外来入口访问 | reuse 跳过最大表入口 | reuse 外来入口访问 | accumulator 容量峰值代理 | 转置 route 头容量代理 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN | 66782 | 23091 | 43591 | 3584 entries | 2816 B |
| ZH | 30614 | 16092 | 14471 | 3136 entries | 2816 B |

任务动态分配使模式间的 producer 局部 map 分布可变化，故 ZH 两模式的
入口总数不必严格相同。`accumulator_capacity_sum_peak` 是每批各 owner
map 的 `HashMap::capacity()` **entry 槽数之和**，再取批次峰值；
它不是字节数或实测同时活跃堆峰值。`transposed_route_header_bytes_peak`
是原/转置路由头容量的保守重叠代理。真正训练进程 HWM 的 W4 值为
EN staged/fresh/reuse 8.60/8.96/9.00 MiB，ZH 8.02/8.09/7.60 MiB，
含启动与 JSON 解析，且 n=1 不支持稳定内存排序。

staged 的频率归约与 birth 填充是两个顺序、互不嵌套的计时区间：
EN W4 .00998+.00569=.01567 s，ZH W4 .00357+.00219=.00576 s。
fused-fresh/fused-reuse 的完整 owner commit 单区间分别为 EN
.01062/.01294 s、ZH .00304/.00600 s；其中还含 route 头转置和
局部链校验。计时范围不完全同构，且阶段变化不能替代完整调用比较。
复用版保留选中 posting 生命周期、expected 向量、局部链与总 count
校验；本小测不能证明这些成本已被消除。

[原始数据](quick.jsonl)、[执行环境](quick.jsonl.environment.json)与
[检查记录](checks.json)保留 call、CPU、训练 HWM 和所有阶段/容量字段。
[逐文件哈希](new-source-hashes.json)、[源码快照](new-source-snapshot.tar.gz)
及[共享依赖证明](shared-source-provenance.json)支持重建；共享 Rust 源与
commit `5536655` 逐字相同。大二进制只留在 ignored `rust/target/reruns/`。
