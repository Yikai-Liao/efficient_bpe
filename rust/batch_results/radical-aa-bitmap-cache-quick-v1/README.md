# AA bitmap：word-cache scatter 小测

独立 crate `owned_aa_bitmap_cache` 在相同二进制内比较逐位置 atomic OR 与
按 word 暂存后 OR。保持 `bitmap-adaptive`、aHash、lazy heap、chunk 4096；
只计 unary 与 AB 两个 64 KiB 输入，W1 固定 CPU5，W4 固定
CPU0/1/2/5，每格一次。它是 **n=1 机制筛选**，未跑 4 MiB 性能或完整矩阵。

debug 库测试 18/18、strict Clippy 和 release 构建通过。
[完整轨迹 oracle](../radical-aa-bitmap-cache-gate-v1/differential.json) 为
标准 20 case × atomic/word-cache × std/aHash × W1/W4 的 160 次，另有
加权 4095/4096/4097 个 A、W4 两模式的 2 次。全部规则及最终 token 与
Python 独立逐轮重计数相同。本次 8 个训练调用的完整轨迹及 fingerprint
也逐字一致。

| 输入 | W | atomic OR 次数 | word-cache OR 次数 | atomic scatter | cached scatter | 完整调用 atomic→cached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unary 64 KiB | 1 | 122876 | 4122 | .001217 s | .000906 s | .01049→.00937 s |
| unary 64 KiB | 4 | 122876 | 4121 | .003102 s | .000424 s | .01586→.00823 s |
| AB 64 KiB | 1 | 57341 | 3083 | .000552 s | .000400 s | .00884→.00888 s |
| AB 64 KiB | 4 | 57341 | 3081 | .000467 s | .000303 s | .00780→.01112 s |

OR 次数和 scatter 子阶段在四格均下降；完整调用的方向不一致，尤其 AB
W4 的 cached 更慢，因此不能把 unary W4 一格的降幅当作稳定速度收益。
两种输入在两模式中均实际触发 bitmap。训练后、trace 序列化前采样的
`train_vm_hwm_mib` 为 3.59–3.94 MiB；这是含输入解析的进程高水位，
不能直接折算为 bitmap 内存。bitmap 的分配容量在 `try_reserve_exact`
后才验证；分配器超配后回退可短暂越过 posting 字节守卫，它不是严格
进程内存上限。

[原始数据](quick.jsonl)、[执行环境](quick.jsonl.environment.json)及
[检查记录](checks.json)保留每格完整 call/CPU/HWM 与 CPU 亲和。
[逐文件哈希](new-source-hashes.json)、[源码快照](new-source-snapshot.tar.gz)
和[共享依赖证明](shared-source-provenance.json)支持重建；共享 Rust 源与
commit `5536655` 逐字相同。大二进制只留在 ignored `rust/target/reruns/`。
