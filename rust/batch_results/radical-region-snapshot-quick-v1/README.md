# Region snapshot：限定 12 次调用小测

同一 `owned_region_snapshot` 二进制比较 `dynamic|region|snapshot`，
固定 tagged-fused 端点、aHash、lazy heap、chunk 4096。EN/ZH 连续
256 KiB 各 512 条规则、piece weight=1、minimum=2，三模式 × W1/W4
共 **12 次计时调用、每格 n=1**。W1 固定 CPU5，W4 固定 CPU0/1/2/5。

[正确性门控](../radical-region-snapshot-gate-v1/README.md)通过局部模型
5/5、全部库测试 23/23、strict Clippy、release 和 160+10+1 次完整
轨迹。此次 12 行同语料的规则、最终 token 与 fingerprint 全相同。

| 输入 | 模式 | W1 call / CPU | W4 call / CPU | W4 训练 HWM |
| --- | --- | ---: | ---: | ---: |
| EN | dynamic | .05620 / .05508 s | .04587 / .12036 s | 8.93 MiB |
| EN | region | .06320 / .06125 s | .05362 / .10874 s | 8.78 MiB |
| EN | snapshot | .06066 / .06034 s | .06198 / .15701 s | 9.01 MiB |
| ZH | dynamic | .02501 / .02500 s | .03270 / .08916 s | 8.36 MiB |
| ZH | region | .03645 / .03555 s | .04105 / .09161 s | 7.94 MiB |
| ZH | snapshot | .02914 / .02892 s | .02216 / .07006 s | 8.30 MiB |

W4 snapshot 实际分配的边界描述符容量代理在两输入均为 600 B，
有效 region 数 4。EN 有 21 次边界查询、2 次延迟端点写和 4 次
跨区出生边；ZH 有 2 次查询、0 次延迟写、0 次跨区出生边。
EN/ZH 的 snapshot plan 分别 .02475/.00580 s，region 分别
.02206/.01085 s，dynamic 分别 .01781/.00789 s。快照构建与刷新
时间很小，但其它阶段和任务调度也变化，不能把完整调用差异归因于
单一访问机制。局部与 trainer 定向验证还强制远端写路径实际激活。
本次 EN W4 snapshot 比两个控制慢，ZH W4 反而比两个控制快；
这种 n=1 反向结果不足以证明净加速或稳定排序。训练 HWM 是包含
启动和输入解析的进程高水位，描述符容量代理不能与它直接相加。

[原始 12 行](screen.jsonl)、[环境与哈希](screen.jsonl.environment.json)、
[检查清单](checks.json)保留全部阶段与容量指标。
[逐文件源码哈希](new-source-hashes.json)、[新源码快照](new-source-snapshot.tar.gz)
和[共享依赖证明](shared-source-provenance.json)可还原，根 Rust 共享文件
与 commit `af7f05d` 逐字一致。大二进制仅在 ignored `rust/target/reruns/`。
