# 端点融合 × AA bitmap：组合小测

独立 crate `owned_endpoint_bitmap_combo` 把 tagged endpoint 规划与
word-cache AA bitmap 放在同一个训练器中。此次只比较同二进制的
`two-pass|tagged-fused` × `sort|bitmap-adaptive`，固定 aHash、lazy heap、
chunk 4096；AB 64 KiB 取完整四格，EN/ZH 连续 256 KiB 固定 tagged-fused
比较 AA 开关。每格仅一次，W1 固定 CPU5、W4 固定 CPU0/1/2/5。
这是 **n=1 筛选，不是 4 MiB 或完整矩阵**。

debug 库测试 23/23、strict Clippy 与 release 构建通过。
[Python 独立 oracle](../radical-endpoint-bitmap-combo-gate-v1/differential.json)
完成标准 20 case × 3 endpoint × 2 AA × std/aHash × W1/W4 的
480 次完整轨迹，另 5 次定向轨迹。定向混合加权输入在同一次训练中
确实经历非 AA 融合、密集 bitmap、sort fallback，产生长度 >255 的 token；
`max_merges=2^31` 小语料触发 tagged 域 fallback 后，AA bitmap 仍运行。
本次 16 个训练调用的完整规则轨迹、最终 token 与 fingerprint 全相同。

| AB 64 KiB 模式 | W1 调用 | W4 调用 | W4 训练 HWM |
| --- | ---: | ---: | ---: |
| two-pass + sort | .01040 s | .00862 s | 4.27 MiB |
| two-pass + bitmap | .01006 s | .00824 s | 3.97 MiB |
| tagged-fused + sort | .00983 s | .00787 s | 4.21 MiB |
| tagged-fused + bitmap | .00863 s | .00634 s | 4.07 MiB |

AB 的组合模式实际执行非 AA 融合 1 批、bitmap 3 批和 sort fallback
12 批。两阶段非 AA 起点峰值载荷代理 131072 B，融合为 0；
AA sort plan 容量峰值 524288 B，adaptive 为 65536 B，bitmap 分配峰值
8200 B。W4 bitmap word-cache 完成 3076 次 atomic OR。HWM 是含输入
解析的进程高水位，且这些容量代理不能相加为总内存。单次调用的排序
只说明此输入值得进一步检查，不证明组合在一般语料上净加速。

EN/ZH 自然输入的 adaptive **均未执行 bitmap 轮**：EN 仅 sort
fallback 1 次，ZH fallback 11 次。固定 tagged-fused 的 EN W4
sort/adaptive 分别 .05414/.03870 s，ZH W4 .02122/.02127 s；EN 的
差异是明确的负控制波动，不能归因于 bitmap。两个输入的非 AA 起点
载荷代理均为 0，表明融合路径确实运行。AA bitmap 分配容量是在
`try_reserve_exact` 之后检查，分配器超配后回退可短暂越过 posting
字节守卫，不能称为严格进程内存预算。

[原始 16 行](quick.jsonl)、[执行环境](quick.jsonl.environment.json)、
[检查记录](checks.json)保留完整 call、CPU、训练 HWM、阶段和容量指标。
[逐文件源码哈希](new-source-hashes.json)、[快照](new-source-snapshot.tar.gz)
及[共享依赖证明](shared-source-provenance.json)可重建，根 Rust 共享源码
与 commit `705bea3` 逐字一致。大二进制仅存 ignored `rust/target/reruns/`。
