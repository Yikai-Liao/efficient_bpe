# Pending owner Entry 正确性门控

最终 `owned_pending_entry` 通过 rustfmt、14/14 debug 库测试、strict
Clippy（全部 targets，`-D warnings`）及 release 构建。
[独立 Python naive oracle](differential.json) 完成 20 个标准输入 ×
`staged|fused-direct-combined|fused-direct-pending` × std/aHash × W1/W4，
合计 240 次完整规则与最终 token 轨迹比对；另有 24 次定向比对。
定向输入包括 256 个均不合格的 fresh key（新两模式 B=256、E=0、
D=256）和非均匀权重下的合格与不合格 fresh key。lib 定向单测另覆盖
哈希碰撞、foreign-only birth 和异常链。全部通过。

[首次失败和修正记录](FIRST_FAILURES.md)保留早期容量测试断言
`448→192` 及随后测试代码的 Clippy 失败；两者均未涉及训练实现。
[检查清单](checks.json)对应最终 lib SHA
`2739779436cef26795f7e6ac63bfc1710d1833eeddf33b0cc577307894ebee01`。
release 二进制存 ignored `rust/target/reruns/radical-pending-entry-gate-v1/`，
其 SHA256 见检查清单。当前门控只证明精确性，不给性能结论。
