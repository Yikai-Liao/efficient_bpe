# 有序 posting 正确性门控

`owned_ordered_posting` 在同一二进制内比较 `region` 与 `global` posting
顺序。最终源码通过 rustfmt、debug 库测试 25/25、全部 targets 的 strict
Clippy（`-D warnings`）和 release 构建。release 二进制留在 ignored
`rust/target/reruns/radical-ordered-posting-gate-v1/`，SHA256 为
`dcd230559be789c94183bd1e954cd753ad3f682f15100749f6a6a68f4c0a5662`。

[独立 Python naive oracle](differential.json) 对 20 个标准输入执行
`region|global` × W1/W4、固定 aHash 与 k1 的 80 次完整规则及最终 token
轨迹比对。另在 snapshot k4/std 下比对相邻新边、AA→非 AA、跨多 cut 的长
token、远端端点与域回退五类输入，两种顺序共 10 次，全部一致。请求
`global+dynamic` 按契约拒绝；超出 tagged 域时有效模式回退 `region`。
debug 断言检查新生 posting 全局严格递增及跨区归属。

定向输入实际触发了有序 birth 链反转与 AA 排序省略；snapshot 延迟写也在
远端端点例中触发。门控证明这些路径的轨迹正确性，不提供速度结论。
