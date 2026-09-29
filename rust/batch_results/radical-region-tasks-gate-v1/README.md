# 微区任务正确性门控

`owned_region_tasks` 将物理区数设为 `T=min(kW,N)`，其中 W 为 worker 数、
N 为语料物理位置数，pair owner 仍为 W。最终源码通过 rustfmt、
debug 库测试 24/24、strict Clippy（全部 targets，`-D warnings`）与
release 构建。

[独立 Python naive oracle](differential.json) 的 20 个标准输入在
`region|snapshot` × W1/W4 × k4 × aHash 下完成 80 次完整规则与最终
token 轨迹比对。另选择 AA、加权、随机、长 token 四类输入，让 k1 的
两个模式在 W4 与[冻结的旧版](../radical-region-snapshot-gate-v1/README.md)
逐条比较，共 8 组新旧二进制对照；再做 12 次 std-hasher 定向轨迹。
全部一致。

定向输入证实 k4 真正产生 T>W：`ab` 重复 129 次的 W4 输入 T=16，
snapshot 执行 67 次边界查询、7 次延迟写；长 token 跨多个 cut，
加权 W16/N14 收敛为 T=14，空输入 W16/N1 为 T=1，AA→非 AA 切换也
实际产生跨区访问。tagged 31 位域回退得到 dynamic/two-pass，轨迹仍一致。
Rust 单测另覆盖因子验证、W>N 和有序 posting 投影等不变量。

release 二进制只保存在 ignored `rust/target/reruns/radical-region-tasks-gate-v1/`，
SHA256 `da21c2f7e00e92a3d67e440476f6ed8cd85f4d10138d8d051bbe7eeada5deb19`。
这份门控只证明精确性，不据此声称性能收益。
