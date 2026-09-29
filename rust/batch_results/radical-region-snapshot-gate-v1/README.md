# Region snapshot 正确性门控

最终 `owned_region_snapshot` 通过 rustfmt、局部边界模型 5/5、全部 debug
库测试 23/23、strict Clippy（全部 targets，`-D warnings`）及 release 构建。
局部模型覆盖六 token 描述窗口、257/513 长度跨多 cut、sentinel、延迟写
和 anchor 刷新。

[独立 Python naive oracle](differential.json) 完成 20 个标准输入 ×
`region|snapshot` × std/aHash × W1/W4，合计 160 次完整规则与最终 token
轨迹比对；另有 10 次定向及 1 次 dynamic smoke。`ab` 重复 129 次的
定向 trainer 用例在 W4 实际触发 snapshot 边界查询及远端延迟写入，
并非只有局部模型覆盖这条路径。其它定向覆盖长 token 跨 cut、加权 AA、
空区、W>N 和 tagged 域回退。全部通过。

[检查清单](checks.json)对应最终 `snapshot.rs` SHA
`3e44508b2f4de23f997e3e1f588c3844250daf3bc98defcc2aa03337c89b4efb`。
release 二进制存 ignored `rust/target/reruns/radical-region-snapshot-gate-v1/`，
其 SHA256 见检查清单。此处不声称完整训练只使用普通非原子 corpus；
snapshot 协议只作用于非 AA 的区域规划路径。
