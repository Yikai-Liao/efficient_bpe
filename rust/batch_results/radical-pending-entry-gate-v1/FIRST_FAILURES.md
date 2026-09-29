# 初次门控失败与修正

首次冻结的 `src/lib.rs` SHA256 为
`9d835086cdd19ea98dd1901d9a88808fb047c426f49d60e251415d6a51bd2549`。
`cargo fmt --check` 通过，但 `cargo test --offline --lib` 为 13/14，唯一失败：

```text
tests::pending_threshold_can_leave_owner_capacity_high_at_commit_stage
src/lib.rs:2153: assertion left == right failed
left: 192
right: 448
```

同一测试在失败前确认 B=256、E=0，combined owner 删除后公开容量为 0，
pending 归约后的公开容量至少 256。原断言要求 pending 删除前后的
`HashMap::capacity()` 恒等；删除造成 tombstone 后公开的可容纳元素总数可下降，
并不等于底层桶分配量。作者仅修改测试与 DESIGN 口径，lib SHA 变为
`9b8d9066d4d8fefcd24056de9d3712dde8d1fd764cd4d882cf8d089aa5c9bb5e`；
复测 14/14 通过。

其后 strict Clippy `--all-targets -- -D warnings` 唯一失败于测试构造：
`src/lib.rs:2103 field_reassign_with_default`。作者把 `Route` 改为结构体
字面量，训练逻辑未变；最终 lib SHA 为
`2739779436cef26795f7e6ac63bfc1710d1833eeddf33b0cc577307894ebee01`。
最终 strict Clippy、release 与 264 次完整轨迹均通过。这些是测试断言和
测试写法问题，未观察到训练轨迹错误。原始失败命令、位置及数值保存在此；
最终性能归档只对应最终源码 SHA。
