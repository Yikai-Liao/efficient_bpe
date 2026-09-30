# 自适应 cut 正确性门控

`owned_adaptive_cuts` 在同一二进制内比较 `fixed` 与 `adaptive`。
最终 lib SHA256 为
`0af4b7b2f3ef8e7c88b6029e3560cc97c3848487dc7b9b7217ad268ebe0c3cfc`。
rustfmt、debug 库测试 28/28、全部 targets 的 strict Clippy
（`-D warnings`）和 release 构建均通过。release 二进制只保存在
ignored `rust/target/reruns/radical-adaptive-cuts-gate-v1/`，SHA256
`d7ff2e3aecddb20cf295c544111dad2990fb86aa7feb9bd4dd8e7950fba2232d`。

[标准 oracle](differential.json) 用 20 个输入×两模式×W1/W4、固定
aHash/lazy、tagged-fused、atomic region k1、global posting 完成
80 次独立 naive 完整规则与最终 token 轨迹比对。另以 std hasher 执行
[14 次定向比对](directed.json)：主导 AB 表、两段不相交热 pair、W1
单区绕过、W16 多于物理位置、weighted AA→非 AA、长 token 跨 cut 与
tagged ID 域回退。全部一致。

定向计数确认两条候选构造路径均实际运行：主导 AB 产生两个最长表候选并
选中两个，累计逐批最大历史访问 552→520；不相交 AB/CD 产生一个
分层采样候选，但按实测访问不优于 fixed，故没有强行选中。W1 绕过
三批，W16 小输入有效 region 数收敛为 4。域回退有效 cut 是 fixed，
轨迹仍正确。计数的改善仅针对历史 posting 访问，不是训练墙钟或
有效改写次数的上界。

[初次编译失败](FIRST_FAILURES.md)是 `visits` 整数类型推断不明确；
作者在本 crate 标注 `usize` 后重新冻结，训练算法未改变。本门控不
提供性能结论；限定性能窗口须待 replay 也通过门控后执行。
