# 原子旧键归约正确性门控

`owned_atomic_old` 在相同二进制内比较旧键归约 `owner` 与
`producer-atomic`。最终源码通过 rustfmt、debug 库测试 17/17、
strict Clippy（全部 targets，`-D warnings`）与 release 构建。

[独立 Python naive oracle](differential.json) 的 20 个标准输入在两模式
× W1/W4、固定 aHash/lazy 下完成 80 次完整规则与最终 token 轨迹
比对。另用 std hasher 对非均匀权重和大 u64 权重各做两模式比对；
`producer-atomic` 的旧键原子调用实际大于零。请求 eager heap 时按契约
回退 `owner`，原子调用数为零，完整轨迹也匹配。共 5 次定向轨迹。
Rust 单测另覆盖并发减频、只产生一个退休标记、已选中或过滤的旧键
不参与共享扣减，以及下溢必须报错而非进入下一轮。

初次 strict Clippy 仅报私有 `route_aa` 参数过多，作者加局部 lint
标注后最终通过；训练逻辑未变。release 二进制只保存在 ignored
`rust/target/reruns/radical-atomic-old-gate-v1/`，SHA256
`8d2d8aebc774ac33147c6680de9917b0e90c8334667a3504df3791c470862039`。
门控本身不提供速度结论。
