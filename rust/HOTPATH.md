# Hotpath 本地剖析约定

核查日期：2026-09-29。依据 [hotpath-rs commit `b7247b70`](https://github.com/pawurb/hotpath-rs/tree/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667) 的源码与文档；对应 crates.io 已发布的 `hotpath = 0.27.0`。固定精确版本，避免后续小版本改变插桩和报告口径。

## 编译开关

本 crate 的 `Cargo.toml` 使用以下配置：

```toml
[dependencies]
hotpath = { version = "=0.27.0", optional = true, default-features = false }

[features]
default = []
profiling = ["dep:hotpath", "hotpath/hotpath"]
profiling-cpu = ["profiling", "hotpath/hotpath-cpu"]
```

普通构建不引入 `hotpath` 依赖或插桩。`default-features = false` 还去掉 hotpath 自带的默认 `threads` 功能；如确实需要线程报告，应显式增加 `hotpath/threads`。计时模式只开 `profiling`，不把 CPU 采样、分配跟踪、TUI、MCP、Prometheus 或 cloud 功能混入基准。参见上游的 [Cargo features](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/crates/hotpath/Cargo.toml) 和 [可选依赖示例](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/docs/src/profiling_modes.md#disabling-hotpath-entirely-with-optional-dependencies)。

同步入口和函数可用条件属性；函数式宏必须将调用点条件编译，否则普通构建找不到可选 crate：

```rust
#[cfg_attr(feature = "profiling", hotpath::main)]
fn main() {
    // ...
}

#[cfg_attr(feature = "profiling", hotpath::measure)]
fn train_rounds() {
    // ...
}

#[cfg(feature = "profiling")]
let result = hotpath::measure_block!("train", { train() });
#[cfg(not(feature = "profiling"))]
let result = train();
```

`measure_block!` 的字符串标签与 `#[measure(label = ...)]` 共享全 crate 唯一性要求。只标记阶段或较粗函数；极短热循环逐次插桩会改变待测成本。`#[hotpath::main]` 的报告在入口正常返回、guard 析构时生成，因此需要报告的路径应正常返回，避免直接 `std::process::exit`。参见 [函数与块宏](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/docs/src/functions.md) 与 [静态报告生命周期](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/docs/src/profiling_modes.md)。

## 只写本地 JSON，不启动监听

从仓库根目录运行时可用：

```bash
mkdir -p rust/results
HOTPATH_OUTPUT_FORMAT=json \
HOTPATH_OUTPUT_PATH=rust/results/hotpath.json \
HOTPATH_METRICS_SERVER_OFF=1 \
cargo run --manifest-path rust/Cargo.toml --release --features profiling \
  --target-dir rust/target/profile -- <本项目参数>
```

按实际命令行替换 `<本项目参数>`。`HOTPATH_OUTPUT_FORMAT` 默认 `table`；`HOTPATH_OUTPUT_PATH` 不设置时报告写 `stdout`，设成 `/dev/stderr` 可写标准错误。`HOTPATH_OUTPUT_FORMAT=none` **只隐藏报告，不会关闭服务器**。以上环境变量令静态报告写本地文件，并在启动时阻止 metrics HTTP 服务。没有 `HOTPATH_METRICS_SERVER_OFF` 时，已启用的 profiler 默认尝试监听 `127.0.0.1:6770`，即使没有使用 TUI。参见 [配置表](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/docs/src/configuration.md) 与 [服务实现](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/crates/hotpath/src/metrics_server.rs)。

此功能组合不启用 `hotpath-mcp`、`hotpath-prometheus`、`hotpath-cloud`。Cloud 上传另需启用 cloud feature 且设置 `HOTPATH_UPLOAD=1`；当前配置两者都没有。参见 [cloud 源码](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/crates/hotpath/src/lib_on/cloud.rs)。Cargo 首次获取依赖属于构建阶段网络行为，与程序运行时剖析报告不同。

## CPU 采样单独运行

需要调用栈与函数 CPU 样本时，另开 `--features profiling-cpu`。上游依赖外部 `samply` 0.13.x 和 `hotpath-samply` 可执行文件；可用 `HOTPATH_SAMPLY_BIN` / `HOTPATH_SAMPLY_WRAPPER_BIN` 指定位置，无需执行 hotpath 的 `init`。Linux 还需允许内核 perf/ptrace 采样，并按上游建议用 `setsid -w` 启动。给二进制保留 debug symbols；可另设继承 release 且 `debug = true` 的 profile。详见 [CPU profiling 指南](https://github.com/pawurb/hotpath-rs/blob/b7247b70aeb5cfb8f9b0046a941ed0425f9ef667/docs/src/cpu_profiling.md)。

CPU feature 会把被 `#[hotpath::measure]` 标记的函数改为 `#[inline(never)]` 以便归属样本。因此 CPU 采样版用于找热点，不与未插桩 release 版比较吞吐；`profiling` 的函数计时报告也会有插桩开销。上游提供编译期 `HOTPATH_KEEP_INLINE=1` 选项，但切换后须确保重新编译。正式吞吐矩阵仍用默认 `--release` 构建。

## 独立 API 探针

曾在仓库外的微型 crate 中，对 commit `b7247b70` 的 `0.27.0` 本地源码依赖执行普通、`profiling`、`profiling-cpu` 三种 `cargo check`，均通过；以 `profiling` 运行并设置上述三个环境变量后生成了可解析的 JSON（包含 `functions_timing`）。这验证了属性、块宏和本地输出的 API；未执行主 crate 的大型性能测试，也未验证本机 samply 的实际 CPU 采样。正式 crates.io 依赖已在本 crate 中固定为 `=0.27.0`。

## 本项目实际剖析

随后对主 crate 的 `profiling` release 构建实际运行了三个输入，均采用 unchecked、3,000 条规则、固定 CPU 5。完整命令、二进制 SHA-256、语义指纹及 CLI 统计见 [profile-runs.json](results/profile-runs.json)。以下是**有插桩的单次函数累计耗时**，不是正式性能矩阵的中位数：

| 输入 | validate | initialize | apply_rule 合计 | pop_best 合计 | train 整体 |
|---|---:|---:|---:|---:|---:|
| en-4m / regex | 3.21 ms | 37.58 ms | 131.03 ms | 6.95 ms | 188.78 ms |
| zh-4m / regex | 6.12 ms | 262.94 ms | 214.07 ms | 10.98 ms | 594.69 ms |
| en-1m / paragraph | 4.12 ms | 58.92 ms | 339.24 ms | 14.39 ms | 435.23 ms |

原始函数报告分别为 [英文 regex](results/hotpath-en-4m-regex.json)、[中文 regex](results/hotpath-zh-4m-regex.json)、[英文 paragraph](results/hotpath-en-1m-paragraph.json)。函数计时是包含内部调用的 elapsed time，`train` 与它的子函数不能相加；报告中的百分比以 `main` 为分母。`train` 还包含结果提取和局部容器释放，范围大于 CLI 的 `train_seconds` 初始化+训练循环。函数之间未归属的差额不能直接认定为某一种瓶颈。

当前粒度表明：中文建初始索引值得单独优化，英文长段落应优先看规则应用。`apply_rule` 同时包含 occurrence 检查、权重查找、hash 更新和分配；这些报告不能区分其中哪项最贵。下一轮可用 CPU 采样或更精细的独立实验验证。`pop_best` 的时间也不涵盖初始化建堆及 `apply_rule` 中新候选入堆，不能据此称整个优先队列成本可以忽略。
