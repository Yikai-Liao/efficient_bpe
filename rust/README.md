# Rust BPE 训练基线

这是原项目后续研究的第一个原生基线：将已经验证的 Python `packed + lean endpoints` 核心迁到 Rust，固定规则语义、输入和统计口径，为紧凑布局及原生多线程实验提供起点。

结果见 [REPORT.md](REPORT.md)，接口契约、复杂度和下一轮实验见 [DESIGN.md](DESIGN.md)，hotpath 使用方法见 [HOTPATH.md](HOTPATH.md)。原始 Python 文件保持原样。这里实现的是 **BPE 训练核心**；预切分、文本解码、推理编码和完整 tokenizer API 不在这个基线内。

## 实现范围

- 加权相邻 pair 计数；最大频率优先，同频按 `(left_id, right_id)` 字典序选取；重叠计数，词内从左到右替换。每轮分配一个全新 ID。
- `u32` 端点语料、token ID 和长度；`u64` pair key 和频率；历史 occurrence 列表及惰性二叉堆。相邻 token 合并只写 2 或 3 个位置。
- checked 与 unchecked 两个单态化版本，算法相同。后者只在私有端点组件中省略部分数组边界检查；所有公开输入先验证，保留长度表检查及必要的不变量检查。
- 当前单线程。Rust 拥有可变语料，未来可以按完整 piece 移交给工作线程；本次尚未实现原生并行。
- 可选 hotpath 0.27.0 插桩；默认构建完全不引入该依赖。正式计时结果来自默认 release 构建。

## 运行和复现

本次实测环境是 Linux x86_64，Rust 1.98.1、CPython 3.12.13。CLI 的 CPU/RSS 测量依赖 Linux；训练库不依赖这些测量接口。以下命令从仓库根目录运行。Python 脚本仅需标准库；真实数据下载/只读快照的来源见 [共同基准说明](../benchmarks/bpe_core_comparison/README.md)。先准备其 `data/` 中的 en/zh/de/ja-1m 和 en/zh-4m 文本。

```sh
cargo test --manifest-path rust/Cargo.toml --locked
cargo clippy --manifest-path rust/Cargo.toml --all-targets --locked -- -D warnings
cargo build --manifest-path rust/Cargo.toml --release --locked

python rust/tools/fixtures.py
python rust/tools/differential.py
rust/target/release/efficient-bpe-rust \
  --input rust/fixtures/zh-1m-regex.json --bounds checked --rules 3000

# 用新输出名，保留归档结果；五次重复、随机交错、每项新进程、固定单 CPU
python rust/tools/benchmark.py --output rust/results/reruns/baseline.jsonl
```

`fixtures.py` 将已有 Python 准备流程的结果写成数字 JSON，Rust 和 Python 读取完全相同的文件；生成的较大文件不提交，提交其来源与 SHA-256 清单。`--trace path.json` 可导出全部规则及最终 token。`tools/summarize.py` 校验归档的 `results/baseline.jsonl`，从模板重新生成本轮报告与汇总。

输入格式为 `{"corpus":[0,1,2,0],"initial_lengths":[1,1,1],"pivots":[1],"weights":[2]}`。ID 0 为永久分隔符，初始 ID 稠密且每个长度为 1；非空 piece 不允许相邻分隔符；权重段从 piece 起点开始且权重为正。空语料采用 `[0]`、长度表 `[1]`、空 pivots/weights。位置、ID、长度范围及频率溢出会显式报错；当前还要求所有初始相邻边的加权总和不超过 `u64::MAX`，这一条件比单独每种 pair 不溢出更保守。

## 本地剖析

使用独立构建目录，避免覆盖正式基准的二进制：

```sh
cargo build --manifest-path rust/Cargo.toml --release --locked \
  --features profiling --target-dir rust/target/profile
HOTPATH_OUTPUT_FORMAT=json \
HOTPATH_OUTPUT_PATH=rust/results/reruns/hotpath.json \
HOTPATH_METRICS_SERVER_OFF=1 \
rust/target/profile/release/efficient-bpe-rust \
  --input rust/fixtures/zh-4m-regex.json --bounds unchecked --rules 3000
```

先创建输出目录。插桩只覆盖验证、初始化、选规则和执行规则等函数，不覆盖逐 occurrence 的数组访问。报告仅写本地；这些计时用于定位工作量，不作为未插桩吞吐数据。CPU 采样的额外依赖和编译行为见 [HOTPATH.md](HOTPATH.md)。
