# Rust 精确 BPE 主实现

本分支选定唯一 owner、分组出生链、内联 posting、aHash 的精确批量训练器为主要实现。源码在 [`src/parallel`](src/parallel/mod.rs)，命令为 `ebpe`，库入口为 `train_parallel`。Cargo 的 `default-run` 已指向 `ebpe`。选择依据、并行协议及与 2024 年原版的差异见 [算法说明](ALGORITHM.md)。

## 构建与训练

以下命令从仓库根目录执行。需要 Rust 工具链；文本准备脚本只依赖 Python 标准库。CLI 的 CPU/VmHWM 观测依赖 Linux，训练库不使用这些 Linux 观测接口。

```sh
cargo build --manifest-path rust/Cargo.toml --release --bin ebpe --locked
rust/target/release/ebpe --help
rust/target/release/ebpe --input rust/examples/tiny.json --workers 4 --rules 2

# cargo run 默认运行相同的主命令
cargo run --manifest-path rust/Cargo.toml --release --locked -- \
  --input rust/examples/tiny.json --workers 4 --rules 2

# 原文作为一个连续 piece，保留空格、标点、CRLF 和 Unicode 字符
python3 rust/tools/prepare_text.py corpus.txt --output /tmp/corpus.prepared.json
rust/target/release/ebpe --input /tmp/corpus.prepared.json \
  --workers 4 --rules 32000 --min-frequency 2 --trace /tmp/bpe-trace.json
```

`prepare_text.py` 使用 UTF-8 严格解码，不规范化文本。初始字符按 Unicode 字符顺序分配稠密 ID，永久分隔符为 0。它在准备 JSON 的 `text_metadata.symbols` 中保存 ID→字符映射及原始字节 SHA-256；训练 CLI 忽略这份附加元数据。准备内存及耗时不在训练调用计时内。空文本生成合法空语料。

## 输入和输出契约

最小输入例子就是 [`examples/tiny.json`](examples/tiny.json)：

```json
{"corpus":[0,1,2,1,2,0],"initial_lengths":[1,1,1],"pivots":[1],"weights":[3]}
```

这表示权重为 3 的 `abab`。两次合并依次选择 `(1,2)`，频率 6、fresh ID 3；再选择 `(3,3)`，频率 3、fresh ID 4；最终 token 为 `[0,4,0]`。

- ID 0 是永久分隔符。语料必须首尾为 0，非空 piece 之间以一个 0 隔开；不会跨分隔符合并。
- 初始 ID 稠密，`initial_lengths` 全为 1。长度表索引就是 token ID。
- `pivots` 从 piece 起点开始标记权重段；`weights` 长度相同、全部为正 u64，可以在多个 piece 间复用同一权重段。
- 空输入为 `corpus=[0]`、`initial_lengths=[1]`、空 pivots/weights。
- 规则优先级是加权频率降序、同频 `(left_id,right_id)` 升序；重叠 pair 计数包含所有相邻边，实际替换从左到右取不重叠出现。
- 每条规则分配 fresh ID，第一条为初始长度表长度，其后依次递增。最小频率为包含边界，即 `frequency >= min_frequency`。

位置、ID 和长度使用 u32，频率为 u64。非法布局、零权重和数值溢出返回 `TrainError`；验证还要求初始所有相邻边的加权总和不超过 u64 上限，比每种 pair 单独不过界更保守。库不将输入错误改成截断数据。

CLI 标准输出为一行 JSON，包含 `rules`、完整结果 `fingerprint`、wall/CPU、训练后 VmHWM 及工作量指标。`--trace` 输出 `{"merges":[[left,right,frequency],...],"final":[...]}`。完整字符串词表可以从准备文件的初始字符及递增 fresh ID 规则恢复；本次交付不提供 HF 模型格式或完整 tokenizer 编码/解码 API。

## 参数与资源

| 参数 | 默认 | 含义 |
|---|---|---|
| `--input` | 必填 | Prepared JSON |
| `--workers` | 可用并行度与 4 的较小值，至少 1 | 私有 Rayon pool 的 worker 数 |
| `--rules` | 32000 | 最大合并次数，不是最终词表大小 |
| `--min-frequency` | 2 | 包含边界，必须为正 |
| `--trace` | 不输出文件 | 保存全部规则和最终 token |
| `--chunk-size` | 4096 | 一个位置任务的上限，必须为正 |
| `--heap-policy` | lazy | eager 保留为研究控制 |
| `--integer-hash` | ahash | std 保留为研究控制 |

worker 数不等于整个进程的严格 CPU 预算，也不自动设置 CPU 亲和性。正式 p 核复核使用 `taskset` 把整个进程限制到 p 个 CPU；具体 CPU 编号取决于运行环境。`--workers 1` 运行同一并行算法用于正确性和扩展率对照，没有自动切换到最快标量内核。

`call_seconds` 包含输入验证、线程池、初始化、合并和收尾，不包括 JSON 读取、指纹构造及 trace 写出。`train_vm_hwm_mib` 在训练后、输出前读取，仍包含进程此前输入解析的高水位；`vm_hwm_mib` 在准备输出后读取。两者不能相减得到纯训练内存。外部进程时间覆盖整个命令。

## Rust 库入口

```rust
use efficient_bpe_rust::{Bounds, ParallelConfig, Prepared, TrainOptions, train_parallel};

fn main() -> Result<(), efficient_bpe_rust::TrainError> {
    let input = Prepared {
        corpus: vec![0, 1, 2, 1, 2, 0],
        initial_lengths: vec![1, 1, 1],
        pivots: vec![1],
        weights: vec![3],
    };
    let output = train_parallel(
        input,
        TrainOptions { bounds: Bounds::Checked, max_merges: 2, min_frequency: 2 },
        ParallelConfig { workers: 4, ..ParallelConfig::default() },
    )?;
    assert_eq!(output.final_tokens, vec![0, 4, 0]);
    Ok(())
}
```

调用消费 Prepared 并返回规则、最终 token 与 `Metrics`。`TrainOptions.bounds` 为与参考 API 共用的字段，主实现始终采用 checked atomic 端点访问；选择 Unchecked 不切换内核。`ParallelConfig` 的 hash/heap 枚举可从 `parallel` 模块导入。原 `train`、`TrainResult` 与 `efficient-bpe-rust` 命令仍保留为早期标量参考，未更改原调用契约。

## 验证与性能证据

```sh
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo test --manifest-path rust/Cargo.toml --locked
cargo clippy --manifest-path rust/Cargo.toml --all-targets --locked -- -D warnings
python3 rust/tools/verify_primary.py --output /tmp/primary-check.json

# 已有两份 16 MiB Prepared fixtures 时，追加两次完整结果指纹核对
python3 rust/tools/verify_primary.py --full-fixtures --output /tmp/primary-full-check.json
```

Python 脚本核对独立完整重算 oracle 的每条规则、频率及最终 token，也验证 CLI 默认值和原文准备。现有 fixtures 来自相同 revision 的 Wikipedia，下载/来源见 [共同基准说明](../benchmarks/bpe_core_comparison/README.md)。大文件不提交；本次提升核验记录在 [results/primary-verification.json](results/primary-verification.json)。

[16 MiB、32,000 合并复核](batch_results/radical-full-v1/README.md)是目前主选择的性能依据：owner W4 英文/中文为 2.591/1.641 秒，相对每种语言最快直接串行约 1.94×/1.63×；每格 n=2。中文 adaptive 与 chain 略快但范围交叠，保留在实验区。未取得通用 3×，没有双路或几十核证据。提升核验的单次时间只用于检查执行，不能当作新的排名。

## 历史实现与研究

主实现的维护位置为 `src/parallel`。`experiments/radical/owned_integer_hash` 保留冻结出处；其他原型和原始测量不删除。

- [当前研究状态](CURRENT_RESEARCH_STATE.md)：全部已采用/未采用方向及证据索引。
- [早期标量接口](DESIGN.md)、[原生基线测量](REPORT.md)和 [hotpath](HOTPATH.md)：旧 `train` 参考入口及其剖析。
- [全消融](ABLATION_REPORT.md)、[覆盖表](ABLATION_COVERAGE.md)、[原生并行设计](PARALLEL_DESIGN.md)及 [ablation_results](ablation_results/README.md)：布局与早期并行路径。
- [精确候选前缀](PARALLEL_RETHINK.md)、[owner 设计](POSTING_OWNER_DESIGN.md)和 [batch_results](batch_results/README.md)：后续状态归属、协议与冻结数据。
- [自适应切分和重放](ADAPTIVE_REPLAY_REPORT.md)：候选的历史小测与后续完整规模复核。

历史消融入口仍使用 `cargo run --manifest-path rust/Cargo.toml --bin ablation -- ...`，旧标量入口使用 `--bin efficient-bpe-rust`。这些入口用于对应历史实验，不改变 `ebpe` 的默认主路径。
