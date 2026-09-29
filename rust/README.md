# Rust BPE 训练实验

这里保留第一个原生基线，并继续移植 Python 消融、探索原生索引布局和精确多线程训练。旧 `train` 接口及其结果仍可复现；新实验由 `ablation` CLI 选择具体版本。

本轮入口是 [Rust 全消融报告](ABLATION_REPORT.md)、[版本覆盖表](ABLATION_COVERAGE.md) 和 [演进设计](EVOLUTION_DESIGN.md)。连续文本并行的匹配顺序、共享写入与屏障不变量见 [PARALLEL_DESIGN.md](PARALLEL_DESIGN.md)。原始数据、环境哈希和复现矩阵见 [ablation_results](ablation_results/README.md)。

当前状态与下一步见 [研究状态索引](CURRENT_RESEARCH_STATE.md)。最新速度候选是 [grouped + inline + aHash](experiments/radical/owned_integer_hash/DESIGN.md)：有限 n=2 复核中，单核和四核相对同二进制标准哈希都改善约 1.43–1.46×；自身四核扩展比仍约英文 2.46×、中文 1.93×，并行目标尚未达成。直接串行的同哈希控制仍待完成。

最新算法演进见 [频率分片与稀疏调度报告](OWNER_PARALLEL_REPORT.md)：pair 频率/堆分片、Plan4、流水线、空间证书，以及位置与频率共用 owner 的原型。此前确定的结构基线是 [grouped + inline](experiments/radical/owned_grouped_inline/DESIGN.md)：按 key 分组传递出生位置，并在 16 字节容器中内联前两个位置。[有限 4 MiB 两次重复对照](batch_results/radical-layout-combo-v1/README.md)中，其四核耗时约为英文 0.680 秒、中文 0.404 秒，比同轮旧 owner 各下降约 20%；训练进程高水位为 94.32/100.20 MiB。相对直接串行参考为 2.47×/2.57×，自身 1→4 为 2.71×/2.10×，扩展性目标仍未达成。中文单独 inline 更省约 2 MiB，保留为内存候选。

此前 worker 局部位置索引与精确批次基础见 [连续语料批量并行报告](BATCH_PARALLEL_REPORT.md)，唯一 owner 的推导见 [POSTING_OWNER_DESIGN.md](POSTING_OWNER_DESIGN.md)。日常使用 256 KiB 轻量筛选，仅对有判别价值的候选做有限 4 MiB 测量，完整矩阵留到变体收敛后运行。原始结果见 [batch_results](batch_results/README.md)。probe 与更多逻辑分片暂不合入；counts 的物理计数和预分配由分组路线继承，单独版本的结果仍保留供对照。

此前多线程扩展不足后的跨领域研究、精确候选前缀证明和串行批宽探针见 [并行重设计](PARALLEL_RETHINK.md)。公开实现的线程曲线见 [扩展性证据](PARALLEL_SCALING_EVIDENCE.md)，下一步状态分片方案见 [双层归属设计](BATCH_OWNERSHIP_DESIGN.md)。

早期基线结果见 [REPORT.md](REPORT.md)，其接口契约见 [DESIGN.md](DESIGN.md)，hotpath 使用方法见 [HOTPATH.md](HOTPATH.md)。原始 Python 文件保持原样。这里实现的是 **BPE 训练核心**；文本解码、推理编码和完整 tokenizer API 不在计时核心内。

```sh
cargo build --manifest-path rust/Cargo.toml --release --bins --locked
python rust/tools/ablation_fixtures.py
rust/target/release/ablation --list-variants
rust/target/release/ablation --input rust/fixtures/zh-1m-regex.json \
  --variant combined_filtered --bounds unchecked --rules 3000
rust/target/release/ablation --input rust/fixtures/ablation/en-4m-continuous.json \
  --variant parallel_certified --workers 4 --bounds unchecked
# 默认轻量筛选；只在最终对比时显式添加 --full
python rust/tools/run_batch_matrix.py \
  --output-dir rust/batch_results/reruns/example
```

每次复现用新的输出目录。初始紧凑 alphabet、u32 position/ID/length 和 u64 count 范围均有显式限制；当前连续并行采用 4U 端点，尚未将所有紧凑后端并行化。外存训练器仍是设计方向。

并行 runner 默认 `--parallel-core-budget workers`：整个训练进程，包括协调线程，只允许使用 p 个 CPU，p=1 与直接串行使用同一 CPU。显式 `all` 保留旧的 worker 数实验，但这种结果不得称为严格 p 核扩展性。每次比较都同时给出同实现 1→p 和最佳直接串行的完整 `call_seconds`；完整规则、频率与最终 token 先验证一致。

新版 ablation、radical v2/v4 和 boxed CLI 提供 `train_vm_hwm_mib`，在训练返回后、构造指纹与完整轨迹 JSON 前读取进程高水位。它包含启动、输入解析及训练，适合相同输入协议下的训练内存比较；它不是单独分配器的净占用。保留的 `vm_hwm_mib` 在输出准备之后读取，可能被完整轨迹的临时内存抬高。旧归档仅有后者，不与新字段混算内存收益。

更新路径的[三项独立 Rust 实验](batch_results/radical-fused-scatter-v1/README.md)已完成：owner 本地归约后立即填充、直接扣减旧 pair 值得组合验证；大列表拆分及写入重叠尚未显示稳定净收益。七种模式通过 280 次完整 oracle，新测进程 CPU 时间帮助区分占用核数与加速比。有限长输入每格仅一次，尚不替代上述 n=2 候选结论。

后续[规划原型与组合复核](batch_results/radical-planning-integrated-v1/README.md)新增 400 次完整 oracle。有界路由缓存、小规则查询表尚未显示一致短测收益；连续 owner 提交＋直接扣减的 n=2 四核中位数为 0.673/0.415 秒，自身 2.94×/2.23×。英文控制范围与候选重叠，中文四核与原组合近乎不变且单核退化，不能据比例宣布并行突破；原 grouped+inline 继续作为主要参考。

[窄端点、缓存复用与上下文累计的小测](batch_results/radical-planning-candidates-v1/README.md)及[分离分配/淘汰相位混杂后的长输入筛选](batch_results/radical-controlled-longscreen-v1/README.md)也已保留。u16 的端点数组载荷确实为相同容量 u32 的一半，ID 超域时自动回退；总进程峰值和速度并不保证改善。上下文累计减少哈希更新、缓存复用减少重复初始化，但目前四核收益均不跨输入一致，暂不叠加为默认实现。

## 早期基线的实现范围

- 加权相邻 pair 计数；最大频率优先，同频按 `(left_id, right_id)` 字典序选取；重叠计数，词内从左到右替换。每轮分配一个全新 ID。
- `u32` 端点语料、token ID 和长度；`u64` pair key 和频率；历史 occurrence 列表及惰性二叉堆。相邻 token 合并只写 2 或 3 个位置。
- checked 与 unchecked 两个单态化版本，算法相同。后者只在私有端点组件中省略部分数组边界检查；所有公开输入先验证，保留长度表检查及必要的不变量检查。
- `train` 基线保持单线程；新 `ablation` 模块另有完整 piece 和连续 occurrence 两类原生并行实验。
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
