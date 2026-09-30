# Efficient BPE

本分支的主要实现是 Rust 精确 BPE 训练器：**唯一 owner＋分组出生链＋内联 posting＋aHash**。支持加权片段和整份连续文本，批量并行仍保持逐条 greedy 的规则、频率与最终 token。源码在 [rust/src/parallel](rust/src/parallel/mod.rs)，默认命令为 `ebpe`。

这是本轮候选比较后选定的默认。16 MiB、32,000 合并的同窗复核中，四线程英文/中文为 2.591/1.641 秒；相对各语言最快直接串行为 1.94×/1.63×，尚未达到通用 3×。中文几个候选范围交叠，不能据此宣称任意输入上的最优。[完整测量与限制](rust/batch_results/radical-full-v1/README.md)

## 使用主实现

从仓库根目录运行，需要 Rust 工具链；准备文本的脚本只用 Python 标准库。

```sh
cargo build --manifest-path rust/Cargo.toml --release --bin ebpe --locked
rust/target/release/ebpe --input rust/examples/tiny.json --rules 2 --workers 4

# 自己的 UTF-8 文本：保留空格、标点及换行，整份文件作为一个 piece
python3 rust/tools/prepare_text.py corpus.txt --output /tmp/corpus.prepared.json
rust/target/release/ebpe --input /tmp/corpus.prepared.json \
  --workers 4 --rules 32000 --min-frequency 2 --trace /tmp/bpe-trace.json
```

`--rules` 是最大合并次数，不是总词表大小。CLI 返回训练指标和完整结果指纹；`--trace` 保存全部规则及最终 token。准备文件附有字符→ID 映射。该实现提供训练核心，Python 绑定、完整 tokenizer 编码/解码和 HF 模型导出尚未迁移。

[使用与 API](rust/README.md)说明输入、默认值和验证命令；[算法与 2024 年版本对照](rust/ALGORITHM.md)说明为什么选它、精确批次协议、行为差异和性能证据。

## 当年的代码与研究记录

本仓库最初对应博客[高效中文 BPE 实现](https://lyk-ai.com/post/2)。2024 年最后提交 `7bfbc63` 的 [ebpe.py](ebpe.py)、[ebpe_v2.py](ebpe_v2.py) 和评测 notebook 保留原样；`requirements.txt` 是旧 notebook 的依赖。它们与当前 fresh-ID greedy 契约存在已记录的差异。

- [原始源码审计和文献](benchmarks/research_archive/research.md)：边缘问题、复杂度与已有算法的关系。
- [共同 Python 核心比较](benchmarks/bpe_core_comparison/REPORT.md)及[实现演进](benchmarks/bpe_core_comparison/evolution/REPORT.md)：统一语义后的布局和并行对照。
- [Rust 原生基线](rust/REPORT.md)、[全消融](rust/ABLATION_REPORT.md)和[当前研究状态](rust/CURRENT_RESEARCH_STATE.md)：历史实验、未采用候选及证据索引。

历史报告与测量原样保留。日常修改主实现使用 `rust/src/parallel`，实验原型继续放在 `rust/experiments`。
