# BPE 后续迭代实验

这一目录实现并衡量几种可组合的改动：端点写入、pair key、低频索引筛选、位置池、紧凑边界、无额外逐位置内存的长长度编码，以及保持全局贪心顺序的并行训练。

结论和测量口径见 [REPORT.md](REPORT.md)。实验基于上一级已固定的 Python rewrite；原项目的 `ebpe.py`、`ebpe_v2.py` 保持原样，也没有修改另一份 tokenizer benchmark。

所有程序只依赖 Python 标准库。记录使用 CPython 3.12.13。输入沿用上一级的只读快照；下载及来源校验方式见 [上一级 README](../README.md)。

```sh
cd benchmarks/bpe_core_comparison/evolution
python -m unittest discover -s . -p 'test_*.py' -q

# 一次单进程实验
python bench.py --dataset zh-1m --variant packed --rules 3000
python bench.py --dataset zh-1m --variant arena_counted --rules 3000

# 精确并行：spawn 是每轮广播；sparse 是按 pair 所在分片派发
python parallel_bench.py --dataset en-4m --variant spawn4
python parallel_bench.py --dataset en-4m --variant sparse4

# 原始 u8 长度问题的独立边界组件；不是完整 tokenizer
python span_bench.py --length 65536
```

复现实验矩阵时，输出路径必须是新文件，以免覆盖原始结果。运行器在变体和重复之间随机交错；每项用新进程，输出规则及最终 token 指纹；同一输入若语义不一致就立即失败。单进程矩阵固定到一个 CPU；并行矩阵保持完整可用 CPU 集合。请顺序运行矩阵，避免它们彼此争用 CPU。

```sh
python run_matrix.py --profile real --variants baseline,lean,packed,filtered,arena,arena_counted,h3_fused,h25,halfword,filtered_h3 --repeats 3 --output reruns/real.jsonl
python run_matrix.py --profile edge --variants baseline,packed,filtered,arena_counted,h3_fused --repeats 3 --output reruns/edge.jsonl
python run_matrix.py --profile chain --variants baseline,packed,filtered,arena_counted,h3_fused --repeats 3 --output reruns/chain.jsonl
python run_matrix.py --profile scale --variants baseline,packed,filtered,arena_counted --repeats 3 --output reruns/scale.jsonl
python run_parallel.py --profile main --repeats 3 --output reruns/parallel.jsonl
python run_parallel.py --profile edge --variants packed,spawn1,spawn4,sparse4 --repeats 3 --output reruns/parallel-edge.jsonl
python memory_probe.py --output reruns/parallel-memory.jsonl
python verify_tests.py
python cpu_probe.py
python preprocessing_memory.py --dataset en-4m
python preprocessing_memory.py --dataset zh-4m
python summarize.py
```

每个结果文件附有 `.environment.json`，记录 Python、CPU affinity、种子、命令和源码哈希。`pilot-results.jsonl` 与 `index-pilot-results.jsonl` 是筛选方案时的探索记录，其中无关模块及部分统计字段仍在调整；正式结论使用 `real/edge/chain/scale/parallel` 矩阵。`summarize.py` 校验正式结果的三次重复、跨变体指纹和输入哈希，依据 `REPORT.template.md` 生成报告及 `summary.json`、`verification.json`。

本次冻结结果包含 63 条真实数据并行运行、24 条并行边缘运行、12 条长 span 边界微基准及 9 条独立进程树 PSS 采样。完整汇总记录为 366 个计时训练进程、35 项单元测试（全通过）和 14 组语义指纹校验；`verification.json` 保存所有正式 JSONL 的 SHA-256。`cpu-probe.json` 实测四进程 CPU 并行度约 3.60，`preprocessing-memory.json` 保存 en-4m 与 zh-4m 准备输入后的 RSS 和数据哈希。PSS 每配置只采样一次，不能和三次性能中位数混为一谈。

这些接口是研究原型的内部接口：初始 ID 对应长度 1，ID 0 是永久分隔符，新 ID 只分配一次，规则按频次最大、pair ID 字典序最小选择；相邻计数允许重叠，实际替换保持词内从左到右。频率阈值至少为 1。u32 ID/位置上限、halfword 初始字母表上限及边界接口限制见报告和各模块说明。它们没有封装成原项目的生产 API。
