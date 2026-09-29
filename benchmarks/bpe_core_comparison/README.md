本目录保存 efficient_bpe 与数组链、Re-Pair 半字/位图结构的独立 Python 核心重写及测试结果。主分析见 [REPORT.md](REPORT.md)，完整时空条件见 [complexity_notes.md](complexity_notes.md)。这是模块对照，不是原版 YTTM、Prezza 或 Hugging Face 的速度排行榜。

已完成 321 个隔离进程的计时运行，同一语义组的输出指纹全部一致。CPython 3.12.13、Linux x86-64、标准库即可运行，无第三方依赖。CPU 固定为当前进程允许的最后一个核；主要使用进程 CPU 时间，墙钟时间和 RSS 也保存在 JSONL。当前结论以 3 次中位数为依据。

主要代码入口：

- [共同训练驱动](python_rewrite/common.py)：加权片段、出现位置倒排、增量计数和懒堆；另有融合邻居接口和队列对照版本。
- [双端与位图/半字后端](python_rewrite/backends_compact.py)、[融合后端](python_rewrite/backends_fused.py)：双端是修正后的 v2 路线，半字复写已有 Re-Pair 机制。
- [数组链](python_rewrite/backend_linked.py)、[融合数组链](python_rewrite/backend_linked_fused.py)：未启用游程压缩，不代表完整 YTTM。
- [高低频队列与堆](python_rewrite/queues.py)：相同确定性顺序；[适用条件](python_rewrite/queues_notes.md)。
- [合法贪心长链测试](python_rewrite/chain_bench.py)：对照整段清零的平方成本。

在本目录准备解释器（也可以使用已有 CPython 3.12），再运行小语料正确性检查：

```bash
uv venv --python 3.12 .venv
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s python_rewrite -p 'test_*.py' -v
```

单项性能运行与完整融合矩阵：

```bash
PYTHONHASHSEED=0 .venv/bin/python python_rewrite/bench_fused.py --dataset en-1m --backend endpoints --rules 3000
PYTHONHASHSEED=0 .venv/bin/python python_rewrite/run_matrix.py --profile fused --repeats 3 --output-dir reruns
```

第二条命令将结果写入独立 `reruns/`，不会覆盖或混入已记录数据。同一个输出文件会追加结果，需要新的对照时选择新的输出目录。支持 profile：`real`（普通接口）、`fused`（复用上下文）、`scale`、`micro`（只测边界）、`queue` 和 `chain`。`--repeats` 可控制重复次数。

`data/` 是从另一 benchmark 的现成文件读取并复制的固定快照，所有源路径和 SHA-256 见 [snapshot-manifest.json](data/snapshot-manifest.json)。该 benchmark 源码及原数据未被修改。文本样本本地已就绪，但不纳入 Git；若需要重新建立这些特定快照，以下命令先检查全部哈希再复制，始终只读来源目录：

```bash
.venv/bin/python python_rewrite/snapshot_data.py /tmp/tokenizers-bpe-bench/data/text
```

数据来自 [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)，2023-11-01，固定 revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`。各语言 manifest 记录原 Parquet 分片与散列、确定性段落抽样和样本散列；数据沿用来源的 CC BY-SA 3.0 / GFDL 条款。来源 benchmark 的方法摘要在 [reused_benchmark_notes.md](reused_benchmark_notes.md)。

已记录结果包括：`real-results.jsonl` 75 行、`fused-results.jsonl` 45 行、`scale-results.jsonl` 45 行、`micro-results.jsonl` 84 行、`queue-results.jsonl` 24 行、`chain-results.jsonl` 48 行。运行 `.venv/bin/python summarize.py` 可检查完整记录、输入哈希与同语义输出，重建报告。它检查的是保存的完整矩阵，不会重跑性能测试。

性能边界：Python 重写会放大解释器、数组装箱和位操作成本，不能据此判断原生编译后的速度顺序。底层数组的逻辑字节数不等于进程 RSS；共同输入、词表和倒排对象必须另计。u8 微测没有计入原 v1 的字符串重建，也没有移植完整 Prezza 的 TP arena / 惰性发现新 pair 或 YTTM 的 RLE / 并行机制。这些范围在主报告中逐项说明。
